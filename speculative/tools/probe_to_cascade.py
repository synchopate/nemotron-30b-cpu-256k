#!/usr/bin/env python3
"""
Distill linear probe predictions into a cascade table.

The probe sees hidden states (semantic info). We run it on
the training corpus and store its predictions as hash entries.
At runtime: pure lookup, zero model cost.

This creates entries where the PROBE disagrees with the cascade —
places where semantic info improves over n-gram statistics.
"""
import struct
import sys
import numpy as np
from collections import Counter, defaultdict


def fnv64(tokens):
    h = 14695981039346656037
    for t in tokens:
        h ^= (int(t) & 0xFFFFFFFF)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def load_probe_data(path):
    with open(path, "rb") as f:
        n_samples, n_embd = struct.unpack("<II", f.read(8))
        tokens = []
        embeddings = []
        for i in range(n_samples):
            tok = struct.unpack("<i", f.read(4))[0]
            emb = np.frombuffer(f.read(n_embd * 4), dtype=np.float32).copy()
            tokens.append(tok)
            embeddings.append(emb)
    return np.array(tokens), np.array(embeddings), n_embd


def load_probe_weights(path):
    with open(path, "rb") as f:
        n_hot, n_embd, _ = struct.unpack("<III", f.read(12))
        hot_ids = np.frombuffer(f.read(n_hot * 4), dtype=np.int32).copy()
        bias = np.frombuffer(f.read(n_hot * 4), dtype=np.float32).copy()
        W = np.frombuffer(f.read(n_hot * n_embd * 4), dtype=np.float32).copy().reshape(n_hot, n_embd)
    return hot_ids, bias, W


def load_corpus(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//4}i", data))


def main():
    probe_data_path = sys.argv[1] if len(sys.argv) > 1 else "probe_data.bin"
    probe_weights_path = sys.argv[2] if len(sys.argv) > 2 else "probe_weights.bin"
    corpus_path = sys.argv[3] if len(sys.argv) > 3 else "corpus_ds_train.bin"
    output_path = sys.argv[4] if len(sys.argv) > 4 else "hc_probe.bin"

    # Load probe
    hot_ids, bias, W = load_probe_weights(probe_weights_path)
    n_hot, n_embd = W.shape
    print(f"Probe: {n_hot} hot tokens, {n_embd} dim")

    # Load probe data (embeddings + actual tokens)
    actual_tokens, embeddings, _ = load_probe_data(probe_data_path)
    n_samples = len(actual_tokens)
    print(f"Probe data: {n_samples} samples")

    # Load corpus (for context tokens at each position)
    corpus = load_corpus(corpus_path)
    print(f"Corpus: {len(corpus):,} tokens")

    # Run probe on all samples
    scores = embeddings @ W.T + bias  # [n_samples, n_hot]
    probe_preds = hot_ids[scores.argmax(axis=1)]  # [n_samples]
    probe_confs = np.exp(scores.max(axis=1)) / np.exp(scores).sum(axis=1)  # softmax confidence

    # Accuracy check
    correct = (probe_preds == actual_tokens).sum()
    print(f"Probe accuracy on data: {correct}/{n_samples} = {correct*100/n_samples:.1f}%")

    # Build cascade entries from probe predictions
    # For each position, hash the context (k=2,3,4) and store probe prediction
    # Only store where probe confidence > threshold
    print(f"\nBuilding probe cascade entries...")

    for k in [2, 3, 4]:
        entries = {}  # hash → (prediction, confidence_count, total_count)

        for i in range(n_samples):
            pos = i  # position in corpus
            if pos < k:
                continue

            ctx = corpus[pos - k:pos]
            h = fnv64(ctx)
            pred = int(probe_preds[i])
            conf = float(probe_confs[i])

            if h not in entries:
                entries[h] = defaultdict(int)
            entries[h][pred] += 1

        # Convert to cascade format: keep most common prediction per hash
        cascade_entries = []
        for h, pred_counts in entries.items():
            total = sum(pred_counts.values())
            if total < 2:
                continue
            best_pred = max(pred_counts, key=pred_counts.get)
            best_conf = pred_counts[best_pred] * 100 // total
            cascade_entries.append((h, best_pred, best_conf))

        cascade_entries.sort(key=lambda e: e[0])

        # Write
        fname = f"hc_probe_k{k}.bin"
        with open(fname, "wb") as f:
            f.write(struct.pack("<II", len(cascade_entries), k))
            for h, pred, conf in cascade_entries:
                f.write(struct.pack("<QII", h, pred, conf))

        print(f"  k={k}: {len(cascade_entries):,} entries → {fname}")

    # Also build the FULL probe cascade: for each position,
    # compare probe vs standard n-gram, store probe prediction
    # where it differs AND the probe is confident
    print(f"\n--- Probe vs N-gram comparison ---")

    # Build standard n-gram for comparison
    ngram = defaultdict(Counter)
    for j in range(3, len(corpus)):
        ngram[tuple(corpus[j-3:j])][corpus[j]] += 1
    ngram_best = {}
    for ctx, cts in ngram.items():
        total = sum(cts.values())
        if total < 2: continue
        best, bcnt = cts.most_common(1)[0]
        ngram_best[fnv64(list(ctx))] = best

    # Count: where probe and ngram differ, who's right?
    probe_right_ngram_wrong = 0
    probe_wrong_ngram_right = 0
    both_right = 0
    both_wrong = 0
    probe_only = 0
    ngram_only = 0

    for i in range(n_samples):
        pos = i
        if pos < 3: continue

        actual = actual_tokens[i]
        probe_pred = int(probe_preds[i])

        h = fnv64(corpus[pos-3:pos])
        ngram_pred = ngram_best.get(h, -1)

        if ngram_pred < 0:
            if probe_pred == actual:
                probe_only += 1
            continue

        if probe_pred == actual and ngram_pred == actual:
            both_right += 1
        elif probe_pred == actual and ngram_pred != actual:
            probe_right_ngram_wrong += 1
        elif probe_pred != actual and ngram_pred == actual:
            probe_wrong_ngram_right += 1
        else:
            both_wrong += 1

    total_compared = both_right + probe_right_ngram_wrong + probe_wrong_ngram_right + both_wrong
    print(f"  Both right:           {both_right:,}")
    print(f"  Probe RIGHT, ngram wrong: {probe_right_ngram_wrong:,}  ← probe value")
    print(f"  Probe wrong, ngram RIGHT: {probe_wrong_ngram_right:,}  ← probe cost")
    print(f"  Both wrong:           {both_wrong:,}")
    print(f"  Probe only (no ngram): {probe_only:,}")
    print(f"  Net probe gain: {probe_right_ngram_wrong - probe_wrong_ngram_right:+,d}")

    # Speedup estimate
    # Replace ngram errors with probe corrections
    ngram_correct = both_right + probe_wrong_ngram_right
    probe_corrected = both_right + probe_right_ngram_wrong
    print(f"\n  N-gram accuracy: {ngram_correct*100/total_compared:.1f}%")
    print(f"  If probe overrides where it disagrees: {probe_corrected*100/total_compared:.1f}%")


if __name__ == "__main__":
    main()
