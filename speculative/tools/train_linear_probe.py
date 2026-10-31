#!/usr/bin/env python3
"""
Train linear probe for hot token prediction.

Input: probe_data.bin (hidden states + next tokens from rc_probe_extract)
Output: probe_weights.bin (W[n_hot × n_embd] + bias[n_hot] + hot_token_ids[n_hot])

The probe predicts: which of the 48 hot tokens comes next?
If none → returns -1 (let cascade handle it).
"""
import struct
import sys
import numpy as np
from collections import Counter


def load_probe_data(path):
    with open(path, "rb") as f:
        n_samples, n_embd = struct.unpack("<II", f.read(8))
        print(f"Loading {n_samples:,} samples, n_embd={n_embd}")

        tokens = []
        embeddings = []
        for i in range(n_samples):
            tok = struct.unpack("<i", f.read(4))[0]
            emb = np.frombuffer(f.read(n_embd * 4), dtype=np.float32).copy()
            tokens.append(tok)
            embeddings.append(emb)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{n_samples}")

    return np.array(tokens), np.array(embeddings), n_embd


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "probe_data.bin"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "probe_weights.bin"
    n_hot = int(sys.argv[3]) if len(sys.argv) > 3 else 48

    tokens, embeddings, n_embd = load_probe_data(data_path)
    n_samples = len(tokens)
    print(f"\nData: {n_samples} samples, {n_embd} dim")

    # Find top-N hot tokens
    freq = Counter(tokens)
    hot_tokens = [tok for tok, _ in freq.most_common(n_hot)]
    hot_set = set(hot_tokens)
    hot_pct = sum(freq[t] for t in hot_tokens) * 100 / n_samples
    print(f"Hot tokens: {n_hot} ({hot_pct:.1f}% of samples)")

    # Train one linear classifier per hot token
    # For each: w[n_embd] + b such that sigmoid(w·x + b) > 0.5 means "this token"
    # Use simple logistic regression via gradient descent

    # Actually, simpler: train a multi-class classifier
    # W[n_hot × n_embd], predict argmax(W × x) among hot tokens
    # Train via softmax cross-entropy

    # Even simpler for a first pass: use sklearn if available
    try:
        from sklearn.linear_model import LogisticRegression
        print("\nUsing sklearn LogisticRegression...")

        # Create labels: hot_token_index (0..47) or -1 (not hot)
        labels = np.array([hot_tokens.index(t) if t in hot_set else -1 for t in tokens])

        # Train only on samples where the next token IS a hot token
        hot_mask = labels >= 0
        X_hot = embeddings[hot_mask]
        y_hot = labels[hot_mask]
        print(f"Training on {len(X_hot):,} hot samples ({len(X_hot)*100/n_samples:.1f}%)")

        # Split train/val
        split = int(len(X_hot) * 0.8)
        X_train, X_val = X_hot[:split], X_hot[split:]
        y_train, y_val = y_hot[:split], y_hot[split:]

        clf = LogisticRegression(max_iter=200, C=1.0, solver='lbfgs',
                                 n_jobs=-1, verbose=1)
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)
        print(f"\nTrain accuracy: {train_acc*100:.1f}%")
        print(f"Val accuracy:   {val_acc*100:.1f}%")

        # Full test: on ALL samples (including non-hot)
        # Predict hot token, check if correct
        # If actual is non-hot, any prediction is wrong
        pred_all = clf.predict(embeddings)
        correct = 0
        fired = 0
        for i in range(n_samples):
            pred_tok = hot_tokens[pred_all[i]]
            fired += 1
            if pred_tok == tokens[i]:
                correct += 1
        print(f"\nFull corpus test:")
        print(f"  Accuracy (on hot predictions): {correct*100/fired:.1f}%")
        print(f"  Accept rate: {correct*100/n_samples:.1f}%")

        # With confidence threshold
        probs = clf.predict_proba(embeddings)
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            max_probs = probs.max(axis=1)
            pred_classes = probs.argmax(axis=1)
            mask = max_probs >= threshold
            n_fired = mask.sum()
            n_correct = sum(1 for i in range(n_samples)
                           if mask[i] and hot_tokens[pred_classes[i]] == tokens[i])
            acc = n_correct / n_fired if n_fired > 0 else 0
            accept = n_correct / n_samples
            print(f"  Threshold {threshold:.1f}: fired={n_fired:,} ({n_fired*100/n_samples:.1f}%) "
                  f"acc={acc*100:.1f}% accept={accept*100:.1f}%")

        # Save weights
        W = clf.coef_.astype(np.float32)       # [n_hot, n_embd]
        bias = clf.intercept_.astype(np.float32)  # [n_hot]
        hot_ids = np.array(hot_tokens, dtype=np.int32)

        with open(output_path, "wb") as f:
            f.write(struct.pack("<III", n_hot, n_embd, 0))  # header
            f.write(hot_ids.tobytes())                        # [n_hot] int32
            f.write(bias.tobytes())                           # [n_hot] float32
            f.write(W.tobytes())                              # [n_hot × n_embd] float32

        size_mb = (12 + n_hot * 4 + n_hot * 4 + n_hot * n_embd * 4) / 1e6
        print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")

    except ImportError:
        print("sklearn not available — using numpy least squares")

        # Simpler: for each hot token, compute mean embedding when it appears
        # Then at inference: find nearest centroid
        W = np.zeros((n_hot, n_embd), dtype=np.float32)
        counts = np.zeros(n_hot, dtype=np.int32)

        for i in range(n_samples):
            if tokens[i] in hot_set:
                idx = hot_tokens.index(tokens[i])
                W[idx] += embeddings[i]
                counts[idx] += 1

        for j in range(n_hot):
            if counts[j] > 0:
                W[j] /= counts[j]
                # Normalize
                norm = np.linalg.norm(W[j])
                if norm > 0:
                    W[j] /= norm

        bias = np.zeros(n_hot, dtype=np.float32)
        hot_ids = np.array(hot_tokens, dtype=np.int32)

        with open(output_path, "wb") as f:
            f.write(struct.pack("<III", n_hot, n_embd, 0))
            f.write(hot_ids.tobytes())
            f.write(bias.tobytes())
            f.write(W.tobytes())

        print(f"Saved centroid probe: {output_path}")


if __name__ == "__main__":
    main()
