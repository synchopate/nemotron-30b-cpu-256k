#!/usr/bin/env python3
"""
Train a probe using LOGIT values instead of hidden states.

logit[i] = W_out[i] · h, so the 200 hot token logits
ARE a 200-dim linear projection of h. Free at runtime.

We train: f(logits[hot_tokens]) → next_token
This avoids embeddings=true entirely.
"""
import struct
import sys
import numpy as np
from collections import Counter


def load_probe_data(path):
    """Load embeddings but we'll convert to logits using W_out."""
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


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "probe_data_50k.bin"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "logit_probe.bin"
    n_hot = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    tokens, embeddings, n_embd = load_probe_data(data_path)
    n_samples = len(tokens)
    print(f"Data: {n_samples} samples, {n_embd} dim")

    # Find hot tokens
    freq = Counter(tokens)
    hot_tokens = [tok for tok, _ in freq.most_common(n_hot)]
    hot_set = set(hot_tokens)
    hot_idx = {t: i for i, t in enumerate(hot_tokens)}
    print(f"Hot tokens: {n_hot} ({sum(freq[t] for t in hot_tokens)*100/n_samples:.1f}%)")

    # We don't have W_out (model weights). But we have h (embeddings).
    # The probe on h gives 84.7%. Can we find which DIMENSIONS of h
    # matter most and build a low-rank approximation?
    #
    # Actually — the better approach: at extraction time, save the LOGITS
    # not the embeddings. Then train directly on logits.
    #
    # For now with existing data: train on embeddings but produce a
    # COMPACT probe that can work with fewer features.
    # We'll create the logit-based extractor next.

    # Step 1: PCA to find top-200 dimensions of h
    print("\nPCA on embeddings...")
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(200, n_embd), random_state=42)
    X_pca = pca.fit_transform(embeddings)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  Top-{pca.n_components} components explain {var_explained*100:.1f}% variance")

    # Step 2: Train probe on PCA features
    from sklearn.linear_model import LogisticRegression

    # Labels: hot token index or -1
    labels = np.array([hot_idx.get(t, -1) for t in tokens])
    hot_mask = labels >= 0
    X_hot = X_pca[hot_mask]
    y_hot = labels[hot_mask]
    print(f"Training on {len(X_hot):,} hot samples")

    split = int(len(X_hot) * 0.8)
    X_train, X_val = X_hot[:split], X_hot[split:]
    y_train, y_val = y_hot[:split], y_hot[split:]

    # Train with different numbers of PCA components
    for n_comp in [20, 50, 100, 200]:
        if n_comp > pca.n_components:
            continue
        clf = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs', verbose=0)
        clf.fit(X_train[:, :n_comp], y_train)
        val_acc = clf.score(X_val[:, :n_comp], y_val)

        # Full test
        pred_all = clf.predict(X_pca[:, :n_comp])
        correct = sum(1 for i in range(n_samples) if hot_tokens[pred_all[i]] == tokens[i])
        accept = correct / n_samples

        print(f"  PCA-{n_comp:3d}: val_acc={val_acc*100:.1f}%  accept={accept*100:.1f}%")

    # Final: train with 100 components (sweet spot)
    n_final = min(100, pca.n_components)
    clf = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs')
    clf.fit(X_hot[:, :n_final], y_hot)

    # The probe is: PCA_transform(h)[:100] → LogReg → hot_token
    # At runtime we need: PCA components (100 × 2048) + LogReg weights (200 × 100)
    # Total: 100*2048 + 200*100 = 204,800 + 20,000 = 225K floats = 900 KB

    # But WAIT — PCA(h) requires h which requires embeddings=true
    # The whole point was to avoid this.
    #
    # Real solution: extract LOGITS at runtime, not h.
    # We need a new extractor that saves logits.

    print(f"\n{'='*60}")
    print(f"CONCLUSION: PCA-100 on h gives good results but still needs h.")
    print(f"The RIGHT approach: save logits during extraction, train on logits.")
    print(f"Creating logit extractor now...")
    print(f"{'='*60}")

    # Alternative: the model's own logits for hot tokens ARE features.
    # logit[hot_i] = W_out[hot_i] · h
    # So logits[hot_tokens] is a n_hot-dim projection of h.
    # If we had these logits, we could train directly on them.
    #
    # Since we have h, we can SIMULATE this:
    # We need W_out rows for hot tokens. We don't have W_out.
    # But we can approximate: train a linear map from h → "simulated logits"
    # using the actual next-token as supervision.
    #
    # Actually, even simpler: h contains all the info. The model computes
    # logits = W_out · h. If we had the top-200 logits, they'd be a
    # linear transform of h. Our PCA-100 captures the most important
    # 100 linear combinations of h. It's essentially the same thing.
    #
    # KEY INSIGHT: Instead of extracting embeddings at runtime,
    # extract the TOP-200 LOGIT values and train on those.
    # Logits are ALWAYS available — no embeddings=true needed!

    print(f"\nBut we can use a simpler approach:")
    print(f"At each model call, we already have ALL logits.")
    print(f"Just read logits[hot_token_id] for each hot token.")
    print(f"Train classifier: logits_of_200_hot → which comes next.")
    print(f"This is what rc_probe_extract_logits should do.")


if __name__ == "__main__":
    main()
