#!/usr/bin/env python3
"""
Train probe on logit features (200 hot token logit values).
Zero runtime overhead — logits are always computed.

Output: logit_probe_weights.bin
  Header: [n_hot:u32, n_features:u32, reserved:u32]
  hot_ids: [int32 × n_hot]
  feature_ids: [int32 × n_features]  (same as hot_ids for now)
  W: [n_hot × n_features] float32
  bias: [n_hot] float32
"""
import struct
import sys
import numpy as np
from collections import Counter


def load_logit_data(path):
    with open(path, "rb") as f:
        n_samples, n_features = struct.unpack("<II", f.read(8))
        hot_ids = np.frombuffer(f.read(n_features * 4), dtype=np.int32).copy()
        print(f"Loading {n_samples:,} samples, {n_features} features")

        tokens = []
        features = []
        for i in range(n_samples):
            tok = struct.unpack("<i", f.read(4))[0]
            feat = np.frombuffer(f.read(n_features * 4), dtype=np.float32).copy()
            tokens.append(tok)
            features.append(feat)
            if (i + 1) % 10000 == 0:
                print(f"  {i+1}/{n_samples}")

    return np.array(tokens), np.array(features), hot_ids


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "logit_data.bin"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "logit_probe_weights.bin"

    tokens, features, hot_ids = load_logit_data(data_path)
    n_samples, n_features = features.shape
    n_hot = len(hot_ids)
    hot_set = set(hot_ids.tolist())
    hot_idx = {int(t): i for i, t in enumerate(hot_ids)}

    print(f"\nData: {n_samples:,} samples, {n_features} logit features")
    hot_pct = sum(1 for t in tokens if t in hot_set) * 100 / n_samples
    print(f"Hot tokens in next_token: {hot_pct:.1f}%")

    # Labels: hot token index or -1
    labels = np.array([hot_idx.get(int(t), -1) for t in tokens])
    hot_mask = labels >= 0

    X_hot = features[hot_mask]
    y_hot = labels[hot_mask]
    print(f"Training on {len(X_hot):,} hot samples")

    split = int(len(X_hot) * 0.8)
    X_train, X_val = X_hot[:split], X_hot[split:]
    y_train, y_val = y_hot[:split], y_hot[split:]

    # Train logistic regression
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=300, C=1.0, solver='lbfgs', verbose=1)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    print(f"\nTrain accuracy: {train_acc*100:.1f}%")
    print(f"Val accuracy:   {val_acc*100:.1f}%")

    # Full test
    pred_all = clf.predict(features)
    correct = sum(1 for i in range(n_samples) if hot_ids[pred_all[i]] == tokens[i])
    print(f"Full corpus accept: {correct}/{n_samples} = {correct*100/n_samples:.1f}%")

    # With confidence thresholds
    probs = clf.predict_proba(features)
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        max_probs = probs.max(axis=1)
        pred_classes = probs.argmax(axis=1)
        mask = max_probs >= threshold
        n_fired = mask.sum()
        n_correct = sum(1 for i in range(n_samples)
                       if mask[i] and hot_ids[pred_classes[i]] == tokens[i])
        acc = n_correct / n_fired if n_fired > 0 else 0
        print(f"  Thr {threshold:.1f}: fired={n_fired:,} ({n_fired*100/n_samples:.1f}%) "
              f"acc={acc*100:.1f}% accept={n_correct*100/n_samples:.1f}%")

    # Save weights
    W = clf.coef_.astype(np.float32)        # [n_hot, n_features]
    bias = clf.intercept_.astype(np.float32)  # [n_hot]

    with open(output_path, "wb") as f:
        f.write(struct.pack("<III", n_hot, n_features, 0))
        f.write(hot_ids.astype(np.int32).tobytes())      # hot token IDs
        f.write(hot_ids.astype(np.int32).tobytes())      # feature token IDs (same)
        f.write(W.tobytes())                               # [n_hot × n_features]
        f.write(bias.tobytes())                            # [n_hot]

    size_kb = (12 + n_hot*4*2 + n_hot*n_features*4 + n_hot*4) / 1024
    print(f"\nSaved: {output_path} ({size_kb:.0f} KB)")
    print(f"Runtime cost: {n_hot}×{n_features} = {n_hot*n_features:,} multiplications per token")


if __name__ == "__main__":
    main()
