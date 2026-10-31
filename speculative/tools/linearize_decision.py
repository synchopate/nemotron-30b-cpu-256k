#!/usr/bin/env python3
"""
Linearizzazione della funzione decisionale per i top-48 token.

Domanda: per ogni hot token, quanta informazione porta ogni
posizione del contesto? Se t-1 da solo spiega l'80% delle decisioni,
la funzione è quasi lineare in una dimensione.

Metrica: mutual information I(target | t-k) per k=1..6
Se I(target|t-1) >> I(target|t-2) → la decisione è "locale"
Se I distribuita → servono interazioni → non linearizzabile

Poi: logistic regression con features sparse (token IDs one-hot)
per misurare accuracy di un modello lineare.
"""
import struct
import sys
import math
from collections import Counter, defaultdict


def load_corpus(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data)//4}i", data))


def mutual_info(joint_counts, x_counts, y_counts, total):
    """I(X;Y) from count tables"""
    mi = 0.0
    for (x, y), c_xy in joint_counts.items():
        if c_xy == 0: continue
        p_xy = c_xy / total
        p_x = x_counts[x] / total
        p_y = y_counts[y] / total
        if p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return mi


def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else "corpus_ds_train.bin"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "corpus_ds_test.bin"

    train = load_corpus(train_path)
    test = load_corpus(test_path)
    print(f"Train: {len(train):,}  Test: {len(test):,}")

    freq = Counter(train)
    top48 = [tok for tok, _ in freq.most_common(48)]
    top48_set = set(top48)

    # Try tokenizer
    tok_names = {}
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True)
        for t in top48:
            try: tok_names[t] = repr(tokenizer.decode([t]))[:15]
            except: tok_names[t] = str(t)
    except:
        for t in top48:
            tok_names[t] = str(t)

    # ══════════════════════════════════════════════════════
    # MUTUAL INFORMATION: I(next_is_hot_token | t-k)
    # Per ogni hot token, per ogni depth k:
    # quanta informazione porta t-k sulla decisione?
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"MUTUAL INFORMATION: I(is_target | t-k) in bits")
    print(f"Quanta informazione porta ogni posizione sulla decisione")
    print(f"{'='*90}")

    print(f"\n  {'Token':<16s}", end="")
    for k in range(1, 7):
        print(f" {'I(t-'+str(k)+')':>8s}", end="")
    print(f" {'sum':>8s} {'%t-1':>6s} {'type':>12s}")
    print(f"  {'-'*90}")

    type_counts = Counter()
    token_types = {}

    for tok in top48:
        p_target = freq[tok] / len(train)

        mi_values = []
        for k in range(1, 7):
            # Joint: (t-k value, is_target)
            joint = Counter()
            x_cnt = Counter()  # t-k values
            y_cnt = Counter()  # is_target
            n = 0
            for j in range(6, len(train)):
                is_t = 1 if train[j] == tok else 0
                prev_k = train[j-k]
                joint[(prev_k, is_t)] += 1
                x_cnt[prev_k] += 1
                y_cnt[is_t] += 1
                n += 1

            mi = mutual_info(joint, x_cnt, y_cnt, n)
            mi_values.append(mi)

        total_mi = sum(mi_values)
        pct_t1 = mi_values[0] / total_mi * 100 if total_mi > 0 else 0

        # Classify decision function type
        if pct_t1 > 70:
            dtype = "LINEAR-1"    # t-1 alone explains 70%+
        elif pct_t1 > 50:
            dtype = "LINEAR-2"    # mostly t-1 but some t-2
        elif mi_values[0] > mi_values[1] > mi_values[2]:
            dtype = "DECAY"       # info decreases with distance
        else:
            dtype = "COMPLEX"     # non-local dependencies

        type_counts[dtype] += 1
        token_types[tok] = dtype

        name = tok_names.get(tok, str(tok))
        print(f"  {name:<16s}", end="")
        for mi in mi_values:
            print(f" {mi:8.4f}", end="")
        print(f" {total_mi:8.4f} {pct_t1:5.1f}% {dtype:>12s}")

    print(f"\n  Tipo distribuzione:")
    for dtype, cnt in type_counts.most_common():
        toks = [t for t in top48 if token_types[t] == dtype]
        occ = sum(freq[t] for t in toks)
        print(f"    {dtype:<12s}: {cnt:2d} token, {occ:,} occ ({occ*100/len(train):.1f}% output)")

    # ══════════════════════════════════════════════════════
    # LOGISTIC REGRESSION: modello lineare su features sparse
    # Feature: one-hot del token a t-1 (+ opzionalmente t-2)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"LOGISTIC REGRESSION EQUIVALENTE (lookup table test)")
    print(f"Per ogni hot token: accuratezza della regola 'if t-1 in SET → predict tok'")
    print(f"{'='*90}")

    # Per ogni hot token, il "modello lineare" ottimale è:
    # P(tok | t-1) = rate_table[t-1]
    # Threshold: predict tok if P > 0.5
    # Questo È un classificatore lineare in feature space one-hot

    print(f"\n  {'Token':<16s} {'P(tok)':>6s} {'#rules':>7s} {'Recall':>8s} "
          f"{'Prec':>8s} {'F1':>6s} {'Cov':>6s}")
    print(f"  {'-'*70}")

    # Build P(tok | t-1) from train
    follows_count = defaultdict(Counter)  # t-1 → Counter(next)
    for j in range(1, len(train)):
        follows_count[train[j-1]][train[j]] += 1

    total_recall = 0
    total_precision_weighted = 0
    total_coverage = 0
    total_hot_occ = 0

    for tok in top48:
        p_tok = freq[tok] / len(train)

        # Find all t-1 where P(tok|t-1) > 0.5
        rules = {}
        for prev, foll in follows_count.items():
            total = sum(foll.values())
            rate = foll.get(tok, 0) / total
            if rate > 0.5:
                rules[prev] = rate

        # Test on test set
        tp = 0  # predicted tok, actually tok
        fp = 0  # predicted tok, actually not tok
        fn = 0  # not predicted, actually tok
        tn = 0

        test_tok_count = 0
        for j in range(1, len(test)):
            is_tok = test[j] == tok
            predicted = test[j-1] in rules

            if predicted and is_tok: tp += 1
            elif predicted and not is_tok: fp += 1
            elif not predicted and is_tok: fn += 1
            if is_tok: test_tok_count += 1

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        coverage = (tp + fp) / len(test)

        total_recall += tp
        total_hot_occ += test_tok_count

        name = tok_names.get(tok, str(tok))
        print(f"  {name:<16s} {p_tok*100:5.1f}% {len(rules):7d} {recall*100:7.1f}% "
              f"{precision*100:7.1f}% {f1*100:5.1f}% {coverage*100:5.1f}%")

    print(f"\n  Total hot token nel test: {total_hot_occ:,}")
    print(f"  Recall totale (linear rules): {total_recall:,} ({total_recall*100/total_hot_occ:.1f}%)")

    # ══════════════════════════════════════════════════════
    # LA FUNZIONE SEMPLIFICATA: per ogni posizione,
    # il classificatore lineare "if t-1 in SET_k → emit tok_k"
    # copre quanti token corretti?
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"FUNZIONE SEMPLIFICATA: predici con regole lineari, fallback cascade")
    print(f"{'='*90}")

    # Build rule engine: predecessor → hot_token (highest P wins)
    rule_engine = {}  # predecessor → (hot_token, P(hot|pred))
    for tok in top48:
        for prev, foll in follows_count.items():
            total = sum(foll.values())
            rate = foll.get(tok, 0) / total
            if rate > 0.5:
                if prev not in rule_engine or rate > rule_engine[prev][1]:
                    rule_engine[prev] = (tok, rate)

    print(f"\n  Rule engine: {len(rule_engine):,} regole (predecessor → hot token)")

    # Test: rule engine only
    re_correct = 0
    re_fired = 0
    for j in range(1, len(test)):
        if test[j-1] in rule_engine:
            re_fired += 1
            if rule_engine[test[j-1]][0] == test[j]:
                re_correct += 1

    n_test = len(test) - 1
    print(f"  Rule engine solo:  cov={re_fired*100/n_test:.1f}%  "
          f"acc={re_correct*100/re_fired:.1f}%  accept={re_correct*100/n_test:.1f}%")

    # Build cascade for comparison and fallback
    cascade = {}
    for k in [6, 5, 4, 3, 2, 1]:
        raw = defaultdict(Counter)
        for j in range(k, len(train)):
            raw[tuple(train[j-k:j])][train[j]] += 1
        for ctx, cts in raw.items():
            total = sum(cts.values())
            if total < 2: continue
            best, bcnt = cts.most_common(1)[0]
            h = 14695981039346656037
            for t in ctx:
                h ^= (int(t) & 0xFFFFFFFF)
                h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            if h not in cascade:
                cascade[h] = (best, bcnt * 100 // total, k)

    def cascade_predict(context, min_k=1):
        for k in [6, 5, 4, 3, 2, min_k]:
            if len(context) < k: continue
            h = 14695981039346656037
            for t in context[-k:]:
                h ^= (int(t) & 0xFFFFFFFF)
                h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            if h in cascade:
                return cascade[h]
        return None

    # Test: cascade only
    casc_correct = 0
    casc_fired = 0
    for j in range(6, len(test)):
        r = cascade_predict(test[:j])
        if r:
            casc_fired += 1
            if r[0] == test[j]:
                casc_correct += 1

    print(f"  Cascade solo:      cov={casc_fired*100/n_test:.1f}%  "
          f"acc={casc_correct*100/casc_fired:.1f}%  accept={casc_correct*100/n_test:.1f}%")

    # Test: rule engine FIRST, cascade fallback
    combo_correct = 0
    combo_fired = 0
    rule_hits = 0
    rule_correct_hits = 0
    casc_hits = 0
    casc_correct_hits = 0

    for j in range(6, len(test)):
        pred = None
        source = None

        # Rule engine first
        if test[j-1] in rule_engine:
            pred = rule_engine[test[j-1]][0]
            source = "rule"
        else:
            # Cascade fallback
            r = cascade_predict(test[:j])
            if r:
                pred = r[0]
                source = "cascade"

        if pred is not None:
            combo_fired += 1
            if pred == test[j]:
                combo_correct += 1
            if source == "rule":
                rule_hits += 1
                if pred == test[j]: rule_correct_hits += 1
            else:
                casc_hits += 1
                if pred == test[j]: casc_correct_hits += 1

    n2 = len(test) - 6
    print(f"  Rules → Cascade:   cov={combo_fired*100/n2:.1f}%  "
          f"acc={combo_correct*100/combo_fired:.1f}%  accept={combo_correct*100/n2:.1f}%")
    print(f"    Rules:   {rule_hits:,} fired, {rule_correct_hits:,} correct ({rule_correct_hits*100/rule_hits:.1f}%)")
    print(f"    Cascade: {casc_hits:,} fired, {casc_correct_hits:,} correct ({casc_correct_hits*100/casc_hits:.1f}%)")
    print(f"    Delta vs cascade solo: {(combo_correct - casc_correct):+,d} token")

    # ══════════════════════════════════════════════════════
    # GENERALIZZABILITÀ: queste regole sono universali?
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"UNIVERSALITÀ: queste regole sono strutturali?")
    print(f"{'='*90}")

    structural_rules = 0
    semantic_rules = 0
    for prev, (hot_tok, rate) in rule_engine.items():
        # Is this a structural rule? (punctuation/whitespace/brackets)
        hot_name = tok_names.get(hot_tok, "")
        # Structural tokens: \n, space, indent, brackets, punctuation
        if hot_tok in {185, 207, 300, 391, 595}:  # whitespace/indent
            structural_rules += 1
        elif hot_tok in {13, 11, 25, 7, 8, 26, 29, 62}:  # punctuation
            structural_rules += 1
        elif hot_tok in {10897, 509, 403}:  # code structure
            structural_rules += 1
        else:
            semantic_rules += 1

    print(f"\n  Regole strutturali (whitespace/punct/code): {structural_rules:,}")
    print(f"  Regole semantiche (parole/keyword):          {semantic_rules:,}")
    print(f"  Ratio: {structural_rules*100/(structural_rules+semantic_rules):.0f}% strutturali")
    print(f"\n  Le regole strutturali sono UNIVERSALI cross-model perché")
    print(f"  dipendono dalla sintassi del linguaggio, non dal modello.")
    print(f"  Un newline dopo '{{' è uguale per GPT, Llama, DeepSeek, Qwen.")


if __name__ == "__main__":
    main()
