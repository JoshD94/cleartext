"""Module-level metrics for the ClearText evaluation pipeline.

Covers two parts of Section 3.1 of the proposal:
  - Word-complexity / jargon detection: precision, recall, F1 against
    human labels (e.g. CWI 2018 annotations).
  - Lexical simplification ranking: how highly the system ranks good
    simpler substitutions (accuracy@k, MRR), the standard style of
    evaluation for benchmarks like TSAR-2022 / MultiLS.
"""

from sklearn.metrics import precision_recall_fscore_support


def detection_metrics(y_true, y_pred, labels=(0, 1)):
    """Precision, recall, F1 for the jargon/complexity detector.

    y_true / y_pred are aligned label sequences (1 = complex/jargon).
    Returns per-class scores plus macro averages.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    out = {}
    for i, lab in enumerate(labels):
        out[f"class_{lab}"] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    out["macro"] = {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }
    return out


def accuracy_at_k(ranked_candidates, gold_sets, k=1):
    """Fraction of instances where a gold substitution appears in the top-k
    ranked candidates. `ranked_candidates[i]` is an ordered list of strings;
    `gold_sets[i]` is a set of acceptable gold substitutions (lowercased)."""
    if not ranked_candidates:
        return 0.0
    hits = 0
    for ranked, gold in zip(ranked_candidates, gold_sets):
        topk = {c.lower() for c in ranked[:k]}
        if topk & {g.lower() for g in gold}:
            hits += 1
    return hits / len(ranked_candidates)


def mean_reciprocal_rank(ranked_candidates, gold_sets):
    """Mean reciprocal rank of the first gold substitution in each ranking."""
    if not ranked_candidates:
        return 0.0
    total = 0.0
    for ranked, gold in zip(ranked_candidates, gold_sets):
        gold_l = {g.lower() for g in gold}
        rr = 0.0
        for rank, cand in enumerate(ranked, start=1):
            if cand.lower() in gold_l:
                rr = 1.0 / rank
                break
        total += rr
    return total / len(ranked_candidates)


def ranking_report(ranked_candidates, gold_sets, ks=(1, 3, 5)):
    """Accuracy@k for several k plus MRR, for lexical-simplification ranking."""
    report = {f"accuracy@{k}": accuracy_at_k(ranked_candidates, gold_sets, k) for k in ks}
    report["mrr"] = mean_reciprocal_rank(ranked_candidates, gold_sets)
    return report
