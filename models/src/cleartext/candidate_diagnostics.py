"""Inspect frozen candidate decisions without fitting or changing acceptance."""

import math


STAGES = (
    "generation",
    "guard",
    "meaning_floor",
    "probability",
    "ranking",
    "selected_listed",
)


def diagnose_case(case, probabilities, threshold=0.45, floor=0.10):
    """Partition listed-substitute misses in the order of the cached policy.

    This uses benchmark list membership, not a judgment of sentence meaning.
    Ties preserve generator order, as in the development evaluator.
    """
    candidates = case["candidates"]
    words = [row["word"] for row in candidates]
    if len(set(words)) != len(words) or set(probabilities) != set(words):
        raise ValueError("candidate words and probability keys must match uniquely")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("probabilities must be finite and between zero and one")
    if not 0 <= threshold <= 1 or not 0 <= floor <= 1:
        raise ValueError("threshold and floor must be between zero and one")
    listed = [row for row in candidates if row["gold"]]
    guarded = [row for row in candidates if row["guard"]]
    plausible = [row for row in guarded if row["x"][0] >= floor]
    eligible = [row for row in plausible if probabilities[row["word"]] >= threshold]
    stages = [
        ("generation", listed),
        ("guard", [row for row in guarded if row["gold"]]),
        ("meaning_floor", [row for row in plausible if row["gold"]]),
        ("probability", [row for row in eligible if row["gold"]]),
    ]
    selected = (
        max(eligible, key=lambda row: probabilities[row["word"]]) if eligible else None
    )
    stage = next(
        (name for name, pool in stages if not pool),
        "selected_listed" if selected and selected["gold"] else "ranking",
    )
    best_listed = (
        max(stages[2][1], key=lambda row: probabilities[row["word"]])
        if stages[2][1]
        else None
    )
    return {
        "stage": stage,
        "outcome": "abstained"
        if selected is None
        else "listed_edit"
        if selected["gold"]
        else "unlisted_edit",
        "selected": None
        if selected is None
        else {
            "word": selected["word"],
            "listed_match": bool(selected["gold"]),
            "probability": float(probabilities[selected["word"]]),
        },
        "listed_pool_counts": {name: len(pool) for name, pool in stages},
        "best_eligible_listed": None
        if best_listed is None
        else {
            "word": best_listed["word"],
            "probability": float(probabilities[best_listed["word"]]),
        },
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
    }


def review_shortlist(proposals, limit=3):
    """Choose by cached BERT rank only. Never read benchmark membership."""
    if type(limit) is not int or not 1 <= limit <= 5:
        raise ValueError("review limit must be between one and five")
    seen, result = set(), []
    for candidate in sorted(proposals, key=lambda row: row["bert_rank"]):
        key = candidate["word"].casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                field: candidate[field]
                for field in (
                    "word",
                    "lemma",
                    "source",
                    "senses",
                    "bert_rank",
                    "bert_probability",
                )
            }
        )
        if len(result) == limit:
            break
    return result
