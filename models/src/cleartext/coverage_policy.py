"""Fixed paired selection and adoption checks for candidate-pool experiments."""

import math


def check_group_separation(train_targets, train_contexts, evaluation_targets, evaluation_contexts):
    """A repaired row's target and normalized full context must both be held out."""
    if set(train_targets) & set(evaluation_targets):
        raise ValueError("repaired target group enters baseline training")
    normalize = lambda text: " ".join(text.casefold().split())
    if {normalize(text) for text in train_contexts} & {normalize(text) for text in evaluation_contexts}:
        raise ValueError("repaired context enters baseline training")


def choose(cases, scores, threshold):
    selections = {}
    edits = listed = 0
    for case in cases:
        pool = [row for row in case["candidates"] if row["guard"]
                and row["x"][0] >= .10 and scores[case["id"], row["word"]] >= threshold]
        selected = max(pool, key=lambda row: scores[case["id"], row["word"]]) if pool else None
        selections[case["id"]] = selected
        edits += selected is not None
        listed += selected is not None and bool(selected["gold"])
    return {"edits": edits, "listed": listed, "unlisted": edits - listed,
            "net": 2 * listed - edits, "precision": listed / edits if edits else 0.,
            "coverage": edits / len(cases) if cases else 0.}, selections


def adoption(repeats):
    """No log-loss claim across changed candidate populations. Require paired net +5."""
    if len(repeats) != 3 or {row["seed"] for row in repeats} != {0, 1, 2}:
        raise ValueError("require all three fixed repeats")
    delta = sum(row["default"]["repaired"]["net"] - row["default"]["baseline"]["net"]
                for row in repeats) / 3
    strict = all(math.isfinite(row["strict"]["repaired"]["precision"])
                 and row["strict"]["repaired"]["precision"] >= .90
                 and row["strict"]["repaired"]["coverage"] >= .20 for row in repeats)
    return {"mean_default_net_delta": delta, "net_bar_passed": delta >= 5,
            "strict_bar_passed_every_repeat": strict, "adoption_bar_passed": delta >= 5 and strict,
            "population_log_loss_adoption_eligible": False}
