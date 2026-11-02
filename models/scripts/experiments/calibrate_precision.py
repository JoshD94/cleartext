"""Select an opt-in edit threshold on grouped development predictions.

This targets listed-substitute precision, never full-sentence accuracy. The
reused TSAR test is read only after the development policy is frozen.
"""

import argparse
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import cleartext.refine as R  # noqa: E402
from cleartext.data import ROOT  # noqa: E402


def wilson(correct, n):
    if not n:
        return [0.0, 1.0]
    z, p = 1.959963984540054, correct / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [center - half, center + half]


def evaluate(cases, probabilities, threshold, floor):
    edits = correct = 0
    for case in cases:
        ok = [c for c in case["candidates"]
              if c["guard"] and probabilities[case["id"], c["word"]] >= threshold
              and c["x"][0] >= floor]
        # sense_fit is resolved by the caller's cached feature order.
        if ok:
            chosen = max(ok, key=lambda c: probabilities[case["id"], c["word"]])
            edits += 1
            correct += bool(chosen["gold"])
    return {"n": len(cases), "edits": edits, "correct": correct,
            "precision": correct / edits if edits else None,
            "coverage": edits / len(cases), "wilson95": wilson(correct, edits)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    # Deliberately do not call R.tables(), which also opens TSAR.
    cache = ROOT / "data/cache"
    names, dev = pickle.load((cache / "refine-v7-benchls-dev.pkl").open("rb"))
    names2, hold = pickle.load((cache / "refine-v7-benchls-holdout.pkl").open("rb"))
    if names != names2 or names.index("sense_fit") != 0:
        raise ValueError("Expected cached v7 sense_fit as first member")
    dev += hold
    X, y, groups, keys = R.flat_specific(dev)
    threshold_grid = [float(v) for v in np.round(np.arange(0.45, 0.976, 0.025), 3)]
    protocol = {
        "target_precision": 0.90, "minimum_coverage_each_repeat": 0.20,
        "selection": "Lowest threshold achieving target precision and minimum coverage in every repeat; otherwise highest minimum precision among eligible thresholds.",
        "folds": 10, "repeats": 3, "seeds": [0, 1, 2], "threshold_grid": threshold_grid,
        "data": "All 929 BenchLS cases are reused development, grouped by target word. TSAR reused exploratory test, opened only after policy freeze.",
        "model": "Same AverageDecision family and cached v7 features, no architecture or weight changes to deployed models",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cache_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [cache / "refine-v7-benchls-dev.pkl", cache / "refine-v7-benchls-holdout.pkl"]},
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    predictions = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        unique = np.unique(groups)
        mapping = dict(zip(unique, rng.permutation(len(unique)) % 10))
        folds = np.array([mapping[group] for group in groups])
        probabilities = np.zeros(len(y))
        for fold in range(10):
            train, valid = folds != fold, folds == fold
            model = R.AverageDecision().fit(X[train], y[train])
            probabilities[valid] = model.predict_proba(X[valid])[:, 1]
        predictions.append(dict(zip(keys, probabilities)))
        print(f"completed grouped repeat {seed + 1}/3", flush=True)
    grid = [{"threshold": t, "repeats": [evaluate(dev, p, t, 0.1) for p in predictions]} for t in threshold_grid]
    eligible = [r for r in grid if all(m["coverage"] >= .2 for m in r["repeats"])]
    passing = [r for r in eligible if all(m["precision"] >= .9 for m in r["repeats"])]
    selected = passing[0] if passing else (max(eligible, key=lambda r: min(m["precision"] for m in r["repeats"])) if eligible else grid[0])
    policy = {"threshold": selected["threshold"], "target_met_on_development": bool(passing),
              "selected_development": selected, "grid": grid, "protocol": protocol}
    policy_file = args.out / "policy.json"
    policy_file.write_text(json.dumps(policy, indent=2))
    policy_hash = hashlib.sha256(policy_file.read_bytes()).hexdigest()
    # The test cannot change this saved policy or threshold.
    names3, test = pickle.load((cache / "refine-v7-tsar-test.pkl").open("rb"))
    if names3 != names:
        raise ValueError("Test cache feature mismatch")
    Xt, _, _, test_keys = R.flat_specific(test)
    saved = pickle.load((ROOT / "runs/ensemble-v7-20260928/decision.pkl").open("rb"))
    probabilities = dict(zip(test_keys, saved.predict_proba(Xt)[:, 1]))
    comparison = {
        "policy_sha256_before_test": policy_hash,
        "baseline": evaluate(test, probabilities, .45, .1),
        "selected": evaluate(test, probabilities, selected["threshold"], .1),
        "scope": "Target-word-supplied lexical substitution. Listed matches are not sentence-level meaning/grammar labels. TSAR has been reused previously.",
    }
    (args.out / "comparison.json").write_text(json.dumps(comparison, indent=2))
    assert hashlib.sha256(policy_file.read_bytes()).hexdigest() == policy_hash
    print(json.dumps({"threshold": selected["threshold"], "dev_target_met": bool(passing), "comparison": comparison}), flush=True)


if __name__ == "__main__":
    main()
