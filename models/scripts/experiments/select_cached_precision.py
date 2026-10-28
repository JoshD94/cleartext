"""Select a precision policy from frozen contextual-ranker development scores."""

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cache = ROOT / "outputs/context-similarity-dev-20261002"
    summary = json.loads((cache / "summary.json").read_text())
    for name, expected in summary["input_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Cached development input changed: {name}")
    cases = []
    for part in ("dev", "holdout"):
        names, rows = pickle.load((ROOT / f"data/cache/refine-v7-benchls-{part}.pkl").open("rb"))
        if names[0] != "sense_fit":
            raise ValueError("Unexpected cached feature order")
        cases.extend(rows)
    keys = [(row["id"], c["word"]) for row in cases for c in row["candidates"]]
    predictions = [dict(zip(keys, np.load(cache / f"similarity-seed{s}.npz")["probability"])) for s in range(3)]
    def evaluate(scores, threshold):
        selections = {}
        for row in cases:
            ok = [c for c in row["candidates"] if c["guard"] and c["x"][0] >= .1
                  and scores[row["id"], c["word"]] >= threshold]
            c = max(ok, key=lambda c: scores[row["id"], c["word"]]) if ok else None
            selections[row["id"]] = None if c is None else {"word": c["word"], "gold": bool(c["gold"])}
        edits = [c for c in selections.values() if c is not None]
        correct = sum(c["gold"] for c in edits)
        return {"n": len(cases), "edits": len(edits), "correct": correct,
                "precision": correct / len(edits) if edits else None,
                "coverage": len(edits) / len(cases)}, selections
    # Reproduce all original selections before trusting array/row alignment.
    for seed, scores in enumerate(predictions):
        result, selected = evaluate(scores, .45)
        assert selected == json.loads((cache / f"similarity-selections-seed{seed}.json").read_text())
        assert result["correct"] == summary["repeats"][seed]["similarity"]["correct"]
    grid = [{"threshold": float(t), "repeats": [evaluate(p, t)[0] for p in predictions]}
            for t in np.round(np.arange(.45, .976, .025), 3)]
    passed = [r for r in grid if all(m["coverage"] >= .2 and m["precision"] >= .9 for m in r["repeats"])]
    if not passed:
        raise ValueError("No contextual-ranker policy meets the predeclared target")
    policy = {
        "threshold": passed[0]["threshold"], "selected_development": passed[0],
        "target_precision": .9, "minimum_coverage": .2,
        "selection": "Lowest threshold meeting both constraints in all3saved target-word-grouped10foldrepeats; no TSAR or pipeline predictions read",
        "scope": "Reused BenchLS development only, no full-sentence accuracy claim",
        "grid": grid, "historical_selection_reproduction": "All929original selections reproduced in all3repeats",
        "input_sha256": {str(p): sha(p) for p in [cache / "summary.json", *[cache / f"similarity-seed{s}.npz" for s in range(3)]]},
    }
    (args.out / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    source = ROOT / "runs/ensemble-context-similarity-20261002"
    run = args.out / "run"
    run.mkdir()
    config = json.loads((source / "config.json").read_text())
    config["decision_threshold"] = policy["threshold"]
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(source / "stacker.pkl", run / "stacker.pkl")
    print(json.dumps(policy["selected_development"]))


if __name__ == "__main__":
    main()
