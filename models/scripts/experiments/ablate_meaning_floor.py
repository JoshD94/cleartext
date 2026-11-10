"""Compare the fixed contextual policies with and without the hard sense floor."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cleartext.audit import sha256
from cleartext.data import ROOT


def choose(cases, keys, probability, threshold, floor):
    scores = dict(zip(keys, probability))
    selected = {}
    for case in cases:
        eligible = [r for r in case["candidates"] if r["guard"] and r["x"][0] >= floor
                    and scores[case["id"], r["word"]] >= threshold]
        selected[case["id"]] = max(eligible, key=lambda r: scores[case["id"], r["word"]]) if eligible else None
    edits = [r for r in selected.values() if r is not None]
    correct = sum(bool(r["gold"]) for r in edits)
    return {"edits": len(edits), "correct": correct, "wrong": len(edits) - correct,
            "net": 2 * correct - len(edits), "precision": correct / len(edits) if edits else None,
            "coverage": len(edits) / len(cases)}, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    assert protocol["thresholds"] == [.45, .80] and protocol["seeds"] == [0, 1, 2]
    baseline = ROOT / "outputs/detail-preservation-final-20261002"
    prior = json.loads((baseline / "summary.json").read_text())
    paths = [ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")]
    cases = []
    for path in paths:
        assert sha256(path) == prior["input_sha256"][str(path.relative_to(ROOT))]
        with path.open("rb") as stream:
            cases.extend(pickle.load(stream)[1])
    keys = [(c["id"], r["word"]) for c in cases for r in c["candidates"]]
    assert json.loads((baseline / "feature-keys.json").read_text()) == [list(k) for k in keys]
    inputs = paths + [args.protocol, baseline / "feature-keys.json", baseline / "summary.json"]
    inputs += [baseline / f"similarity-seed{seed}.npz" for seed in protocol["seeds"]]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"protocol": protocol, "input_sha256": {str(p.resolve()): sha256(p) for p in inputs},
                "source_sha256": sha256(Path(__file__))}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for seed in protocol["seeds"]:
        with np.load(baseline / f"similarity-seed{seed}.npz", allow_pickle=False) as f:
            probability = f["probability"]
        assert len(probability) == len(keys) and np.isfinite(probability).all()
        for threshold in protocol["thresholds"]:
            a, old = choose(cases, keys, probability, threshold, .10)
            b, new = choose(cases, keys, probability, threshold, 0)
            paired = [{"id": c["id"],
                       "baseline": {"word": old[c["id"]]["word"], "listed": bool(old[c["id"]]["gold"])} if old[c["id"]] else None,
                       "no_floor": {"word": new[c["id"]]["word"], "listed": bool(new[c["id"]]["gold"])} if new[c["id"]] else None}
                      for c in cases]
            changed = [r for r in paired if r["baseline"] != r["no_floor"]]
            row = {"seed": seed, "threshold": threshold, "baseline": a, "no_floor": b,
                   "net_delta": b["net"] - a["net"], "changed_selections": len(changed),
                   "correct_to_wrong": sum(bool(r["baseline"] and r["baseline"]["listed"] and r["no_floor"] and not r["no_floor"]["listed"]) for r in changed),
                   "wrong_to_correct": sum(bool(r["baseline"] and not r["baseline"]["listed"] and r["no_floor"] and r["no_floor"]["listed"]) for r in changed)}
            (args.output / f"paired-seed{seed}-threshold{threshold:.2f}.json").write_text(json.dumps(changed, indent=2) + "\n")
            results.append(row)
    delta = float(np.mean([r["net_delta"] for r in results if r["threshold"] == .45]))
    strict = all(r["no_floor"]["precision"] is not None and r["no_floor"]["precision"] >= .9
                 and r["no_floor"]["coverage"] >= .2 for r in results if r["threshold"] == .8)
    summary = {"scope": protocol["scope"], "results": results, "mean_default_net_delta": delta,
               "strict_constraints_passed": strict, "retention_bar_passed": delta >= 5 and strict,
               "default_changed": False, "new_model_fitted": False,
               "manifest_sha256": sha256(args.output / "manifest.json")}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
