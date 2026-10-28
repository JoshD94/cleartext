"""Apply fresh paired guard masks to the unchanged contextual development scores."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cleartext.audit import sha256
from cleartext.data import ROOT
from ablate_meaning_floor import choose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads((args.comparison / "summary.json").read_text())
    assert comparison["records_sha256"] == sha256(args.comparison / "records.jsonl")
    masks = [json.loads(line) for line in (args.comparison / "records.jsonl").read_text().splitlines()]
    baseline = ROOT / "outputs/detail-preservation-final-20261002"
    old = json.loads((baseline / "summary.json").read_text())
    paths = [ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")]
    cases = []
    for path in paths:
        assert sha256(path) == old["input_sha256"][str(path.relative_to(ROOT))]
        with path.open("rb") as stream:
            cases.extend(pickle.load(stream)[1])
    assert [c["id"] for c in cases] == [c["id"] for c in masks]
    keys = [(c["id"], r["word"]) for c in cases for r in c["candidates"]]
    assert json.loads((baseline / "feature-keys.json").read_text()) == [list(k) for k in keys]
    for case, mask in zip(cases, masks):
        assert [r["word"] for r in case["candidates"]] == [r["word"] for r in mask["rows"]]
    variants = {name: [{**case, "candidates": [{**r, "guard": mask[name]}
                  for r, mask in zip(case["candidates"], scan["rows"])]}
                  for case, scan in zip(cases, masks)] for name in ("legacy", "aligned")}
    inputs = paths + [args.comparison / n for n in ("summary.json", "manifest.json", "records.jsonl")]
    inputs += [baseline / "feature-keys.json", baseline / "summary.json"]
    inputs += [baseline / f"similarity-seed{s}.npz" for s in range(3)]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"input_sha256": {str(p.resolve()): sha256(p) for p in inputs},
                "source_sha256": {str(p.resolve()): sha256(p) for p in
                                  [Path(__file__), Path(__file__).with_name("ablate_meaning_floor.py")]}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for seed in range(3):
        with np.load(baseline / f"similarity-seed{seed}.npz", allow_pickle=False) as f:
            probabilities = f["probability"]
        for threshold in (.45, .80):
            metrics, chosen = {}, {}
            for name, rows in variants.items():
                metrics[name], chosen[name] = choose(rows, keys, probabilities, threshold, .10)
            changed = []
            for case in cases:
                a, b = chosen["legacy"][case["id"]], chosen["aligned"][case["id"]]
                if (a["word"] if a else None) != (b["word"] if b else None):
                    changed.append({"id": case["id"],
                                    "legacy": {"word": a["word"], "listed": bool(a["gold"])} if a else None,
                                    "aligned": {"word": b["word"], "listed": bool(b["gold"])} if b else None})
            (args.output / f"changes-seed{seed}-threshold{threshold:.2f}.json").write_text(json.dumps(changed, indent=2) + "\n")
            results.append({"seed": seed, "threshold": threshold, **metrics,
                            "net_delta": metrics["aligned"]["net"] - metrics["legacy"]["net"],
                            "changed_selections": len(changed)})
    ceiling = {name: {"listed_guarded": sum(any(r["gold"] and r["guard"] for r in c["candidates"]) for c in rows),
                      "listed_eligible": sum(any(r["gold"] and r["guard"] and r["x"][0] >= .10 for r in c["candidates"]) for c in rows)}
               for name, rows in variants.items()}
    summary = {"scope": "Paired guard-only comparison on all929 reused BenchLS development cases with target supplied. Frozen contextual OOF scores; no new model or semantic labels.",
               "results": results, "candidate_ceiling": ceiling, "manifest_sha256": sha256(args.output / "manifest.json")}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
