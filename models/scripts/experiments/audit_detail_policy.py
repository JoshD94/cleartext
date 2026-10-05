"""Fixed direct-hypernym policy versus v7 on cached BenchLS development only.

Three target-word-grouped 10-fold repeats, fixed existing v7 threshold/floor.
No threshold search, no TSAR labels, no artifact updates. This is an exploratory
development comparison, not an independent test of full-sentence meaning.
"""

import argparse
import json
import importlib.metadata
import os
import pickle
import sys
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "4"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from cleartext import refine as R
from cleartext.audit import sha256
from cleartext.data import ROOT
from cleartext.preservation import broader_links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    tables = [pickle.load(path.open("rb")) for path in paths]
    names = tables[0][0]
    assert names == tables[1][0]
    cases = tables[0][1] + tables[1][1]
    config_path = ROOT / "runs/ensemble-v7-20260928/config.json"
    config = json.loads(config_path.read_text())
    threshold, floor = config["decision_threshold"], config["meaning_floor"]
    sense_column = names.index("sense_fit")
    X, y, groups, keys = R.flat_specific(cases)
    blocked = {
        (case["id"], candidate["word"])
        for case in cases
        for candidate in case["candidates"]
        if broader_links(candidate, case.get("target_info"))
    }
    repeats = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        unique = np.unique(groups)
        assignment = dict(zip(unique, rng.permutation(len(unique)) % 10))
        folds = np.array([assignment[group] for group in groups])
        probability = np.zeros(len(y))
        for fold in range(10):
            train, dev = folds != fold, folds == fold
            model = R.AverageDecision().fit(X[train], y[train])
            probability[dev] = model.predict_proba(X[dev])[:, 1]
        scores = dict(zip(keys, probability))

        def chooser(preserve_detail):
            def choose(case):
                eligible = [
                    c
                    for c in case["candidates"]
                    if c["guard"]
                    and c["x"][sense_column] >= floor
                    and scores[case["id"], c["word"]] >= threshold
                    and not (preserve_detail and (case["id"], c["word"]) in blocked)
                ]
                return (
                    max(eligible, key=lambda c: scores[case["id"], c["word"]])
                    if eligible
                    else None
                )

            return choose

        row = {"seed": seed}
        for name, enabled in (("v7", False), ("detail_policy", True)):
            row[name] = R.evaluate(cases, chooser(enabled))
            row[name]["coverage"] = row[name]["edits"] / len(cases)
        row["net_delta"] = row["detail_policy"]["net"] - row["v7"]["net"]
        repeats.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "scope": "All BenchLS is development; historical holdout is now development too. "
        "Existing target-word grouped CV is reused. Candidate labels do not judge full-sentence detail.",
        "threshold": threshold,
        "meaning_floor": floor,
        "repeats": repeats,
        "mean_net_delta": float(np.mean([r["net_delta"] for r in repeats])),
        "blocked_candidate_pairs": len(blocked),
        "adoption_noise_bar": "net +5 or log-loss -0.001",
        "input_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in paths + [config_path]
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                Path(__file__).resolve(),
                ROOT / "src/cleartext/preservation.py",
                ROOT / "src/cleartext/refine.py",
                ROOT / "src/cleartext/generation.py",
            ]
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "nltk")
        },
        "default_changed": False,
        "tsar_gold_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
