"""Isolate noun-head proposals with the existing decision weights frozen.

Reuse verified candidate tables, train only the baseline in each grouped fold,
and predict both proposal policies. The saved run shares prior decision weights.
"""

import argparse
import json
import os
import pickle
import shutil
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
from cleartext.relations import features_relations
from refine_relations import feature_table, choose_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables", type=Path, default=ROOT / "outputs/noun-head-dev-review-20261002"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-run", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.save_run.exists():
        raise FileExistsError("output and saved run must be new paths")
    run = args.save_run.resolve()
    relative_run = run.relative_to(ROOT)
    previous = json.loads((args.tables / "summary.json").read_text())
    for relative, expected in {
        **previous["input_sha256"],
        **previous["source_sha256"],
    }.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(
                f"tables were built with different input or source: {relative}"
            )
    base_run = ROOT / "runs/ensemble-windowed-20261002"
    config = json.loads((base_run / "config.json").read_text())
    tables = {
        name: pickle.load((args.tables / f"{name}.pkl").open("rb"))
        for name in ("baseline", "headed")
    }
    if tables["baseline"][0] != tables["headed"][0]:
        raise ValueError("fit-member order differs")
    cases = {name: value[1] for name, value in tables.items()}
    if [(case["id"], case["target"]) for case in cases["baseline"]] != [
        (case["id"], case["target"]) for case in cases["headed"]
    ]:
        raise ValueError("development cases differ")
    matrices = {
        name: feature_table(value, features_relations) for name, value in cases.items()
    }
    unique = np.unique(np.concatenate([value[2] for value in matrices.values()]))
    threshold, floor = config["decision_threshold"], config["meaning_floor"]
    args.output.mkdir(parents=True)
    repeats = []
    for seed in range(3):
        assignment = dict(
            zip(unique, np.random.default_rng(seed).permutation(len(unique)) % 10)
        )
        folds = {
            name: np.asarray([assignment[group] for group in value[2]])
            for name, value in matrices.items()
        }
        probabilities = {
            name: np.zeros(len(value[1])) for name, value in matrices.items()
        }
        baseline, labels, _, _ = matrices["baseline"]
        for fold in range(10):
            model = R.AverageDecision().fit(
                baseline[folds["baseline"] != fold], labels[folds["baseline"] != fold]
            )
            for name, value in matrices.items():
                dev = folds[name] == fold
                probabilities[name][dev] = model.predict_proba(value[0][dev])[:, 1]
        row = {"seed": seed}
        for name, value in matrices.items():
            row[name], selections = choose_scores(
                cases[name],
                dict(zip(value[3], probabilities[name])),
                0,
                threshold,
                floor,
            )
            np.savez_compressed(
                args.output / f"{name}-seed{seed}.npz",
                probability=probabilities[name],
                folds=folds[name],
            )
            (args.output / f"{name}-selections-seed{seed}.json").write_text(
                json.dumps(
                    {
                        key: None
                        if candidate is None
                        else {
                            "word": candidate["word"],
                            "gold": bool(candidate["gold"]),
                        }
                        for key, candidate in selections.items()
                    },
                    indent=2,
                )
                + "\n"
            )
        row["net_delta"] = row["headed"]["net"] - row["baseline"]["net"]
        repeats.append(row)
        print(json.dumps(row), flush=True)
    net = float(np.mean([row["net_delta"] for row in repeats]))
    paths = [args.tables / f"{name}.pkl" for name in tables] + [
        args.tables / "summary.json",
        base_run / "config.json",
    ]
    sources = [Path(__file__).resolve()] + [
        ROOT / relative for relative in previous["source_sha256"]
    ]
    result = {
        "scope": "All BenchLS is reused development. Candidate grammar and full-sentence meaning are ungraded.",
        "repeats": repeats,
        "mean_net_delta": net,
        "noise_bar_passed": net >= 5,
        "noise_bar": "paired net +5; no calibration claim for different candidate populations",
        "threshold": threshold,
        "meaning_floor": floor,
        "tsar_gold_used": False,
        "default_changed": False,
        "decision_training": "Each held-out fold uses a model trained on baseline candidates only. "
        "The saved run shares the prior full-data decision weights.",
        "input_sha256": {
            str(path.resolve().relative_to(ROOT)): sha256(path) for path in paths
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "saved_run": str(relative_run),
    }
    for path in sources:
        destination = args.output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    run.mkdir(parents=True)
    shutil.copy2(base_run / "stacker.pkl", run / "stacker.pkl")
    config["noun_phrase_heads"] = True
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (run / "development.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
