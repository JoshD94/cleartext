"""Compare learned relation features with frozen v7 on BenchLS development.

Fixed deployed threshold/floor, three target-word grouped 10-fold repeats. No
TSAR reads or threshold search. An optional new run is fitted only when the
development gain clears the existing noise bar. The demo/default stay on v7.
"""

import argparse
import importlib.metadata
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
from sklearn.metrics import log_loss
from cleartext import refine as R
from cleartext.audit import sha256
from cleartext.data import ROOT
from cleartext.relations import RELATION_NAMES, features_relations


def feature_table(cases, featurizer):
    rows, labels, groups, keys = [], [], [], []
    for case in cases:
        feature = featurizer(case)
        for candidate in case["candidates"]:
            rows.append(feature(case, candidate))
            labels.append(candidate["gold"])
            groups.append(case["target"].lower())
            keys.append((case["id"], candidate["word"]))
    return np.asarray(rows), np.asarray(labels, bool), np.asarray(groups), keys


def choose_scores(cases, scores, sense_column, threshold, floor):
    selections = {}
    for case in cases:
        eligible = [
            candidate
            for candidate in case["candidates"]
            if candidate["guard"]
            and candidate["x"][sense_column] >= floor
            and scores[case["id"], candidate["word"]] >= threshold
        ]
        selections[case["id"]] = (
            max(eligible, key=lambda c: scores[case["id"], c["word"]])
            if eligible
            else None
        )
    result = R.evaluate(cases, lambda case: selections[case["id"]])
    result["coverage"] = result["edits"] / len(cases)
    return result, selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new output directory"
    )
    parser.add_argument(
        "--fit-run", type=Path, help="optional new run, only fitted if noise bar passes"
    )
    args = parser.parse_args()
    if args.output.exists() or (args.fit_run and args.fit_run.exists()):
        raise FileExistsError("output and run paths must be new")
    args.output.mkdir(parents=True)
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    tables = [pickle.load(path.open("rb")) for path in paths]
    names = tables[0][0]
    if names != tables[1][0] or names[0] != "sense_fit":
        raise ValueError("feature member order differs from v7")
    cases = tables[0][1] + tables[1][1]
    original_run = ROOT / "runs/ensemble-v7-20260928"
    config_path = original_run / "config.json"
    config = json.loads(config_path.read_text())
    threshold, floor = config["decision_threshold"], config["meaning_floor"]
    X, y, groups, keys = feature_table(cases, R.features_specific)
    extra, y_extra, groups_extra, keys_extra = feature_table(cases, features_relations)
    if not (
        np.array_equal(y, y_extra)
        and np.array_equal(groups, groups_extra)
        and keys == keys_extra
    ):
        raise ValueError("feature variants have mismatched candidate rows")
    np.testing.assert_array_equal(extra[:, : X.shape[1]], X)
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "src/cleartext/relations.py",
        ROOT / "src/cleartext/refine.py",
        ROOT / "src/cleartext/generation.py",
        ROOT / "src/cleartext/ensemble_pipeline.py",
    ]
    for path in source_paths:
        destination = args.output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    repeats = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        unique = np.unique(groups)
        assignment = dict(zip(unique, rng.permutation(len(unique)) % 10))
        folds = np.array([assignment[group] for group in groups])
        row = {"seed": seed}
        selected = {}
        for name, matrix in (("v7", X), ("relations", extra)):
            probability = np.zeros(len(y))
            for fold in range(10):
                train, dev = folds != fold, folds == fold
                probability[dev] = (
                    R.AverageDecision()
                    .fit(matrix[train], y[train])
                    .predict_proba(matrix[dev])[:, 1]
                )
            metrics, selected[name] = choose_scores(
                cases, dict(zip(keys, probability)), 0, threshold, floor
            )
            metrics["log_loss"] = float(
                log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6))
            )
            row[name] = metrics
            np.savez_compressed(
                args.output / f"{name}-seed{seed}.npz",
                probability=probability,
                folds=folds,
            )
            print(json.dumps({"seed": seed, "variant": name, **metrics}), flush=True)
        row["net_delta"] = row["relations"]["net"] - row["v7"]["net"]
        row["log_loss_delta"] = row["relations"]["log_loss"] - row["v7"]["log_loss"]
        repeats.append(row)
        with (args.output / f"selections-seed{seed}.jsonl").open("x") as stream:
            for case in cases:
                choices = {
                    name: None
                    if value is None
                    else {
                        "word": value["word"],
                        "gold": bool(value["gold"]),
                        "source": value["source"],
                    }
                    for name in selected
                    for value in [selected[name][case["id"]]]
                }
                stream.write(
                    json.dumps(
                        {
                            "id": case["id"],
                            "target": case["target"],
                            "text": case["text"],
                            "choices": choices,
                        }
                    )
                    + "\n"
                )
    mean_net = float(np.mean([row["net_delta"] for row in repeats]))
    mean_loss = float(np.mean([row["log_loss_delta"] for row in repeats]))
    passed = mean_net >= 5 or mean_loss <= -0.001
    result = {
        "scope": "All BenchLS is reused development, including historical holdout. "
        "Candidate labels do not measure retained full-sentence detail.",
        "repeats": repeats,
        "mean_net_delta": mean_net,
        "mean_log_loss_delta": mean_loss,
        "noise_bar_passed": passed,
        "noise_bar": "net +5 or log-loss -0.001",
        "threshold": threshold,
        "meaning_floor": floor,
        "new_feature_names": list(RELATION_NAMES),
        "shape": {"v7": list(X.shape), "relations": list(extra.shape)},
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in paths + [config_path]
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "nltk")
        },
        "tsar_gold_used": False,
        "default_changed": False,
        "fitted_run": None,
    }
    if passed and args.fit_run:
        run = args.fit_run.resolve()
        relative_run = run.relative_to(ROOT)
        run.mkdir(parents=True)
        with (run / "decision.pkl").open("xb") as stream:
            pickle.dump(R.AverageDecision().fit(extra, y), stream)
        shutil.copy2(original_run / "stacker.pkl", run / "stacker.pkl")
        config.update(
            decision=str(relative_run / "decision.pkl"), decision_features="relations"
        )
        (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        result["fitted_run"] = str(relative_run)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["fitted_run"]:
        (args.fit_run / "development.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
