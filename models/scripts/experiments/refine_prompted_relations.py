"""Test original-visible BERT support while retaining WordNet and meaning gates."""

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
from sklearn.metrics import log_loss
from cleartext import refine as R
from cleartext.audit import sha256
from cleartext.bert_candidates import proposals
from cleartext.data import ROOT
from cleartext.features import nlp, target_token
from cleartext.prompted_relations import features_prompted_relations, support_map
from cleartext.relations import features_relations
from refine_relations import feature_table, choose_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-run", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.save_run.exists():
        raise FileExistsError("output and run must be new")
    run = args.save_run.resolve()
    relative_run = run.relative_to(ROOT)
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    tables = [pickle.load(path.open("rb")) for path in paths]
    assert tables[0][0] == tables[1][0]
    cases = tables[0][1] + tables[1][1]
    records = [
        json.loads(line)
        for line in (args.predictions / "records.jsonl").read_text().splitlines()
    ]
    summary = json.loads((args.predictions / "summary.json").read_text())
    for relative, expected in {
        **summary["source_sha256"],
        **summary["input_sha256"],
    }.items():
        assert sha256(ROOT / relative) == expected, relative
    assert [case["id"] for case in cases] == [record["id"] for record in records]
    for case, record, doc in zip(
        cases, records, nlp().pipe([case["text"] for case in cases])
    ):
        assert case["text"] == record["text"] and case["target"] == record["target"]
        token = target_token(doc, case["target"], record["start"])
        assert token is not None and token.idx == record["start"]
        _, candidates = proposals(
            token, record["raw_predictions"], case.get("target_info")
        )
        case["bert_support"] = support_map(candidates)
    X, y, groups, keys = feature_table(cases, features_relations)
    new, new_y, new_groups, new_keys = feature_table(cases, features_prompted_relations)
    assert (
        np.array_equal(y, new_y)
        and np.array_equal(groups, new_groups)
        and keys == new_keys
    )
    np.testing.assert_array_equal(new[:, : X.shape[1]], X)
    base_run = ROOT / "runs/ensemble-windowed-20261002"
    config = json.loads((base_run / "config.json").read_text())
    args.output.mkdir(parents=True)
    repeats = []
    for seed in range(3):
        unique = np.unique(groups)
        assignment = dict(
            zip(unique, np.random.default_rng(seed).permutation(len(unique)) % 10)
        )
        folds = np.asarray([assignment[group] for group in groups])
        row = {"seed": seed}
        for name, matrix in [("baseline", X), ("prompted", new)]:
            probability = np.zeros(len(y))
            for fold in range(10):
                train, dev = folds != fold, folds == fold
                probability[dev] = (
                    R.AverageDecision()
                    .fit(matrix[train], y[train])
                    .predict_proba(matrix[dev])[:, 1]
                )
            row[name], selections = choose_scores(
                cases,
                dict(zip(keys, probability)),
                0,
                config["decision_threshold"],
                config["meaning_floor"],
            )
            row[name]["log_loss"] = float(
                log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6))
            )
            np.savez_compressed(
                args.output / f"{name}-seed{seed}.npz",
                probability=probability,
                folds=folds,
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
        row["net_delta"] = row["prompted"]["net"] - row["baseline"]["net"]
        row["log_loss_delta"] = (
            row["prompted"]["log_loss"] - row["baseline"]["log_loss"]
        )
        repeats.append(row)
        print(json.dumps(row), flush=True)
    net = float(np.mean([row["net_delta"] for row in repeats]))
    loss = float(np.mean([row["log_loss_delta"] for row in repeats]))
    passed = net >= 5 or loss <= -0.001
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("refine_relations.py"),
    ]
    sources += [
        ROOT / "src/cleartext" / name
        for name in (
            "bert_candidates.py",
            "prompted_relations.py",
            "relations.py",
            "refine.py",
            "ensemble_pipeline.py",
        )
    ]
    for path in sources:
        destination = args.output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    inputs = paths + [
        args.predictions / "records.jsonl",
        args.predictions / "summary.json",
        base_run / "config.json",
    ]
    result = {
        "scope": "Reused BenchLS development, fixed target-word folds/threshold/floor. No TSAR or new labels.",
        "repeats": repeats,
        "mean_net_delta": net,
        "mean_log_loss_delta": loss,
        "noise_bar_passed": passed,
        "noise_bar": "net +5 or log-loss -0.001",
        "feature_shapes": {"baseline": list(X.shape), "prompted": list(new.shape)},
        "threshold": config["decision_threshold"],
        "meaning_floor": config["meaning_floor"],
        "default_changed": False,
        "saved_run": None,
        "input_sha256": {
            str(path.resolve().relative_to(ROOT)): sha256(path) for path in inputs
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    if passed:
        run.mkdir(parents=True)
        with (run / "decision.pkl").open("xb") as stream:
            pickle.dump(R.AverageDecision().fit(new, y), stream)
        shutil.copy2(base_run / "stacker.pkl", run / "stacker.pkl")
        config.update(
            decision=str(relative_run / "decision.pkl"),
            decision_features="prompted_relations",
        )
        (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        result["saved_run"] = str(relative_run)
        (run / "development.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
