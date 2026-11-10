"""Fixed development comparison of BERT support and relation interactions."""

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "4"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from sklearn.metrics import log_loss
from cleartext import refine as R
from cleartext.audit import sha256
from cleartext.bert_candidates import proposals
from cleartext.data import ROOT
from cleartext.features import nlp, target_token
from cleartext.prompted_relations import features_prompted_relations, support_map
from cleartext.prompted_specificity import (
    features_prompted_specificity,
    INTERACTION_NAMES,
)
from refine_relations import feature_table, choose_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    predictions = ROOT / "outputs/bert-candidate-coverage-20261002"
    inputs = [
        ROOT / f"data/cache/refine-v7-benchls-{p}.pkl" for p in ("dev", "holdout")
    ]
    cases = sum([pickle.load(path.open("rb"))[1] for path in inputs], [])
    records_path = predictions / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    prior = json.loads(
        (ROOT / "outputs/prompted-relations-dev-20261002/summary.json").read_text()
    )
    for path, expected in prior["input_sha256"].items():
        assert sha256(ROOT / path) == expected, path
    assert [c["id"] for c in cases] == [r["id"] for r in records]
    for case, record, doc in zip(
        cases, records, nlp().pipe([c["text"] for c in cases])
    ):
        assert case["text"] == record["text"] and case["target"] == record["target"]
        token = target_token(doc, case["target"], record["start"])
        assert token is not None and token.idx == record["start"]
        _, candidates = proposals(token, record["raw_predictions"], case["target_info"])
        case["bert_support"] = support_map(candidates)
    X, y, groups, keys = feature_table(cases, features_prompted_relations)
    new, labels, grouped, new_keys = feature_table(cases, features_prompted_specificity)
    assert (
        np.array_equal(y, labels)
        and np.array_equal(groups, grouped)
        and keys == new_keys
    )
    np.testing.assert_array_equal(X, new[:, :51])
    args.output.mkdir(parents=True)
    repeats = []
    for seed in range(3):
        unique = np.unique(groups)
        assignment = dict(
            zip(unique, np.random.default_rng(seed).permutation(len(unique)) % 10)
        )
        folds = np.asarray([assignment[group] for group in groups])
        result = {"seed": seed}
        for name, matrix in [("prompted", X), ("interactions", new)]:
            probability = np.zeros(len(y))
            for fold in range(10):
                train, dev = folds != fold, folds == fold
                probability[dev] = (
                    R.AverageDecision()
                    .fit(matrix[train], y[train])
                    .predict_proba(matrix[dev])[:, 1]
                )
            result[name], selections = choose_scores(
                cases, dict(zip(keys, probability)), 0, 0.45, 0.10
            )
            result[name]["log_loss"] = float(
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
                        if c is None
                        else {"word": c["word"], "gold": bool(c["gold"])}
                        for key, c in selections.items()
                    },
                    indent=2,
                )
                + "\n"
            )
        result["net_delta"] = result["interactions"]["net"] - result["prompted"]["net"]
        result["log_loss_delta"] = (
            result["interactions"]["log_loss"] - result["prompted"]["log_loss"]
        )
        repeats.append(result)
        print(json.dumps(result), flush=True)
    net = float(np.mean([r["net_delta"] for r in repeats]))
    loss = float(np.mean([r["log_loss_delta"] for r in repeats]))
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("refine_relations.py"),
    ]
    sources += [
        ROOT / "src/cleartext" / name
        for name in (
            "prompted_specificity.py",
            "prompted_relations.py",
            "relations.py",
            "refine.py",
            "bert_candidates.py",
        )
    ]
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    summary = {
        "scope": "Reused BenchLS development, three target-word grouped repeats, fixed threshold/floor. No TSAR.",
        "repeats": repeats,
        "mean_net_delta": net,
        "mean_log_loss_delta": loss,
        "noise_bar_passed": net >= 5 or loss <= -0.001,
        "noise_bar": "net +5 or log-loss -0.001",
        "new_features": list(INTERACTION_NAMES),
        "feature_shapes": {"prompted": list(X.shape), "interactions": list(new.shape)},
        "threshold": 0.45,
        "meaning_floor": 0.10,
        "default_changed": False,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "input_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in inputs + [records_path]
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Store a review artifact only after the predeclared noise bar passes.
    # No loader registration, config change, or deployment happens here.
    if summary["noise_bar_passed"]:
        with (args.output / "decision-review.pkl").open("xb") as stream:
            pickle.dump(R.AverageDecision().fit(new, y), stream)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
