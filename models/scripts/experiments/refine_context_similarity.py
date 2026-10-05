"""Compare contextual replacement similarity with the fixed BERT support model."""

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
from cleartext.context_similarity import features_prompted_similarity
from cleartext.data import ROOT
from cleartext.features import nlp, target_token
from cleartext.prompted_relations import features_prompted_relations, support_map
from refine_relations import feature_table, choose_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-run", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.save_run.exists():
        raise FileExistsError("output and run must be new")
    cached = json.loads((args.cache / "summary.json").read_text())
    for path, expected in {**cached["input_sha256"], **cached["source_sha256"]}.items():
        assert sha256(ROOT / path) == expected, path
    assert sha256(args.cache / "records.jsonl") == cached["records_sha256"]
    paths = [ROOT / f"data/cache/refine-v7-benchls-{p}.pkl" for p in ("dev", "holdout")]
    cases = sum([pickle.load(p.open("rb"))[1] for p in paths], [])
    predictions_path = ROOT / "outputs/bert-candidate-coverage-20261002/records.jsonl"
    raw = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    similarity = [
        json.loads(line)
        for line in (args.cache / "records.jsonl").read_text().splitlines()
    ]
    assert (
        [c["id"] for c in cases]
        == [r["id"] for r in raw]
        == [r["id"] for r in similarity]
    )
    for case, record, scores, doc in zip(
        cases, raw, similarity, nlp().pipe([c["text"] for c in cases])
    ):
        assert case["text"] == record["text"] and case["target"] == record["target"]
        token = target_token(doc, case["target"], record["start"])
        assert token is not None and token.idx == record["start"] == scores["start"]
        _, candidates = proposals(token, record["raw_predictions"], case["target_info"])
        case["bert_support"] = support_map(candidates)
        case["replacement_similarity"] = scores["scores"]
    X, y, groups, keys = feature_table(cases, features_prompted_relations)
    new, labels, grouped, new_keys = feature_table(cases, features_prompted_similarity)
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
        for name, matrix in [("prompted", X), ("similarity", new)]:
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
        result["net_delta"] = result["similarity"]["net"] - result["prompted"]["net"]
        result["log_loss_delta"] = (
            result["similarity"]["log_loss"] - result["prompted"]["log_loss"]
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
            "context_similarity.py",
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
    inputs = paths + [
        predictions_path,
        args.cache / "records.jsonl",
        args.cache / "summary.json",
    ]
    summary = {
        "scope": "Reused BenchLS development, fixed target-word folds/threshold/floor. No TSAR or new labels.",
        "repeats": repeats,
        "mean_net_delta": net,
        "mean_log_loss_delta": loss,
        "noise_bar_passed": net >= 5 or loss <= -0.001,
        "noise_bar": "net +5 or log-loss -0.001",
        "feature_shapes": {"prompted": list(X.shape), "similarity": list(new.shape)},
        "threshold": 0.45,
        "meaning_floor": 0.10,
        "default_changed": False,
        "saved_run": None,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "input_sha256": {str(p.resolve().relative_to(ROOT)): sha256(p) for p in inputs},
    }
    if summary["noise_bar_passed"]:
        run = args.save_run.resolve()
        relative = run.relative_to(ROOT)
        base_run = ROOT / "runs/ensemble-prompted-relations-20261002"
        config = json.loads((base_run / "config.json").read_text())
        run.mkdir(parents=True)
        with (run / "decision.pkl").open("xb") as stream:
            pickle.dump(R.AverageDecision().fit(new, y), stream)
        shutil.copy2(base_run / "stacker.pkl", run / "stacker.pkl")
        config.update(
            decision=str(relative / "decision.pkl"),
            decision_features="prompted_similarity",
        )
        (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        summary["saved_run"] = str(relative)
        (run / "development.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
