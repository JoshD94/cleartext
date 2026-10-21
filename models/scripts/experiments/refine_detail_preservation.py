"""Compare contextual detail features and frozen term guards on reused development."""

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
from cleartext.context_similarity import features_prompted_similarity
from cleartext.data import ROOT
from cleartext.detail_preservation import (
    DETAIL_FEATURE_NAMES,
    features_detail_preservation,
)
from cleartext.features import nlp, target_token
from cleartext.prompted_relations import support_map
from refine_relations import feature_table, choose_scores


def term_guard_metrics(cases, scores, blocked):
    guarded = [
        {**case, "candidates": []} if case["id"] in blocked else case for case in cases
    ]
    return choose_scores(guarded, scores, 0, 0.45, 0.10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-run", type=Path, required=True)
    parser.add_argument("--term-run", type=Path, required=True)
    parser.add_argument(
        "--frozen-features",
        type=Path,
        help="optional saved paired feature table for exact reproduction",
    )
    args = parser.parse_args()
    if any(path.exists() for path in (args.output, args.save_run, args.term_run)):
        raise FileExistsError("output and run directories must be new")
    summary = json.loads((args.cache / "summary.json").read_text())
    for path, expected in {
        **summary["input_sha256"],
        **summary["source_sha256"],
    }.items():
        assert sha256(ROOT / path) == expected, path
    assert sha256(args.cache / "records.jsonl") == summary["records_sha256"]
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    cases = sum([pickle.load(path.open("rb"))[1] for path in paths], [])
    prompted_path = ROOT / "outputs/bert-candidate-coverage-20261002/records.jsonl"
    similarity_path = (
        ROOT / "outputs/context-similarity-cache-review-20261002/records.jsonl"
    )
    similarity_summary = json.loads(
        (similarity_path.parent / "summary.json").read_text()
    )
    assert sha256(similarity_path) == similarity_summary["records_sha256"]
    raw = [json.loads(line) for line in prompted_path.read_text().splitlines()]
    similarity = [json.loads(line) for line in similarity_path.read_text().splitlines()]
    senses = [
        json.loads(line)
        for line in (args.cache / "records.jsonl").read_text().splitlines()
    ]
    assert (
        [c["id"] for c in cases]
        == [r["id"] for r in raw]
        == [r["id"] for r in similarity]
        == [r["id"] for r in senses]
    )
    blocked = {row["id"] for row in senses if row["protected_target"]}
    for case, prompted, cosine, sense, doc in zip(
        cases, raw, similarity, senses, nlp().pipe([c["text"] for c in cases])
    ):
        assert case["text"] == prompted["text"] and case["target"] == prompted["target"]
        token = target_token(doc, case["target"], prompted["start"])
        assert (
            token is not None
            and token.idx == prompted["start"] == cosine["start"] == sense["start"]
        )
        _, candidates = proposals(
            token, prompted["raw_predictions"], case["target_info"]
        )
        case.update(
            bert_support=support_map(candidates),
            replacement_similarity=cosine["scores"],
            sense_distribution=sense["sense_distribution"],
        )
    if args.frozen_features:
        frozen = np.load(args.frozen_features, allow_pickle=False)
        X, extra, y, groups = (
            frozen[name] for name in ("similarity", "detail", "labels", "groups")
        )
        keys = [
            (case["id"], candidate["word"])
            for case in cases
            for candidate in case["candidates"]
        ]
        assert json.loads(
            (args.frozen_features.parent / "feature-keys.json").read_text()
        ) == [list(key) for key in keys]
        np.testing.assert_array_equal(
            y, [candidate["gold"] for case in cases for candidate in case["candidates"]]
        )
        np.testing.assert_array_equal(
            groups,
            [case["target"].lower() for case in cases for _ in case["candidates"]],
        )
    else:
        X, y, groups, keys = feature_table(cases, features_prompted_similarity)
        extra, labels, grouped, new_keys = feature_table(
            cases, features_detail_preservation
        )
        assert (
            np.array_equal(y, labels)
            and np.array_equal(groups, grouped)
            and keys == new_keys
        )
    np.testing.assert_array_equal(X, extra[:, :52])
    assert extra.shape[1] == 52 + len(DETAIL_FEATURE_NAMES)
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "feature-tables.npz",
        similarity=X,
        detail=extra,
        labels=y,
        groups=groups,
    )
    (args.output / "feature-keys.json").write_text(json.dumps(keys) + "\n")
    repeats = []
    for seed in range(3):
        unique = np.unique(groups)
        assignment = dict(
            zip(unique, np.random.default_rng(seed).permutation(len(unique)) % 10)
        )
        folds = np.asarray([assignment[group] for group in groups])
        result = {"seed": seed}
        for name, matrix in [("similarity", X), ("detail", extra)]:
            probability = np.zeros(len(y))
            for fold in range(10):
                train, dev = folds != fold, folds == fold
                probability[dev] = (
                    R.AverageDecision()
                    .fit(matrix[train], y[train])
                    .predict_proba(matrix[dev])[:, 1]
                )
            scores = dict(zip(keys, probability))
            result[name], selections = choose_scores(cases, scores, 0, 0.45, 0.10)
            result[name]["log_loss"] = float(
                log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6))
            )
            result[name + "_terms"], guarded = term_guard_metrics(
                cases, scores, blocked
            )
            if name == "similarity":
                previous = np.load(
                    ROOT
                    / f"outputs/context-similarity-dev-20261002/similarity-seed{seed}.npz"
                )
                np.testing.assert_array_equal(previous["folds"], folds)
                result["historical_similarity"], _ = choose_scores(
                    cases, dict(zip(keys, previous["probability"])), 0, 0.45, 0.10
                )
                result["historical_probability_max_difference"] = float(
                    np.max(np.abs(previous["probability"] - probability))
                )
            np.savez_compressed(
                args.output / f"{name}-seed{seed}.npz",
                probability=probability,
                folds=folds,
            )
            (args.output / f"{name}-selections-seed{seed}.json").write_text(
                json.dumps(
                    {
                        case["id"]: {
                            variant: None
                            if candidate is None
                            else {
                                "word": candidate["word"],
                                "gold": bool(candidate["gold"]),
                            }
                            for variant, candidate in [
                                ("plain", selections[case["id"]]),
                                ("terms", guarded[case["id"]]),
                            ]
                        }
                        for case in cases
                    },
                    indent=2,
                )
                + "\n"
            )
        result["net_delta"] = result["detail"]["net"] - result["similarity"]["net"]
        result["log_loss_delta"] = (
            result["detail"]["log_loss"] - result["similarity"]["log_loss"]
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
            "detail_preservation.py",
            "technical_terms.py",
            "ensemble_pipeline.py",
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
        prompted_path,
        similarity_path,
        args.cache / "records.jsonl",
        args.cache / "summary.json",
    ]
    if args.frozen_features:
        inputs.append(args.frozen_features)
    result = {
        "scope": "Reused BenchLS development, three target-word-grouped 10-fold repeats. No TSAR or new labels.",
        "repeats": repeats,
        "mean_net_delta": net,
        "mean_log_loss_delta": loss,
        "noise_bar_passed": net >= 5 or loss <= -0.001,
        "noise_bar": "net +5 or log-loss -0.001",
        "feature_shapes": {"similarity": list(X.shape), "detail": list(extra.shape)},
        "new_feature_names": list(DETAIL_FEATURE_NAMES),
        "protected_targets": sorted(blocked),
        "baseline_drift": "Legacy breadth traversal can overshoot 20,000 by hash-dependent sibling counts. "
        "Prior probabilities are recorded separately; both compared models share the same frozen 52 columns.",
        "feature_tables_sha256": sha256(args.output / "feature-tables.npz"),
        "threshold": 0.45,
        "meaning_floor": 0.10,
        "default_changed": False,
        "saved_run": None,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "input_sha256": {str(p.resolve().relative_to(ROOT)): sha256(p) for p in inputs},
    }
    base_run = ROOT / "runs/ensemble-context-similarity-20261002"
    config = json.loads((base_run / "config.json").read_text())
    # A frozen term-only variant is useful even if new learned features miss the acceptance bar.
    args.term_run.mkdir(parents=True)
    shutil.copy2(base_run / "stacker.pkl", args.term_run / "stacker.pkl")
    (args.term_run / "config.json").write_text(
        json.dumps({**config, "technical_terms": True}, indent=2) + "\n"
    )
    result["term_run"] = str(args.term_run.resolve().relative_to(ROOT))
    if result["noise_bar_passed"]:
        relative = args.save_run.resolve().relative_to(ROOT)
        args.save_run.mkdir(parents=True)
        with (args.save_run / "decision.pkl").open("xb") as stream:
            pickle.dump(R.AverageDecision().fit(extra, y), stream)
        shutil.copy2(base_run / "stacker.pkl", args.save_run / "stacker.pkl")
        config.update(
            decision=str(relative / "decision.pkl"),
            decision_features="detail_preservation",
            technical_terms=True,
        )
        (args.save_run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        result["saved_run"] = str(relative)
        (args.save_run / "development.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    (args.term_run / "development.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
