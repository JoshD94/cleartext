"""Fixed candidate-sense contrast experiment on reused BenchLS development."""

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from sklearn.metrics import log_loss
from cleartext import ensemble as E, refine as R
from cleartext.audit import sha256
from cleartext.context_window import windowed_word_vectors, windowed_word_vector
from cleartext.data import ROOT
from cleartext.ensemble_pipeline import build_senses
from cleartext.features import nlp, target_token
from cleartext.lexical import synsets
from cleartext.sense_contrast import (
    CONTRAST_FEATURE_NAMES,
    SenseContrastScorer,
    rewritten_sense_contrast,
)
from cleartext.wsd import word_pos
from refine_relations import choose_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-run", type=Path, required=True)
    parser.add_argument(
        "--rewritten",
        action="store_true",
        help="Read the five highest-fit candidates in their rewritten slots",
    )
    args = parser.parse_args()
    if args.output.exists() or args.save_run.exists():
        raise FileExistsError("use new output/run paths")
    args.output.mkdir(parents=True)
    base_run = ROOT / "runs/ensemble-building-blocks-20261002"
    config = json.loads((base_run / "config.json").read_text())
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    features_path = (
        ROOT / "outputs/detail-preservation-final-20261002/feature-tables.npz"
    )
    sense_path = ROOT / "outputs/detail-senses-final-20261002/records.jsonl"
    previous_path = ROOT / "outputs/detail-preservation-final-20261002/summary.json"
    previous = json.loads(previous_path.read_text())
    assert sha256(features_path) == previous["feature_tables_sha256"]
    sense_summary = json.loads((sense_path.parent / "summary.json").read_text())
    assert sha256(sense_path) == sense_summary["records_sha256"]
    sources = list((ROOT / "src/cleartext").glob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "scripts/experiments/refine_relations.py",
        ROOT / "tests/test_sense_contrast.py",
    ]
    inputs = paths + [
        features_path,
        sense_path,
        previous_path,
        sense_path.parent / "summary.json",
        features_path.parent / "feature-keys.json",
        base_run / "config.json",
        ROOT / config["sense_model"],
        ROOT / config["bert_sense"]["vectors"],
    ]
    inputs += [features_path.parent / f"similarity-seed{seed}.npz" for seed in range(3)]
    artifacts = json.loads(
        (ROOT / "outputs/six-blocks-final-20261002/manifest.json").read_text()
    )["artifact_sha256"]
    for name, digest in artifacts.items():
        assert sha256(ROOT / name) == digest, name
    manifest = {
        "scope": "All BenchLS is reused development, including historical holdout. No new labels or TSAR gold.",
        "feature_names": list(CONTRAST_FEATURE_NAMES),
        "threshold": 0.45,
        "meaning_floor": 0.10,
        "variant": "rewritten_sense_contrast" if args.rewritten else "sense_contrast",
        "candidate_limit": 5 if args.rewritten else None,
        "noise_bar": "mean net +5 or candidate log-loss -0.001",
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "artifact_sha256": artifacts,
        "default_changed": False,
        "tsar_gold_used": False,
        "new_labels": False,
    }
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cases = sum([pickle.load(path.open("rb"))[1] for path in paths], [])
    senses = [json.loads(line) for line in sense_path.read_text().splitlines()]
    assert [case["id"] for case in cases] == [row["id"] for row in senses]
    with (ROOT / config["sense_model"]).open("rb") as stream:
        sense_model, _ = build_senses(pickle.load(stream), config)
    scorer = SenseContrastScorer(sense_model.bert)
    names = set(name for row in senses for name in row["sense_distribution"])
    for case in cases:
        names.update(
            sense.name()
            for candidate in case["candidates"]
            for sense in synsets(
                candidate["lemma"].lower().replace(" ", "_"),
                word_pos(case["target_info"][1]),
            )
        )
    started = time.monotonic()
    scorer.prepare(names)
    print(
        json.dumps(
            {
                "prepared_sense_vectors": len(names),
                "dictionary_vectors": len(scorer.bert.gloss),
            }
        ),
        flush=True,
    )
    extra, records, native_checks = [], [], []
    docs = nlp().pipe([case["text"] for case in cases])
    with (args.output / "records.jsonl").open("x") as stream:
        for offset in range(0, len(cases), 32):
            chunk = list(
                zip(
                    cases[offset : offset + 32],
                    senses[offset : offset + 32],
                    [next(docs) for _ in cases[offset : offset + 32]],
                )
            )
            tokens = [
                target_token(doc, case["target"], row["start"])
                for case, row, doc in chunk
            ]
            assert all(
                token is not None and token.idx == row["start"]
                for token, (_, row, _) in zip(tokens, chunk)
            )
            vectors = (
                [[None]] * len(chunk)
                if args.rewritten
                else windowed_word_vectors(
                    [
                        ([token.text for token in doc], [target.i])
                        for target, (_, _, doc) in zip(tokens, chunk)
                    ]
                )
            )
            for index, (case, row, doc), token, vector in zip(
                range(offset, offset + len(chunk)), chunk, tokens, vectors
            ):
                if args.rewritten:
                    case.update(
                        start=row["start"], sense_distribution=row["sense_distribution"]
                    )
                    scored = rewritten_sense_contrast(case, sense_model)
                    values = [
                        scored[candidate["word"]] for candidate in case["candidates"]
                    ]
                else:
                    values = (
                        scorer.score(
                            vector[0],
                            row["sense_distribution"],
                            case["target_info"][1],
                            case["candidates"],
                        )
                        if case["candidates"]
                        else []
                    )
                assert len(values) == len(case["candidates"]) and all(
                    np.isfinite(value).all() for value in values
                )
                extra.extend(values)
                record = {
                    "id": case["id"],
                    "start": row["start"],
                    "scores": {
                        candidate["word"]: value.tolist()
                        for candidate, value in zip(case["candidates"], values)
                    },
                }
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                records.append(record)
                if index in (0, 464, 928) and case["candidates"]:
                    fresh = E.target_senses(
                        E.Slot(doc, token, target=case["target_info"]), sense_model
                    )
                    if args.rewritten:
                        scored = rewritten_sense_contrast(
                            {**case, "sense_distribution": fresh}, sense_model
                        )
                        native = [
                            scored[candidate["word"]]
                            for candidate in case["candidates"]
                        ]
                    else:
                        native = scorer.score(
                            windowed_word_vector(
                                [token.text for token in doc], token.i
                            ),
                            fresh,
                            case["target_info"][1],
                            case["candidates"],
                        )
                    error = (
                        float(np.max(np.abs(np.asarray(native) - values)))
                        if values
                        else 0.0
                    )
                    assert error < 1e-5, (case["id"], error)
                    native_checks.append({"id": case["id"], "max_feature_error": error})
            print(
                json.dumps(
                    {
                        "completed_cases": len(records),
                        "total_cases": len(cases),
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    frozen = np.load(features_path, allow_pickle=False)
    X, y, groups = (frozen[name] for name in ("similarity", "labels", "groups"))
    keys = [
        (case["id"], candidate["word"])
        for case in cases
        for candidate in case["candidates"]
    ]
    assert json.loads((features_path.parent / "feature-keys.json").read_text()) == [
        list(key) for key in keys
    ]
    np.testing.assert_array_equal(
        y, [candidate["gold"] for case in cases for candidate in case["candidates"]]
    )
    np.testing.assert_array_equal(
        groups, [case["target"].lower() for case in cases for _ in case["candidates"]]
    )
    matrix = np.c_[X, np.asarray(extra)]
    assert X.shape == (len(keys), 52) and matrix.shape[1] == 52 + len(
        CONTRAST_FEATURE_NAMES
    )
    np.testing.assert_array_equal(matrix[:, :52], X)
    np.savez_compressed(
        args.output / "feature-tables.npz",
        similarity=X,
        contrast=matrix,
        labels=y,
        groups=groups,
    )
    (args.output / "feature-keys.json").write_text(json.dumps(keys) + "\n")
    repeats = []
    for seed in range(3):
        old = np.load(
            features_path.parent / f"similarity-seed{seed}.npz", allow_pickle=False
        )
        folds = old["folds"]
        row = {"seed": seed}
        assignment = dict(
            zip(
                np.unique(groups),
                np.random.default_rng(seed).permutation(len(np.unique(groups))) % 10,
            )
        )
        np.testing.assert_array_equal(folds, [assignment[group] for group in groups])
        selected = {}
        for name, table in [("similarity", X), ("contrast", matrix)]:
            probability = np.zeros(len(y))
            for fold in range(10):
                train, dev = folds != fold, folds == fold
                probability[dev] = (
                    R.AverageDecision()
                    .fit(table[train], y[train])
                    .predict_proba(table[dev])[:, 1]
                )
            if name == "similarity":
                np.testing.assert_allclose(
                    probability, old["probability"], atol=1e-12, rtol=0
                )
            row[name], selected[name] = choose_scores(
                cases, dict(zip(keys, probability)), 0, 0.45, 0.10
            )
            row[name]["log_loss"] = float(
                log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6))
            )
            np.savez_compressed(
                args.output / f"{name}-seed{seed}.npz",
                probability=probability,
                folds=folds,
            )
            print(json.dumps({"seed": seed, "variant": name, **row[name]}), flush=True)
        (args.output / f"selections-seed{seed}.json").write_text(
            json.dumps(
                {
                    case["id"]: {
                        name: None
                        if selected[name][case["id"]] is None
                        else {
                            "word": selected[name][case["id"]]["word"],
                            "gold": bool(selected[name][case["id"]]["gold"]),
                        }
                        for name in selected
                    }
                    for case in cases
                },
                indent=2,
            )
            + "\n"
        )
        row["net_delta"] = row["contrast"]["net"] - row["similarity"]["net"]
        row["log_loss_delta"] = (
            row["contrast"]["log_loss"] - row["similarity"]["log_loss"]
        )
        repeats.append(row)
    net = float(np.mean([row["net_delta"] for row in repeats]))
    loss = float(np.mean([row["log_loss_delta"] for row in repeats]))
    result = {
        **manifest,
        "repeats": repeats,
        "mean_net_delta": net,
        "mean_log_loss_delta": loss,
        "noise_bar_passed": net >= 5 or loss <= -0.001,
        "saved_run": None,
        "native_checks": native_checks,
        "shape": list(matrix.shape),
        "records_sha256": sha256(args.output / "records.jsonl"),
        "feature_tables_sha256": sha256(args.output / "feature-tables.npz"),
        "elapsed_seconds": time.monotonic() - started,
    }
    if result["noise_bar_passed"]:
        relative = args.save_run.resolve().relative_to(ROOT)
        args.save_run.mkdir(parents=True)
        with (args.save_run / "decision.pkl").open("xb") as stream:
            pickle.dump(R.AverageDecision().fit(matrix, y), stream)
        shutil.copy2(base_run / "stacker.pkl", args.save_run / "stacker.pkl")
        config.update(
            decision=str(relative / "decision.pkl"),
            decision_features=manifest["variant"],
        )
        (args.save_run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        result["saved_run"] = str(relative)
        (args.save_run / "development.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "mean_net_delta",
                    "mean_log_loss_delta",
                    "noise_bar_passed",
                    "saved_run",
                    "elapsed_seconds",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
