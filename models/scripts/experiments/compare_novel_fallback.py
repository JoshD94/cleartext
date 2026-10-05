"""Paired frozen BenchLS comparison and a fixed, unpromoted shadow fallback."""

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
from cleartext import ensemble as E
from cleartext.audit import sha256
from cleartext.context_similarity import replacement_similarities
from cleartext.data import ROOT
from cleartext.features import nlp, target_token
from cleartext.novel_context import (
    FEATURE_NAMES,
    SIMILARITY_FEATURE_NAMES,
    choose_novel,
    context_features,
    validate_feature_names,
)
from refine_relations import choose_scores


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def metrics(selections, cases):
    edits = sum(value is not None for value in selections.values())
    correct = sum(
        value is not None and value["word"].lower() in case["gold"]
        for case in cases
        for value in [selections[case["id"]]]
    )
    return {
        "edits": edits,
        "correct": correct,
        "wrong": edits - correct,
        "net": 2 * correct - edits,
        "precision": correct / edits if edits else 0.0,
        "coverage": edits / len(cases),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validator-run", type=Path, required=True)
    parser.add_argument(
        "--reuse-context-records",
        type=Path,
        help="Verified earlier comparison; reuse its raw seven columns, not predictions.",
    )
    args = parser.parse_args()
    validator = args.validator_run.resolve()
    development = json.loads((validator / "development.json").read_text())
    config = json.loads((validator / "config.json").read_text())
    names = validate_feature_names(config["feature_names"])
    assert sha256(validator / "validator.pkl") == development["validator_sha256"]
    model = pickle.load((validator / "validator.pkl").open("rb"))
    assert model.n_features_in_ == len(names)
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    cases = sum([pickle.load(path.open("rb"))[1] for path in paths], [])
    raw_path = ROOT / "outputs/bert-candidate-coverage-20261002/records.jsonl"
    review_path = ROOT / "outputs/proposal-validation-20261003/review.jsonl"
    raw, review = read_rows(raw_path), read_rows(review_path)
    assert (
        [case["id"] for case in cases]
        == [row["id"] for row in raw]
        == [row["id"] for row in review]
    )
    baseline = ROOT / "outputs/detail-preservation-final-20261002"
    sources = list((ROOT / "src/cleartext").glob("*.py")) + [
        Path(__file__).resolve(),
        Path(__file__).with_name("refine_relations.py"),
    ]
    inputs = paths + [
        raw_path,
        review_path,
        validator / "config.json",
        validator / "validator.pkl",
        validator / "development.json",
    ]
    inputs += [baseline / f"similarity-seed{seed}.npz" for seed in range(3)] + [
        baseline / "summary.json"
    ]
    cached_rows = None
    if args.reuse_context_records:
        cached_directory = args.reuse_context_records.resolve()
        cached_summary = json.loads((cached_directory / "summary.json").read_text())
        cached_path = cached_directory / "records.jsonl"
        assert sha256(cached_path) == cached_summary["records_sha256"]
        for path in paths + [raw_path, review_path]:
            assert (
                sha256(path)
                == cached_summary["input_sha256"][str(path.relative_to(ROOT))]
            )
        cached_rows = read_rows(cached_path)
        assert [row["id"] for row in cached_rows] == [case["id"] for case in cases]
        inputs += [cached_path, cached_directory / "summary.json"]
    manifest = {
        "scope": "Reused BenchLS development. Paired fixed folds and baseline predictions, no fitting on BenchLS.",
        "safe_threshold": config["context_threshold"],
        "shadow_threshold": 0.50,
        "shadow_policy": "Fixed 0.50 diagnostic threshold declared before this comparison. It does not bypass the live disabled gate.",
        "candidate_limit": 3,
        "fallback_only": True,
        "meaning_accuracy": None,
        "feature_names": list(names),
        "raw_context_columns_reused": cached_rows is not None,
        "default_changed": False,
        "deployment_enabled": False,
        "new_labels": False,
        "tsar_used": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "adoption_bar": "SWORDS context gate passes, safe mean net +5 and mean listed precision at least frozen baseline. Domain audit required separately.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    scored, native_checks, started = [], [], time.monotonic()
    with (args.output / "records.jsonl").open("x") as stream:
        for index, (case, origin, record, doc) in enumerate(
            zip(cases, raw, review, nlp().pipe([case["text"] for case in cases]))
        ):
            token = target_token(doc, case["target"], origin["start"])
            assert token is not None and token.idx == origin["start"]
            # These conditions do not read benchmark membership or baseline probabilities.
            eligible = [
                row.copy()
                for row in record["proposals"]
                if row["validation"]["dictionary_valid"]
                and row["mechanical_checks_pass"]
                and row["complexity_delta"] > 0
                and row["wordnet_evidence"]["source_support_by_relation"]["antonym"]
                < 0.10
            ]
            if eligible:
                if cached_rows is None:
                    features = context_features(
                        E.Slot(doc, token, target=origin["target_info"]),
                        eligible,
                        names,
                    )
                else:
                    cached = cached_rows[index]
                    assert (
                        cached["text"] == case["text"]
                        and cached["start"] == origin["start"]
                    )
                    assert cached["target"] == case["target"]
                    excluded_fields = {"context_fit", "raw_context_features"}
                    assert [
                        {
                            key: value
                            for key, value in row.items()
                            if key not in excluded_fields
                        }
                        for row in cached["proposals"]
                    ] == eligible
                    features = np.asarray(
                        [row["raw_context_features"] for row in cached["proposals"]]
                    )
                    assert features.shape == (len(eligible), len(FEATURE_NAMES))
                    if names == SIMILARITY_FEATURE_NAMES:
                        similarity = replacement_similarities(
                            case["text"],
                            token.idx,
                            token.idx + len(token),
                            [row["word"] for row in eligible],
                        )
                        features = np.column_stack([features, similarity])
                probability = model.predict_proba(features)[:, 1]
                for row, values, score in zip(eligible, features, probability):
                    row["context_fit"] = float(score)
                    row["raw_context_features"] = values.tolist()
                if (
                    len(native_checks) < 3
                    and index >= len(native_checks) * len(cases) // 3
                ):
                    fresh = context_features(
                        E.Slot(doc, token, target=origin["target_info"]),
                        [eligible[0]],
                        names,
                    )[0]
                    error = float(np.max(np.abs(fresh - features[0])))
                    assert error < 1e-5
                    native_checks.append(
                        {"id": case["id"], "maximum_batch_feature_difference": error}
                    )
            row = {
                "id": case["id"],
                "text": case["text"],
                "target": case["target"],
                "start": origin["start"],
                "proposals": eligible,
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            scored.append(row)
            if (index + 1) % 64 == 0 or index + 1 == len(cases):
                print(
                    json.dumps(
                        {
                            "completed_cases": index + 1,
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    keys = [
        (case["id"], candidate["word"])
        for case in cases
        for candidate in case["candidates"]
    ]
    previous = json.loads((baseline / "summary.json").read_text())
    repeats = []
    for seed in range(3):
        array = np.load(baseline / f"similarity-seed{seed}.npz", allow_pickle=False)
        base_metrics, selected = choose_scores(
            cases, dict(zip(keys, array["probability"])), 0, 0.45, 0.10
        )
        for field in ("edits", "correct", "wrong", "net"):
            assert base_metrics[field] == previous["repeats"][seed]["similarity"][field]
        choices = {"baseline": selected, "safe": {}, "shadow": {}}
        for case, row in zip(cases, scored):
            for variant, threshold in (
                ("safe", config["context_threshold"]),
                ("shadow", 0.50),
            ):
                choices[variant][case["id"]] = selected[case["id"]] or choose_novel(
                    row["proposals"], threshold
                )
        repeat = {
            "seed": seed,
            **{name: metrics(values, cases) for name, values in choices.items()},
        }
        repeat["shadow_net_delta"] = repeat["shadow"]["net"] - repeat["baseline"]["net"]
        repeat["shadow_precision_delta"] = (
            repeat["shadow"]["precision"] - repeat["baseline"]["precision"]
        )
        repeat["safe_net_delta"] = repeat["safe"]["net"] - repeat["baseline"]["net"]
        repeat["safe_precision_delta"] = (
            repeat["safe"]["precision"] - repeat["baseline"]["precision"]
        )
        repeats.append(repeat)
        (args.output / f"selections-seed{seed}.json").write_text(
            json.dumps(
                {
                    case["id"]: {
                        name: None
                        if values[case["id"]] is None
                        else {
                            "word": values[case["id"]]["word"],
                            "listed_match": values[case["id"]]["word"].lower()
                            in case["gold"],
                        }
                        for name, values in choices.items()
                    }
                    for case in cases
                },
                indent=2,
            )
            + "\n"
        )
    net = float(np.mean([row["shadow_net_delta"] for row in repeats]))
    precision = float(np.mean([row["shadow_precision_delta"] for row in repeats]))
    safe_net = float(np.mean([row["safe_net_delta"] for row in repeats]))
    safe_precision = float(np.mean([row["safe_precision_delta"] for row in repeats]))
    shadow = args.output / "shadow-run"
    shadow.mkdir()
    shutil.copy2(validator / "validator.pkl", shadow / "validator.pkl")
    config.update(context_threshold=0.50, shadow_only=True, deployment_enabled=False)
    (shadow / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    result = {
        **manifest,
        "cases": len(cases),
        "scored_novel_rows": sum(len(row["proposals"]) for row in scored),
        "repeats": repeats,
        "mean_shadow_net_delta": net,
        "mean_shadow_precision_delta": precision,
        "mean_safe_net_delta": safe_net,
        "mean_safe_precision_delta": safe_precision,
        "context_gate_passed": manifest["safe_threshold"] is not None,
        "adoption_bar_passed": manifest["safe_threshold"] is not None
        and safe_net >= 5
        and safe_precision >= 0,
        "native_checks": native_checks,
        "records_sha256": sha256(args.output / "records.jsonl"),
        "shadow_run": str(shadow.resolve().relative_to(ROOT)),
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "scored_novel_rows",
                    "repeats",
                    "mean_shadow_net_delta",
                    "mean_shadow_precision_delta",
                    "adoption_bar_passed",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
