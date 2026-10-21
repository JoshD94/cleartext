"""Frozen grammar ablation and native six-block diagnostics on existing data."""

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
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from cleartext.audit import audit_case, load_cases, sha256
from cleartext.building_blocks import BuildingBlockClearText
from cleartext.data import ROOT
from cleartext.features import nlp, target_token
from cleartext.ensemble_pipeline import substitute
from cleartext.grammar_validation import GrammarValidator
from refine_relations import choose_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    cases = sum([pickle.load(path.open("rb"))[1] for path in paths], [])
    keys = [
        (case["id"], candidate["word"])
        for case in cases
        for candidate in case["candidates"]
    ]
    inputs = paths + [
        ROOT / f"outputs/detail-preservation-final-20261002/similarity-seed{seed}.npz"
        for seed in range(3)
    ]
    offsets_path = ROOT / "outputs/bert-candidate-coverage-20261002/records.jsonl"
    inputs.append(offsets_path)
    inputs += [
        ROOT / "diagnostics/domain-audit.jsonl",
        ROOT / "diagnostics/term-spans.jsonl",
        args.run / "config.json",
    ]
    sources = [Path(__file__).resolve()] + list((ROOT / "src/cleartext").glob("*.py"))
    manifest = {
        "scope": "Frozen grammar-only guard ablation on reused BenchLS development; ungraded native diagnostics.",
        "input_sha256": {
            str(path.resolve().relative_to(ROOT)): sha256(path) for path in inputs
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "training": False,
        "tsar_gold_used": False,
        "new_labels": False,
    }
    previous = json.loads(
        (
            ROOT / "outputs/domain-audit-context-similarity-20261002/manifest.json"
        ).read_text()
    )
    for path, expected in previous["artifact_sha256"].items():
        assert sha256(ROOT / path) == expected, path
    manifest["artifact_sha256"] = {
        **previous["artifact_sha256"],
        str((args.run / "stacker.pkl").resolve().relative_to(ROOT)): sha256(
            args.run / "stacker.pkl"
        ),
    }
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    docs = dict(
        zip(
            [case["id"] for case in cases], nlp().pipe([case["text"] for case in cases])
        )
    )
    offsets = [json.loads(line) for line in offsets_path.read_text().splitlines()]
    assert [case["id"] for case in cases] == [row["id"] for row in offsets]
    for case, offset in zip(cases, offsets):
        assert case["text"] == offset["text"] and case["target"] == offset["target"]
        token = target_token(docs[case["id"]], case["target"], offset["start"])
        assert token is not None and token.idx == offset["start"]
        for candidate in case["candidates"]:
            candidate["output"] = substitute(docs[case["id"]], token, candidate["word"])
    validator, reports, repeats = GrammarValidator(), {}, []
    for seed in range(3):
        probability = np.load(inputs[len(paths) + seed])["probability"]
        scores = dict(zip(keys, probability))
        baseline, _ = choose_scores(cases, scores, 0, 0.45, 0.10)
        guarded = []
        for case in cases:
            eligible = sorted(
                [
                    candidate
                    for candidate in case["candidates"]
                    if candidate["guard"]
                    and candidate["x"][0] >= 0.10
                    and scores[case["id"], candidate["word"]] >= 0.45
                ],
                key=lambda candidate: -scores[case["id"], candidate["word"]],
            )
            retained = []
            for candidate in eligible:
                key = case["id"], candidate["word"]
                if key not in reports:
                    reports[key] = validator.validate(
                        docs[case["id"]], candidate["output"]
                    )
                if reports[key]["pass"]:
                    retained = [candidate]
                    break
            guarded.append({**case, "candidates": retained})
        metrics, selections = choose_scores(guarded, scores, 0, 0.45, 0.10)
        row = {
            "seed": seed,
            "baseline": baseline,
            "grammar_guard": metrics,
            "net_delta": metrics["net"] - baseline["net"],
        }
        repeats.append(row)
        (args.output / f"grammar-selections-seed{seed}.json").write_text(
            json.dumps(
                {
                    key: None if value is None else value["word"]
                    for key, value in selections.items()
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(row), flush=True)
    with (args.output / "grammar-reports.jsonl").open("x") as stream:
        for (case, word), report in reports.items():
            stream.write(json.dumps({"id": case, "word": word, **report}) + "\n")
    pipe = BuildingBlockClearText.load(args.run)
    diagnostics = load_cases(ROOT / "diagnostics/domain-audit.jsonl", 13) + load_cases(
        ROOT / "diagnostics/term-spans.jsonl", 6
    )
    records = []
    with (args.output / "records.jsonl").open("x") as stream:
        for case in diagnostics:
            record = audit_case(pipe, case)
            assert all(
                field in record
                for field in [
                    "grammar_validation",
                    "document_consistency",
                    "jargon_explanations",
                    "edit_quality",
                    "term_spans",
                ]
            )
            stream.write(json.dumps(record, default=float) + "\n")
            stream.flush()
            records.append(record)
            print(
                json.dumps(
                    {
                        "id": record["id"],
                        "output": record["output"],
                        "glosses": len(record["jargon_explanations"]),
                        "rolled_back": record["block_rollbacks"],
                    }
                ),
                flush=True,
            )
    summary = {
        "repeats": repeats,
        "mean_net_delta": float(np.mean([row["net_delta"] for row in repeats])),
        "grammar_candidates_checked": len(reports),
        "grammar_candidates_rejected": sum(not r["pass"] for r in reports.values()),
        "native_cases": len(records),
        "native_rollbacks": sum(bool(r["block_rollbacks"]) for r in records),
        "gloss_annotations": sum(len(r["jargon_explanations"]) for r in records),
        "gloss_statuses": {
            status: sum(
                sum(g["status"] == status for g in r["jargon_explanations"])
                for r in records
            )
            for status in ("available", "ambiguous", "unavailable")
        },
        "semantic_certified": False,
        "default_changed": False,
        "model_weights_changed": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
