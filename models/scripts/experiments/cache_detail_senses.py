"""Cache full contextual sense distributions for existing development targets."""

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
from cleartext.data import ROOT
from cleartext.ensemble_pipeline import build_senses
from cleartext.features import nlp, target_token
from cleartext.generation import relation_map
from cleartext.technical_terms import detect_terms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "runs/ensemble-context-similarity-20261002/config.json"
    config = json.loads(config_path.read_text())
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    offsets_path = ROOT / "outputs/bert-candidate-coverage-20261002/records.jsonl"
    artifacts = [
        config_path,
        ROOT / config["sense_model"],
        ROOT / config["bert_sense"]["vectors"],
    ]
    sources = [Path(__file__).resolve()] + [
        ROOT / "src/cleartext" / name
        for name in (
            "detail_preservation.py",
            "technical_terms.py",
            "ensemble.py",
            "ensemble_pipeline.py",
            "context_window.py",
            "contextual.py",
            "generation.py",
            "lexical.py",
            "features.py",
        )
    ]
    manifest = {
        "scope": "All BenchLS is reused development. No labels used for sense scoring or term detection.",
        "input_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in paths + [offsets_path] + artifacts
        },
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "config": config,
        "meaning_parity_tolerance": 1e-5,
        "tsar_gold_used": False,
    }
    records_path = args.output / "records.jsonl"
    if args.resume:
        assert json.loads((args.output / "manifest.json").read_text()) == manifest
        assert not (args.output / "summary.json").exists(), "cache already completed"
        completed = [json.loads(line) for line in records_path.read_text().splitlines()]
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        completed = []
        for source in sources:
            destination = args.output / "source" / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    cases = sum([pickle.load(path.open("rb"))[1] for path in paths], [])
    offsets = [json.loads(line) for line in offsets_path.read_text().splitlines()]
    assert [c["id"] for c in cases] == [r["id"] for r in offsets]
    assert [c["id"] for c in cases[: len(completed)]] == [r["id"] for r in completed]
    with (ROOT / config["sense_model"]).open("rb") as stream:
        sense, _ = build_senses(pickle.load(stream), config)
    started = time.monotonic()
    remaining = cases[len(completed) :]
    with records_path.open("a" if args.resume else "x") as stream:
        for case, offset, doc in zip(
            remaining,
            offsets[len(completed) :],
            nlp().pipe([c["text"] for c in remaining]),
        ):
            assert case["text"] == offset["text"] and case["target"] == offset["target"]
            token = target_token(doc, case["target"], offset["start"])
            assert token is not None and token.idx == offset["start"]
            slot = E.Slot(doc, token, target=case["target_info"])
            distribution = {
                name: float(value)
                for name, value in E.target_senses(slot, sense).items()
            }
            assert not distribution or np.isclose(sum(distribution.values()), 1.0)
            relations = relation_map(slot.lemma, slot.pos)
            errors = []
            for candidate in case["candidates"]:
                origins = relations[candidate["lemma"].replace(" ", "_")][1]
                expected = sum(distribution.get(name, 0.0) for name in origins)
                errors.append(abs(expected - float(candidate["x"][0])))
            error = max(errors, default=0.0)
            if error > 1e-5:
                raise ValueError(f"{case['id']}: cached/live meaning drift {error}")
            spans = detect_terms(doc)
            blocked = any(
                span.protected
                and span.start < token.idx + len(token)
                and token.idx < span.end
                for span in spans
            )
            row = {
                "id": case["id"],
                "start": token.idx,
                "sense_distribution": distribution,
                "term_spans": [span.record() for span in spans],
                "protected_target": blocked,
                "meaning_parity_max_error": error,
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            completed.append(row)
            if len(completed) % 32 == 0 or len(completed) == len(cases):
                print(
                    json.dumps(
                        {
                            "completed_cases": len(completed),
                            "total_cases": len(cases),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    summary = {
        **manifest,
        "cases": len(completed),
        "protected_targets": sum(r["protected_target"] for r in completed),
        "proposal_spans": sum(
            sum(not span["protected"] for span in r["term_spans"]) for r in completed
        ),
        "meaning_parity_max_error": max(
            r["meaning_parity_max_error"] for r in completed
        ),
        "records_sha256": sha256(records_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
