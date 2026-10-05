"""Verify complete BERT candidate scoring and inspect BenchLS input geometry."""

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
from cleartext import contextual
from cleartext.audit import sha256
from cleartext.complete_slot import complete_slot_scores, slot_inputs
from cleartext.data import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-run", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.save_run.exists():
        raise FileExistsError("output and run must be new")
    args.output.mkdir(parents=True)
    tokenizer, _ = contextual.model()
    text = "They recorded the sequence carefully."
    start = text.index("sequence")
    words = [
        "one two three four five six seven",
        "one two three four five six banana",
        "list",
        "unaffordable",
    ]
    legacy = contextual.slot_scores(text, start, start + 8, words)
    reference = contextual.slot_scores(text, start, start + 8, words, max_pieces=64)
    complete = complete_slot_scores(text, start, start + 8, words)
    assert legacy[0] == legacy[1]
    assert abs(complete[0] - complete[1]) > 0.001
    np.testing.assert_allclose(complete, reference, atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(complete[2:], legacy[2:], atol=1e-4, rtol=1e-5)
    long_text = "record " * 400 + "target" + " record" * 400
    long_input = slot_inputs(
        tokenizer,
        long_text,
        long_text.index("target"),
        long_text.index("target") + 6,
        ["one two three four five six seven"],
    )[0]
    assert len(long_input.pieces) == 7 and len(long_input.input_ids) <= 320
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    affected = []
    affected_targets = []
    total = 0
    for path in paths:
        _, cases = pickle.load(path.open("rb"))
        for case in cases:
            target_pieces = len(
                tokenizer(case["target"], add_special_tokens=False).input_ids
            )
            if target_pieces > 6:
                affected_targets.append(
                    {
                        "id": case["id"],
                        "target": case["target"],
                        "pieces": target_pieces,
                    }
                )
            for candidate in case["candidates"]:
                total += 1
                pieces = tokenizer(
                    candidate["word"], add_special_tokens=False
                ).input_ids
                if len(pieces) > 6:
                    affected.append(
                        {
                            "id": case["id"],
                            "target": case["target"],
                            "word": candidate["word"],
                            "pieces": len(pieces),
                        }
                    )
    base = ROOT / "runs/ensemble-windowed-20261002"
    config = json.loads((base / "config.json").read_text())
    run = args.save_run.resolve()
    run.relative_to(ROOT)
    run.mkdir(parents=True)
    config["bert_slot_complete"] = True
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(base / "stacker.pkl", run / "stacker.pkl")
    sources = [Path(__file__).resolve()] + [
        ROOT / "src/cleartext" / name
        for name in (
            "complete_slot.py",
            "contextual.py",
            "ensemble_pipeline.py",
            "ensemble.py",
        )
    ]
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    result = {
        "scope": "Ungraded scoring checks and BenchLS candidate geometry. No labels used, no fitting, no TSAR.",
        "text": text,
        "start": start,
        "end": start + 8,
        "words": words,
        "legacy_scores": legacy.tolist(),
        "untruncated_reference_scores": reference.tolist(),
        "complete_scores": complete.tolist(),
        "max_reference_error": float(np.max(np.abs(complete - reference))),
        "long_context_input_pieces": len(long_input.input_ids),
        "long_candidate_pieces_retained": len(long_input.pieces),
        "benchls_candidate_rows": total,
        "affected_rows": affected,
        "affected_targets": affected_targets,
        "default_changed": False,
        "saved_run": str(run.relative_to(ROOT)),
        "source_sha256": {
            str(source.relative_to(ROOT)): sha256(source) for source in sources
        },
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in paths + [base / "config.json"]
        },
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (run / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
