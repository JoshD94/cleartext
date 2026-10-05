"""Reproduce lost BERT targets and verify an opt-in target-centered window.

Only cached local weights and existing BenchLS input text are used. No fitting,
accuracy labels or TSAR data. Artificial padding is an ungraded boundary test.
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from dataclasses import asdict
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
from cleartext.audit import sha256
from cleartext.contextual import model, word_vectors
from cleartext.context_window import target_windows, windowed_word_vectors
from cleartext.data import ROOT, BENCHLS_SHA256
from cleartext.features import nlp, target_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-run", type=Path, default=ROOT / "runs/ensemble-relations-20261002"
    )
    parser.add_argument("--save-run", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.save_run.exists():
        raise FileExistsError("output and saved run must be new paths")
    run = args.save_run.resolve()
    run.relative_to(ROOT)
    config = json.loads((args.base_run / "config.json").read_text())
    if config["bert_sense"].get("target_window", False):
        raise ValueError("base run already enables target windows")
    args.output.mkdir(parents=True)
    tokenizer, _ = model()
    short = "Although this sentence is convoluted , the main idea is simple .".split()
    long = (
        ["record"] * 350
        + ["The", "algorithm", "encodes", "each", "message", "."]
        + ["record"] * 40
    )
    boundary = ["record"] * 253 + ["unaffordable", "database", "."] + ["record"] * 40
    inputs = [
        (short, [0, 4, 10]),
        (long, [352]),
        (boundary, [253]),
        (["bank", "[PAD]", "river"], [0, 2]),
    ]
    original = word_vectors(inputs)
    windowed = windowed_word_vectors(inputs)
    short_error = float(
        max(
            np.max(np.abs(a - b))
            for group in (0, 3)
            for a, b in zip(original[group], windowed[group])
        )
    )
    assert short_error < 5e-6, short_error
    assert np.linalg.norm(original[1][0]) == 0, (
        "boundary diagnostic no longer reproduces the original bug"
    )
    assert abs(np.linalg.norm(windowed[1][0]) - 1) < 1e-6
    assert abs(np.linalg.norm(windowed[2][0]) - 1) < 1e-6
    rows = []
    for case, ((words, indices), before, after) in enumerate(
        zip(inputs, original, windowed)
    ):
        for index, old, new, window in zip(
            indices, before, after, target_windows(tokenizer, words, indices)
        ):
            visible = sum(
                position + window.source_start - 1 < 254
                for position in window.target_positions
            )
            if case == 2:
                assert 0 < visible < len(window.target_positions), (
                    "boundary no longer splits the target"
                )
            rows.append(
                {
                    "case": case,
                    "words": words,
                    "target_index": index,
                    "target": words[index],
                    "old_vector_norm": float(np.linalg.norm(old)),
                    "new_vector_norm": float(np.linalg.norm(new)),
                    "old_visible_wordpieces": visible,
                    "target_wordpieces": len(window.target_positions),
                    "max_vector_difference": float(np.max(np.abs(old - new))),
                    "window": asdict(window),
                }
            )
    with (args.output / "boundary-diagnostics.json").open("x") as stream:
        json.dump(rows, stream, indent=2)
    np.savez_compressed(
        args.output / "vectors.npz",
        original=np.asarray([v for group in original for v in group]),
        windowed=np.asarray([v for group in windowed for v in group]),
    )

    archive = ROOT / "data/raw/BenchLS.zip"
    assert sha256(archive) == BENCHLS_SHA256
    with zipfile.ZipFile(archive) as source:
        records = [
            line.split("\t")[:3]
            for line in source.read("BenchLS/BenchLS.txt").decode().splitlines()
        ]
    affected, missing = [], []
    max_context_pieces = max_target_end = 0
    docs = nlp().pipe([fields[0] for fields in records], batch_size=128)
    for index, (fields, doc) in enumerate(zip(records, docs)):
        text, target, word_index = fields
        offset = sum(len(word) + 1 for word in text.split(" ")[: int(word_index)])
        token = target_token(doc, target, offset)
        if token is None:
            missing.append(f"benchls-{index}")
            continue
        words = [word.text for word in doc]
        window = target_windows(tokenizer, words, [token.i])[0]
        positions = [
            position + window.source_start - 1 for position in window.target_positions
        ]
        max_context_pieces = max(max_context_pieces, window.source_length)
        max_target_end = max(max_target_end, max(positions) + 1)
        kept = sum(position < 254 for position in positions)
        if window.source_start:
            affected.append(
                {
                    "id": f"benchls-{index}",
                    "text": text,
                    "target": target,
                    "start": token.idx,
                    "target_wordpieces": len(positions),
                    "old_visible_wordpieces": kept,
                    "window": asdict(window),
                }
            )
    (args.output / "benchls-changed-windows.json").write_text(
        json.dumps(affected, indent=2) + "\n"
    )
    counts = {
        "input_cases": len(records),
        "unresolved_targets": missing,
        "changed_windows": len(affected),
        "max_context_wordpieces": max_context_pieces,
        "max_target_wordpiece_end": max_target_end,
        "fully_lost_targets": sum(
            row["old_visible_wordpieces"] == 0 for row in affected
        ),
        "partially_lost_targets": sum(
            0 < row["old_visible_wordpieces"] < row["target_wordpieces"]
            for row in affected
        ),
    }
    result = {
        "scope": "Ungraded boundary tests and existing BenchLS input geometry only. "
        "No fit, no substitute-accuracy measurement, no TSAR.",
        "short_vector_max_abs_error": short_error,
        "lost_target_reproduced_and_corrected": True,
        "benchls": counts,
        "window_limit": 256,
        "default_changed": False,
        "base_run": str(args.base_run.resolve()),
        "saved_run": str(run),
        "input_sha256": {
            str(archive.relative_to(ROOT)): sha256(archive),
            str((args.base_run / "config.json").resolve().relative_to(ROOT)): sha256(
                args.base_run / "config.json"
            ),
        },
    }
    sources = [
        Path(__file__).resolve(),
        ROOT / "src/cleartext/context_window.py",
        ROOT / "src/cleartext/contextual.py",
        ROOT / "src/cleartext/ensemble_pipeline.py",
    ]
    result["source_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in sources
    }
    for path in sources:
        copy = args.output / "source" / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
    run.mkdir(parents=True)
    config["bert_sense"]["target_window"] = True
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(args.base_run / "stacker.pkl", run / "stacker.pkl")
    result["saved_artifact_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (run / "config.json", run / "stacker.pkl")
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (run / "window-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
