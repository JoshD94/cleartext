"""Cache a bounded contextual replacement score on existing BenchLS candidates."""

import argparse
import importlib.metadata
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from cleartext.audit import sha256
from cleartext.context_similarity import replacement_similarities
from cleartext.data import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    sources = [Path(__file__).resolve()] + [
        ROOT / "src/cleartext" / name
        for name in (
            "context_similarity.py",
            "contextual.py",
            "complete_slot.py",
            "prompted_relations.py",
        )
    ]
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    offset_path = ROOT / "outputs/bert-candidate-coverage-20261002/records.jsonl"
    manifest = {
        "scope": "Existing BenchLS development candidates. No labels used in scoring, fitting, gold generation or TSAR.",
        "input_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in paths + [offset_path]
        },
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "model": "bert-base-uncased",
        "layers": "last four mean",
        "limit": 256,
        "batch_size": 16,
        "packages": {
            p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")
        },
    }
    records_path = args.output / "records.jsonl"
    completed = []
    if args.resume:
        assert json.loads((args.output / "manifest.json").read_text()) == manifest
        completed = [json.loads(line) for line in records_path.read_text().splitlines()]
        assert not (args.output / "summary.json").exists(), "cache already completed"
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        for source in sources:
            destination = args.output / "source" / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    cases = sum([pickle.load(path.open("rb"))[1] for path in paths], [])
    offsets = [json.loads(line) for line in offset_path.read_text().splitlines()]
    assert [c["id"] for c in cases] == [r["id"] for r in offsets]
    for case, row in zip(cases, completed):
        assert row["id"] == case["id"] and list(row["scores"]) == [
            c["word"] for c in case["candidates"]
        ]
    started = time.monotonic()
    with records_path.open("a" if args.resume else "x") as stream:
        for index in range(len(completed), len(cases)):
            case, offset = cases[index], offsets[index]
            assert case["text"] == offset["text"] and case["target"] == offset["target"]
            start = offset["start"]
            assert (
                case["text"][start : start + len(case["target"])].casefold()
                == case["target"].casefold()
            )
            words = [c["word"] for c in case["candidates"]]
            scores = replacement_similarities(
                case["text"], start, start + len(case["target"]), words
            )
            assert len(scores) == len(words) and np.isfinite(scores).all()
            row = {
                "id": case["id"],
                "start": start,
                "scores": dict(zip(words, scores.tolist())),
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            completed.append(row)
            if (index + 1) % 16 == 0 or index + 1 == len(cases):
                print(
                    json.dumps(
                        {
                            "completed_cases": index + 1,
                            "total_cases": len(cases),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    # Fixed native checks, selected by cache position, independent of labels.
    checks = []
    for index in (0, len(cases) // 2, len(cases) - 1):
        case, row = cases[index], completed[index]
        words = [c["word"] for c in case["candidates"][:2]]
        start = row["start"]
        fresh = replacement_similarities(
            case["text"],
            start,
            start + len(case["target"]),
            [case["target"]] + words,
            batch_size=1,
        )
        np.testing.assert_allclose(fresh[0], 1.0, atol=1e-6)
        expected = [row["scores"][w] for w in words]
        np.testing.assert_allclose(fresh[1:], expected, atol=1e-5, rtol=1e-5)
        checks.append(
            {
                "id": case["id"],
                "identity_similarity": float(fresh[0]),
                "max_batch_difference": float(np.max(np.abs(fresh[1:] - expected)))
                if words
                else 0.0,
            }
        )
    summary = {
        **manifest,
        "cases": len(cases),
        "candidate_rows": sum(len(r["scores"]) for r in completed),
        "native_checks": checks,
        "records_sha256": sha256(records_path),
        "elapsed_seconds": time.monotonic() - started,
        "default_changed": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
