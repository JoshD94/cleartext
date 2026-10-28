#!/usr/bin/env python3
"""Save four-beam outputs on a complete, already inspected development panel.

No fitting or policy comparison occurs here. Replay both policies on these same
saved beams afterward. Checkpoint and source fingerprints must stay unchanged.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, required=True)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--run", type=Path, required=True)
    args = cli.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    if len(rows) != 128 or len({r["id"] for r in rows}) != 128:
        raise ValueError("Retain the complete 128-input reused development panel")
    if any(not isinstance(r.get("source"), str) or not r["source"].strip() for r in rows):
        raise ValueError("Every row needs an unchanged source")
    files = [args.input, Path(__file__), *sorted((ROOT / "src").rglob("*.py")),
             *sorted(p for p in args.checkpoint.rglob("*") if p.is_file())]
    fingerprints = {str(p.resolve()): sha(p) for p in files}
    args.run.mkdir(parents=True, exist_ok=False)
    write(args.run / "freeze.json", {
        "files_sha256": fingerprints, "n": len(rows),
        "package_versions": {name: importlib.metadata.version(name) for name in (
            "transformers", "tokenizers", "torch", "spacy", "en-core-web-sm", "numpy")},
        "scope": "Reused development only. No fitting, holdout inference, semantic labels or promotion.",
        "decoding": "CandidateGenerator CPU, four deterministic beams; save identical raw pool for paired replay",
    })
    from cleartext.structure.search import CandidateGenerator
    import torch

    torch.set_num_threads(1)
    generator = CandidateGenerator(args.checkpoint, "cpu")
    started = time.monotonic()
    with (args.run / "dev.jsonl").open("x") as stream:
        for index, row in enumerate(rows):
            details = generator.predict(row["source"])
            if not details.get("model_candidates"):
                raise ValueError(f"No raw beams for {row['id']}; incomplete panel is not a result")
            stream.write(json.dumps({"id": row["id"], "source": row["source"],
                                     "reference": row.get("reference"), "search_details": details}) + "\n")
            stream.flush()
            write(args.run / "progress.json", {"completed": index + 1, "total": len(rows)})
            if (index + 1) % 8 == 0:
                print(f"generated {index + 1}/{len(rows)}", flush=True)
    if fingerprints != {str(p.resolve()): sha(p) for p in files}:
        raise ValueError("Input, code or checkpoint changed during generation")
    write(args.run / "completion.json", {
        "n": len(rows), "elapsed_seconds": time.monotonic() - started,
        "outputs_sha256": sha(args.run / "dev.jsonl"), "complete": True,
    })


if __name__ == "__main__":
    main()
