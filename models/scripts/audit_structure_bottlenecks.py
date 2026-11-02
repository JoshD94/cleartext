#!/usr/bin/env python3
"""Audit previously inspected Cleartext runs as development diagnostics."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.features import nlp
from cleartext.structure.audit import (
    beam_audit,
    exposure,
    forced_losses,
    funnel,
    reference_audit,
)
from cleartext.structure.corpus import write_json
from cleartext.structure.search import checked


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--teacher-forcing", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    expanded = args.state / "structure-expanded-20261026/run"
    search = args.state / "structure-search-20261027"
    inputs = {
        "data": expanded / "data.json",
        "training": expanded / "seq2seq_training.json",
        "old_dev": expanded / "dev-evaluation/outputs.jsonl",
        "old_test": expanded / "test-evaluation/outputs.jsonl",
        "current_dev": search / "dev-v2/dev.jsonl",
        "current_test": search / "test-v2/test.jsonl",
        "probes": args.probes,
    }
    checkpoint = expanded / "seq2seq"
    sources = list(
        (Path(__file__).resolve().parents[1] / "src/cleartext/structure").glob("*.py")
    )
    write_json(
        args.output / "freeze.json",
        {
            "purpose": "Previously inspected material, development diagnosis only; no training or promotion",
            "inputs": {
                k: {"path": str(p), "sha256": sha(p)} for k, p in inputs.items()
            },
            "source_sha256": {p.name: sha(p) for p in [Path(__file__), *sources]},
            "checkpoint_sha256": sha(checkpoint / "model.safetensors"),
            "teacher_forcing": args.teacher_forcing,
        },
    )
    data = json.loads(inputs["data"].read_text())
    training = json.loads(inputs["training"].read_text())
    summary = {
        "purpose": "Development diagnosis, no human quality labels",
        "exposure": exposure(data["train"], training),
    }
    summary["references"] = {}
    for split, rows in data.items():
        print("references", split, len(rows), flush=True)
        report, records = reference_audit(rows)
        summary["references"][split] = report
        write_json(args.output / f"reference_failures_{split}.json", records)
    summary["funnels"], summary["beams"], summary["losses"] = {}, {}, {}
    loaded = {
        name: read_rows(inputs[name])
        for name in ("old_dev", "old_test", "current_dev", "current_test")
    }
    for name, rows in loaded.items():
        print("saved_outputs", name, flush=True)
        old = name.startswith("old")
        summary["funnels"][name] = funnel(
            rows,
            "continued" if old else "baseline",
            "continued_details" if old else "baseline_details",
        )
        if not old:
            report, records = beam_audit(rows)
            summary["beams"][name] = report
            write_json(args.output / f"all_beams_{name}.json", records)
            if args.teacher_forcing:
                report, records = forced_losses(rows, str(checkpoint))
                summary["losses"][name] = report
                write_json(args.output / f"losses_{name}.json", records)
    before = {r["id"]: r for r in loaded["old_dev"]}
    now = loaded["current_dev"]
    summary["same_raw_dev_control"] = {
        "n": len(now),
        "same_source_and_raw": sum(
            r["source"] == before[r["id"]]["source"]
            and r["baseline_details"]["raw"]
            == before[r["id"]]["continued_details"]["raw"]
            for r in now
        ),
        "previous_changed_now_copy_ids": [
            r["id"]
            for r in now
            if before[r["id"]]["continued"] != before[r["id"]]["source"]
            and r["baseline"] == r["source"]
        ],
    }
    probes = []
    for probe in json.loads(args.probes.read_text()):
        probes.append(
            {
                **probe,
                "checks": checked(
                    probe["source"],
                    probe["candidate"],
                    nlp()(probe["source"]),
                    boundaries=False,
                ),
            }
        )
    write_json(args.output / "authored_probes.json", probes)
    summary["authored_probes"] = [
        {
            "id": p["id"],
            "expected_equivalence": p["expected_equivalence"],
            "failed": p["checks"]["failed"],
        }
        for p in probes
    ]
    write_json(args.output / "summary.json", summary)
    print("completed", args.output, flush=True)


if __name__ == "__main__":
    main()
