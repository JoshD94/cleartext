#!/usr/bin/env python3
"""Score authored diagnostics and an already inspected search run as development."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.structure.common import normalize, sari
from cleartext.structure.corpus import write_json
from cleartext.structure.meaning import MeaningScorer, classify, sha256


def evaluate(args):
    if args.output.exists():
        raise FileExistsError("Diagnostics need a fresh directory")
    args.output.mkdir(parents=True)
    classifier = json.loads(args.classifier.read_text())
    scorer = MeaningScorer(args.cache / "model")
    if classifier["nli_model_sha256"] != scorer.model_sha256:
        raise ValueError("Mismatched NLI checkpoint")
    write_json(
        args.output / "freeze.json",
        {
            "classifier_sha256": sha256(args.classifier),
            "authored_inputs_sha256": sha256(args.authored),
            "script_sha256": sha256(__file__),
            "model_sha256": scorer.model_sha256,
            "purpose": "Development diagnostics only, no human ratings",
        },
    )
    authored = []
    for row in json.loads(args.authored.read_text()):
        scores = scorer.score(row["source"], row["candidate"])
        authored.append(
            {**row, "scores": scores, "decision": classify(scores, classifier)}
        )
    write_json(args.output / "authored.json", authored)
    summary = {
        "purpose": "Development diagnostics only, not independent simplification quality",
        "new_human_ratings": 0,
        "authored": {
            "n": len(authored),
            "match_authored_expectation": sum(
                r["decision"]["accepted"] == r["expected_equivalence"] for r in authored
            ),
            "false_accepts": [
                r["id"]
                for r in authored
                if r["decision"]["accepted"] and not r["expected_equivalence"]
            ],
            "false_rejections": [
                r["id"]
                for r in authored
                if not r["decision"]["accepted"] and r["expected_equivalence"]
            ],
        },
        "previous_outputs": {},
    }
    cached = {}
    for relative in (
        "dev-v2/dev.jsonl",
        "test-v2/test.jsonl",
        "dev-v2/domain_dev.jsonl",
    ):
        path = args.prior_run / relative
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        aggregate = {
            "input_sha256": sha256(path),
            "n": len(rows),
            "label": "Previously inspected; now development",
        }
        records = []
        for method in ("baseline", "search"):
            counts, before, after = Counter(), [], []
            for row in rows:
                candidate = row[method]
                changed = normalize(candidate) != normalize(row["source"])
                output = candidate
                if changed:
                    counts["changed_before"] += 1
                    key = (row["source"], candidate)
                    if key not in cached:
                        scores = scorer.score(*key)
                        cached[key] = {
                            "scores": scores,
                            "decision": classify(scores, classifier),
                        }
                    value = cached[key]
                    if not value["decision"]["accepted"]:
                        output = row["source"]
                        counts["rolled_back"] += 1
                    records.append(
                        {
                            "id": row["id"],
                            "method": method,
                            "source": row["source"],
                            "candidate": candidate,
                            **value,
                        }
                    )
                counts["changed_after"] += normalize(output) != normalize(row["source"])
                if "reference" in row:
                    before.append(sari(row["source"], candidate, row["reference"]))
                    after.append(sari(row["source"], output, row["reference"]))
            aggregate[method] = dict(counts)
            if before:
                aggregate[method]["development_sari_before"] = float(np.mean(before))
                aggregate[method]["development_sari_after"] = float(np.mean(after))
        summary["previous_outputs"][relative] = aggregate
        write_json(
            args.output / (path.parent.name + "-" + path.stem + ".json"), records
        )
    write_json(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "classifier", "authored", "prior-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(evaluate(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
