"""Require the frozen input matrix before comparing grammar/pipeline variants.

Comparing systems only against each other cannot detect the same missing input
in every system. This check uses the independent frozen passages as authority.
It reads metadata and saved predictions only, without loading any models.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def validate_matrix(passages, rows, systems):
    if not systems or len(set(systems)) != len(systems):
        raise ValueError("Expected systems must be nonempty and unique")
    expected = {p["id"]: p for p in passages}
    if not passages or len(expected) != len(passages):
        raise ValueError("Frozen passages must have unique IDs and be nonempty")
    actual = {}
    for row in rows:
        key = row["system"], row["id"]
        if key in actual:
            raise ValueError(f"Duplicate prediction: {key}")
        if row["system"] not in systems or row["id"] not in expected:
            raise ValueError(f"Unexpected prediction: {key}")
        source = expected[row["id"]]
        for field in ("original", "panel", "references"):
            if row.get(field) != source.get(field, [] if field == "references" else None):
                raise ValueError(f"Prediction differs from frozen {field}: {key}")
        if not isinstance(row.get("output"), str):
            raise ValueError(f"Prediction output is not text: {key}")
        actual[key] = row
    missing = [(system, p["id"]) for system in systems for p in passages
               if (system, p["id"]) not in actual]
    if missing:
        raise ValueError(f"Missing {len(missing)} frozen system/input pairs; first: {missing[0]}")
    return {"input_n": len(passages), "system_n": len(systems), "prediction_n": len(rows),
            "panel_counts": dict(Counter(p["panel"] for p in passages)),
            "systems": list(systems), "complete": True,
            "limits": "Checks input/reference alignment and completeness only. No output quality verdict or accuracy labels."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--systems", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    passages = json.loads(args.passages.read_text())
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    result = validate_matrix(passages, rows, args.systems.split(","))
    result["files"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (args.passages, args.predictions, Path(__file__))}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
