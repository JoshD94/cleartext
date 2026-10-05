"""Prepare a bounded, ungraded parliamentary audit from existing CompLex text."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample(path, limit):
    if not 1 <= limit <= 30:
        raise ValueError("limit must be between 1 and 30")
    records, seen = [], set()
    with path.open() as stream:
        for index, row in enumerate(csv.DictReader(stream, delimiter="\t"), 1):
            if row["corpus"] != "europarl" or row["sentence"] in seen:
                continue
            seen.add(row["sentence"])
            records.append(
                {
                    "id": f"complex-europarl-{index}",
                    "domain": "political_science",
                    "text": row["sentence"],
                    "source": {
                        "kind": "corpus",
                        "path": str(path.relative_to(ROOT)),
                        "dataset": "CompLex",
                        "corpus": "europarl",
                        "split": "test",
                        "record_index": index,
                        "attribution": "Existing CompLex test text, europarl corpus. First distinct sentences in source order. "
                        "Complexity ratings and annotated targets are excluded. "
                        "This split was already used for component evaluation; it is not a fresh pipeline test.",
                    },
                }
            )
            if len(records) == limit:
                break
    if len(records) != limit:
        raise ValueError("not enough distinct parliamentary sentences")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    path = ROOT / "data/raw/complex_test.tsv"
    records = sample(path, args.limit)
    args.output.mkdir(parents=True, exist_ok=False)
    payload = "".join(json.dumps(record) + "\n" for record in records)
    (args.output / "input.jsonl").write_text(payload)
    commit = ROOT / "data/raw/CompLex.commit.json"
    manifest = {
        "scope": "Existing real parliamentary text; ungraded robustness audit, no semantic labels.",
        "source_path": str(path.relative_to(ROOT)),
        "source_sha256": sha256(path),
        "dataset_commit": json.loads(commit.read_text())["sha"],
        "commit_receipt_sha256": sha256(commit),
        "script_sha256": sha256(Path(__file__).resolve()),
        "selection": "First N distinct europarl sentences in source order",
        "records": len(records),
        "input_sha256": sha256(args.output / "input.jsonl"),
        "ratings_used": False,
        "annotated_targets_used": False,
        "fresh_test": False,
        "new_labels": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"records": len(records), "ratings_used": False, "fresh_test": False}
        )
    )


if __name__ == "__main__":
    main()
