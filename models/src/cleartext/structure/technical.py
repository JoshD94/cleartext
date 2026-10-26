"""Small reproducible technical-text audit, without invented reference rewrites."""

import argparse
import hashlib
import json
import pickle
import re
import urllib.request
from pathlib import Path

from ..features import nlp
from ..rewrites import structural
from .common import normalize, structure_checks
from .corpus import write_json
from .experiment import source_provenance


def prepare_inputs(cache, output):
    """First six 20..70-word sentences in each RFC introduction.

    Select before model predictions. Keep source receipts and input hashes.
    Only whitespace and RFC page furniture are removed. No gold or ratings.
    """
    cache, output = Path(cache), Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for number in (9110, 8446):
        url = f"https://www.rfc-editor.org/rfc/rfc{number}.txt"
        path = cache / f"rfc{number}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        text = path.read_text(encoding="utf-8-sig")
        start = re.search(r"(?m)^1\. +Introduction\s*$", text)
        if start is None:
            raise ValueError(f"Missing RFC {number} introduction")
        text = text[start.end() :]
        end = re.search(r"(?m)^2\. +", text)
        text = text[: end.start()] if end else text
        lines = [
            line.strip()
            for line in text.splitlines()
            if not re.search(r"\[Page \d+\]|^RFC \d+|^1\.\d+\.", line.strip())
        ]
        cleaned = re.sub(r"(?<=\w)-\n(?=\w)", "-", "\n".join(lines))
        paragraphs = re.split(r"\n\s*\n", cleaned)
        selected = []
        for paragraph in paragraphs:
            if (
                not paragraph.strip()
                or paragraph.strip().endswith(":")
                or paragraph.strip().startswith(("-", "*"))
            ):
                continue
            for sentence in nlp()(" ".join(paragraph.split())).sents:
                value = sentence.text.strip()
                if 20 <= len(value.split()) <= 70 and value.endswith("."):
                    selected.append(value)
                    if len(selected) == 6:
                        break
            if len(selected) == 6:
                break
        if len(selected) != 6:
            raise ValueError(f"Not enough eligible sentences in RFC {number}")
        for i, value in enumerate(selected):
            rows.append(
                {
                    "id": f"rfc{number}-intro-{i}",
                    "source": value,
                    "source_url": url,
                    "section": "1. Introduction, including subsections",
                    "document_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "input_sha256": hashlib.sha256(value.encode()).hexdigest(),
                    "label_status": "Real technical text, no reference simplification or human rating",
                }
            )
    write_json(output, rows)
    return rows


def audit(run, inputs, device="cpu"):
    run = Path(run)
    output = run / "technical_outputs.json"
    if output.exists():
        raise FileExistsError(output)
    rows = json.loads(Path(inputs).read_text())
    models = {}
    for name in ("ranker", "editor"):
        with (run / f"{name}.pkl").open("rb") as stream:
            models[name] = pickle.load(stream)
    from .seq2seq import SentenceGenerator

    models["seq2seq"] = SentenceGenerator(run / "seq2seq", device)
    records = []
    for row in rows:
        doc = nlp()(row["source"])
        predictions = {
            name: model.predict(row["source"], doc) for name, model in models.items()
        }
        predictions["existing_rules"] = {"output": structural(row["source"])[0]}
        records.append(
            {
                **row,
                "predictions": predictions,
                "checks": {
                    name: structure_checks(row["source"], pred["output"], doc)
                    for name, pred in predictions.items()
                },
            }
        )
        print("technical", row["id"], flush=True)
    write_json(output, records)
    summary = {
        "n": len(records),
        "input_file_sha256": hashlib.sha256(Path(inputs).read_bytes()).hexdigest(),
        "label_status": "Ungraded technical transfer audit; changed counts and parser checks are not quality ratings",
        "systems": {
            name: {
                "changed": sum(
                    normalize(r["source"])
                    != normalize(r["predictions"][name]["output"])
                    for r in records
                ),
                "changed_with_failed_checks": sum(
                    normalize(r["source"])
                    != normalize(r["predictions"][name]["output"])
                    and not r["checks"][name]["pass"]
                    for r in records
                ),
            }
            for name in records[0]["predictions"]
        },
    }
    write_json(run / "technical_summary.json", summary)
    source_provenance(run, "technical-audit")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "audit"])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.cache is None:
            parser.error("--cache is required")
        prepare_inputs(args.cache, args.inputs)
    else:
        if args.run is None:
            parser.error("--run is required")
        audit(args.run, args.inputs, args.device)


if __name__ == "__main__":
    main()
