"""Prepare domain inputs and human review sheets, without inventing labels."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import urllib.request
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from ..features import nlp
from .common import complete_clause, normalize
from .corpus import write_json

SPLITS = {"train", "dev", "test"}
DOMAINS = {"technical", "corporate", "political"}
REFERENCE_FIELDS = [
    "id",
    "split",
    "domain",
    "source_url",
    "context",
    "source",
    "reference",
    "author",
    "reviewer",
    "meaning_preserved",
    "grammar_ok",
    "simpler",
    "decision",
]


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.current = []
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            if self.depth:
                self.handle_endtag("p")
            self.depth = 1
        elif tag == "br" and self.depth:
            self.current.append(" ")

    def handle_endtag(self, tag):
        if tag == "p" and self.depth:
            self.paragraphs.append("".join(self.current))
            self.current, self.depth = [], 0

    def handle_data(self, data):
        if self.depth:
            self.current.append(data)


def paragraphs(path, kind):
    if kind == "pdf":
        # Raw PDF/text stay local. pdftotext preserves paragraph breaks; only
        # whitespace is normalized below. Reviewers must flag extraction errors.
        text = subprocess.check_output(
            ["pdftotext", "-nopgbrk", str(path), "-"], text=True
        )
    else:
        text = path.read_text(encoding="utf-8-sig")
    if kind == "html":
        parser = Paragraphs()
        parser.feed(text)
        parser.close()
        return parser.paragraphs
    if kind == "rfc":
        start = re.search(r"(?m)^1\. +[^\n]+$", text)
        if start is None:
            raise ValueError("RFC has no main section 1")
        text = text[start.end() :]
        text = re.sub(r"(?m)^.*(?:\[Page \d+\]|^RFC \d+).*$", "", text)
    return re.split(r"\n\s*\n", text)


def excluded_keys(paths):
    keys = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"source", "reference"} and isinstance(item, str):
                    keys.add(normalize(item))
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for path in paths:
        visit(json.loads(Path(path).read_text()))
    return keys


def prepare(config, cache, output, count=8, exclude=()):
    cache, output = Path(cache), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if count < 1:
        raise ValueError("Count must be positive")
    specs = json.loads(Path(config).read_text())
    seen_ids, seen_urls, document_hashes = set(), set(), set()
    if {(s["split"], s["domain"]) for s in specs} != {
        (split, domain) for split in SPLITS for domain in DOMAINS
    }:
        raise ValueError("Supply a separate document for every domain and split")
    seen = excluded_keys(exclude)
    rows, receipts = [], []
    cache.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        name, url, kind = spec["id"], spec["url"], spec["format"]
        if not re.fullmatch(r"[a-z0-9-]+", name) or kind not in {"html", "pdf", "rfc"}:
            raise ValueError("Invalid source id or format")
        if name in seen_ids or url in seen_urls:
            raise ValueError("Document ids and URLs must be unique across splits")
        seen_ids.add(name)
        seen_urls.add(url)
        path = cache / f"{name}.{kind}"
        if not path.exists():
            request = urllib.request.Request(
                url, headers={"User-Agent": "CleartextResearch/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read(64 * 1024**2 + 1)
            if len(payload) > 64 * 1024**2:
                raise ValueError("Document exceeds 64 MB download limit")
            if kind == "pdf" and not payload.startswith(b"%PDF"):
                raise ValueError("PDF URL returned another format")
            path.write_bytes(payload)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if spec.get("sha256") and digest != spec["sha256"]:
            raise ValueError("Source hash differs from pinned document")
        if digest in document_hashes:
            raise ValueError("Duplicate document bytes across splits")
        document_hashes.add(digest)
        selected = []
        started = "start_at" not in spec
        for paragraph in paragraphs(path, kind):
            context = " ".join(paragraph.split())
            if not started:
                match = re.search(spec["start_at"], context)
                if match is None:
                    continue
                context = context[match.start() :]
                started = True
            if not context or len(context) > 8000:
                continue
            for sentence in nlp()(context).sents:
                source = sentence.text.strip()
                key = normalize(source)
                if (
                    not 20 <= len(source.split()) <= 70
                    or not source.endswith(".")
                    or not source[:1].isupper()
                    or source.startswith(("-", "*", "•"))
                    or not complete_clause(sentence)
                    or key in seen
                ):
                    continue
                seen.add(key)
                selected.append(
                    {
                        "id": f"{name}-{len(selected)}",
                        "document_id": name,
                        "split": spec["split"],
                        "domain": spec["domain"],
                        "source_url": url,
                        "document_sha256": digest,
                        "input_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "source": source,
                        "context": context,
                        "label_status": "Unreviewed real text; no reference or human rating",
                    }
                )
                if len(selected) == count:
                    break
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(
                f"Not enough eligible sentences in {name}: {len(selected)}"
            )
        rows.extend(selected)
        receipts.append(
            {
                **spec,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "selected": len(selected),
            }
        )
    manifest = {
        "documents": receipts,
        "inputs": len(rows),
        "counts": dict(Counter(f"{r['split']}/{r['domain']}" for r in rows)),
        "selection": "First eligible sentences after any declared section anchor: 20..70 words, initial capital, terminal period, parser-complete declarative clause, no leading list marker; exact normalized overlap excluded",
        "split_policy": "Different documents in train/dev/test, assigned before predictions; test stays reserved",
        "exclude_sha256": [
            hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in exclude
        ],
        "label_status": "No human references or ratings collected",
        "config_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        "preparation_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "limitations": [
            "Small convenience sample",
            "PDF extraction and sentence parsing need human checking",
            "Exact deduplication does not remove paraphrases or pretraining overlap",
        ],
    }
    output.mkdir(parents=True)
    for split in sorted(SPLITS):
        write_json(
            output / f"{split}_inputs.json", [r for r in rows if r["split"] == split]
        )
    write_json(output / "manifest.json", manifest)
    return manifest


def write_csv(path, fields, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reference_sheet(inputs, output):
    rows = json.loads(Path(inputs).read_text())
    if any(r["split"] not in {"train", "dev"} for r in rows):
        raise ValueError("Reference preparation cannot expose reserved test inputs")
    write_csv(
        output,
        REFERENCE_FIELDS,
        [{field: row.get(field, "") for field in REFERENCE_FIELDS} for row in rows],
    )


def import_references(inputs, sheet, output):
    if Path(output).exists():
        raise FileExistsError(output)
    originals = json.loads(Path(inputs).read_text())
    by_id = {r["id"]: r for r in originals}
    if len(by_id) != len(originals):
        raise ValueError("Repeated input id")
    data, seen = {"train": [], "dev": []}, set()
    with Path(sheet).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            original = by_id.get(row["id"])
            if original is None or row["id"] in seen:
                raise ValueError("Unknown or repeated review id")
            seen.add(row["id"])
            if any(row[k] != original[k] for k in REFERENCE_FIELDS[:6]):
                raise ValueError("Review changed source text or split metadata")
            if original["split"] not in data:
                raise ValueError(
                    "Test references cannot enter training/development data"
                )
            decision = row["decision"].strip().lower()
            if decision in {"", "reject"}:
                continue
            if decision not in {"approve", "keep"}:
                raise ValueError("Decision must be approve, keep, reject, or blank")
            author, reviewer = row["author"].strip(), row["reviewer"].strip()
            reference = row["reference"].strip()
            if not author or not reviewer or author.casefold() == reviewer.casefold():
                raise ValueError(
                    "An author and a different human reviewer are required"
                )
            if not reference or any(
                row[k].strip().lower() != "yes"
                for k in ("meaning_preserved", "grammar_ok")
            ):
                raise ValueError(
                    "Reference needs text and explicit meaning/grammar approval"
                )
            identical = normalize(reference) == normalize(original["source"])
            if decision == "keep":
                if not identical or row["simpler"].strip().lower() != "no":
                    raise ValueError(
                        "Keep examples must preserve the input and mark simpler=no"
                    )
            elif identical or row["simpler"].strip().lower() != "yes":
                raise ValueError(
                    "Approved simplifications must change the input and mark simpler=yes"
                )
            data[original["split"]].append(
                {
                    **original,
                    "reference": reference,
                    "author": author,
                    "reviewer": reviewer,
                    "label_status": "Human-authored and separately approved, as declared in review sheet",
                }
            )
    if seen != set(by_id):
        raise ValueError(
            "Review sheet must retain all input ids, including blank or rejected rows"
        )
    write_json(output, data)
    return {k: len(v) for k, v in data.items()}


def blind_sheet(predictions, output):
    """Blind final development outputs; keep the system key in a separate file."""
    rows = json.loads(Path(predictions).read_text())
    sheet, key = [], []
    if Path(output).exists():
        raise FileExistsError(output)
    key_path = Path(str(output) + ".key.json")
    if key_path.exists():
        raise FileExistsError(key_path)
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Repeated prediction id")
        seen.add(row["id"])
        if row["split"] != "dev":
            raise ValueError(
                "Only development outputs can be reviewed before the final freeze"
            )
        groups = {}
        for system, prediction in row["predictions"].items():
            groups.setdefault(prediction["output"], []).append(system)
        ordered = sorted(
            groups.items(),
            key=lambda p: hashlib.sha256(
                f"4701:{row['id']}:{','.join(sorted(p[1]))}".encode()
            ).hexdigest(),
        )
        for i, (text, systems) in enumerate(ordered):
            variant = f"{row['id']}-{chr(65 + i)}"
            sheet.append(
                {
                    "id": variant,
                    "domain": row["domain"],
                    "context": row["context"],
                    "source": row["source"],
                    "output": text,
                    "simpler": "",
                    "grammar_ok": "",
                    "meaning_preserved": "",
                    "reviewer": "",
                    "notes": "",
                }
            )
            key.append(
                {"id": variant, "input_id": row["id"], "systems": sorted(systems)}
            )
    if not sheet:
        raise ValueError("No development predictions")
    write_csv(output, list(sheet[0]), sheet)
    write_json(
        key_path,
        {
            "predictions_sha256": hashlib.sha256(
                Path(predictions).read_bytes()
            ).hexdigest(),
            "variants": key,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--cache", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--count", type=int, default=8)
    prep.add_argument("--exclude", type=Path, action="append", default=[])
    for stage in ("references", "import", "blind"):
        command = sub.add_parser(stage)
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if stage == "import":
            command.add_argument("--sheet", type=Path, required=True)
    args = vars(parser.parse_args())
    stage = args.pop("stage")
    if stage == "prepare":
        print(json.dumps(prepare(**args), indent=2))
    elif stage == "references":
        reference_sheet(**args)
    elif stage == "import":
        print(json.dumps(import_references(**args)))
    else:
        blind_sheet(args["inputs"], args["output"])


if __name__ == "__main__":
    main()
