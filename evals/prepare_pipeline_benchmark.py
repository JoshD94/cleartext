"""Freeze reused diagnostics and a source-hash sample of ASSET validation."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def key(text):
    return " ".join(re.findall(r"\w+|[^\w\s]", text.casefold()))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structure-data", type=Path, required=True)
    ap.add_argument("--domain-dev", type=Path, required=True)
    ap.add_argument("--authored", type=Path, required=True)
    ap.add_argument("--asset-dir", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, action="append", default=[])
    ap.add_argument("--asset-size", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite a frozen panel")
    structure = json.loads(args.structure_data.read_text())
    passages = [{"id": r["id"], "original": r["source"],
                 "references": [r["reference"]], "panel": "desse_reused_dev",
                 "domain": r["domain"]} for r in structure["broad_dev"]]
    passages += [{"id": r["id"], "original": r["source"], "references": [],
                  "panel": "domain_reused_dev", "domain": r["domain"]}
                 for r in json.loads(args.domain_dev.read_text())]
    passages += [{**r, "references": [], "panel": "authored_smoke"}
                 for r in json.loads(args.authored.read_text())]
    exclusions = {key(p["original"]) for p in passages}
    def collect(value):
        if isinstance(value, list):
            for row in value:
                collect(row)
        elif isinstance(value, dict):
            for field, item in value.items():
                if field in {"source", "reference", "target", "original"} and isinstance(item, str):
                    exclusions.add(key(item))
                elif isinstance(item, (dict, list)):
                    collect(item)
    collect(structure)
    for path in args.exclude:
        collect(json.loads(path.read_text()))
    original = args.asset_dir / "asset_valid.orig"
    refs = [args.asset_dir / f"asset_valid.simp.{i}" for i in range(10)]
    lines = original.read_text().splitlines()
    reference_lines = [p.read_text().splitlines() for p in refs]
    if any(len(r) != len(lines) for r in reference_lines):
        raise ValueError("ASSET source/reference counts differ")
    eligible = [i for i, text in enumerate(lines) if key(text) not in exclusions]
    eligible.sort(key=lambda i: hashlib.sha256(("4701:" + lines[i]).encode()).hexdigest())
    selected = eligible[:args.asset_size]
    if len(selected) != args.asset_size:
        raise ValueError("Too few ASSET validation rows after source overlap exclusion")
    passages += [{"id": f"asset-valid-{i}", "original": lines[i],
                  "references": [r[i] for r in reference_lines],
                  "panel": "asset_validation", "domain": "Wikipedia"} for i in selected]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(passages, indent=2) + "\n")
    manifest = {
        "panels": dict(Counter(p["panel"] for p in passages)),
        "input_files": {str(p): sha(p) for p in [args.structure_data, args.domain_dev, args.authored, original, *refs, *args.exclude]},
        "passages_sha256": sha(args.out), "selected_asset_ids": selected,
        "asset_selection": "First source SHA256 values with seed4701 after exact source exclusion, before prediction, no reference/grammar-quality selection",
        "asset_available": len(lines), "asset_source_overlap_excluded": len(lines) - len(eligible),
        "scope": "Development only; reused DeSSE/domain data, authored placeholders, and new ASSET validation sample. Exact source matching does not exclude near duplicates or pretraining overlap. Domain/smoke have no simplification accuracy labels.",
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["panels"]))


if __name__ == "__main__":
    main()
