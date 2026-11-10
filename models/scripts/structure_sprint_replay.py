#!/usr/bin/env python3
"""Freeze, then replay identical saved development beams. No generator loads."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def code_hashes():
    paths = [Path(__file__), *sorted((ROOT / "src/cleartext/structure").glob("*.py")),
             *sorted((ROOT / "src/cleartext").glob("*.py"))]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def parser_environment():
    """Fingerprint installed parser metadata without importing/loading spaCy."""
    versions = {}
    for package in ("spacy", "en-core-web-sm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def freeze(inputs, output):
    """Pure Python preparation, allowed while native model jobs are blocked."""
    if not inputs:
        raise ValueError("At least one saved development JSONL is required")
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for index, path in enumerate(inputs):
        snapshot = output / f"input-{index}.jsonl"
        snapshot.write_bytes(path.read_bytes())
        # Validate the beam provenance without scoring or inspecting references.
        rows = [json.loads(line) for line in snapshot.read_text().splitlines()]
        if not rows or any(not r.get("search_details", {}).get("model_candidates") for r in rows):
            raise ValueError(f"Missing raw model_candidates: {path}")
        if any("terminated" not in c or "text" not in c
               for r in rows for c in r["search_details"]["model_candidates"]):
            raise ValueError(f"Missing termination/text provenance: {path}")
        files.append({"original": str(path.resolve()), "snapshot": snapshot.name,
                      "sha256": sha(snapshot), "rows": len(rows)})
    manifest = {
        "purpose": "Reused development only; no fitting, inference, reserved domain test or semantic accuracy claims",
        "baseline": "model_order with current unchanged generation/boundary checks",
        "candidate": "source_anchored_split with identical raw beams and termination flags",
        "ranking": "smallest longest sentence, then beam rank; identity if none qualifies",
        "weights": "No weights loaded; raw outputs retain their historical checkpoint provenance",
        "inputs": files, "code_sha256": code_hashes(),
        "parser_package_versions": parser_environment(),
        "report": "All differences, overlap gains/losses, copy-reference edits, coverage, per-check failures",
    }
    write(output / "manifest.json", manifest)
    return manifest


def replay(run):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["code_sha256"] != code_hashes():
        raise ValueError("Source changed after freeze; use a fresh run")
    if manifest["parser_package_versions"] != parser_environment():
        raise ValueError("Parser package versions changed after freeze")
    for record in manifest["inputs"]:
        if sha(run / record["snapshot"]) != record["sha256"]:
            raise ValueError("Frozen raw candidates changed")
    output = run / "replay"
    output.mkdir(exist_ok=False)
    # This stage loads the native spaCy parser only. Parent schedules it when
    # resources fit. Never imports or instantiates a transformer generator.
    from cleartext.features import nlp
    from cleartext.structure.common import normalize, sari
    from cleartext.structure.search import select_candidate

    parser = nlp()
    summary = {}
    with (output / "records.jsonl").open("w") as stream:
        for record in manifest["inputs"]:
            counts, failures, deltas = Counter(), Counter(), []
            rows = [json.loads(line) for line in (run / record["snapshot"]).read_text().splitlines()]
            for row in rows:
                source, proposals = row["source"], row["search_details"]["model_candidates"]
                old = parser(source)
                baseline = select_candidate(source, proposals, old)
                candidate = select_candidate(source, proposals, old, policy="source_anchored_split")
                counts["inputs"] += 1
                reference = row.get("reference")
                copy_reference = reference is not None and normalize(source) == normalize(reference)
                counts["copy_references"] += copy_reference
                for name, prediction in [("baseline", baseline), ("candidate", candidate)]:
                    changed = normalize(prediction["output"]) != normalize(source)
                    counts[f"{name}_changes"] += changed
                    counts[f"{name}_copy_reference_changes"] += copy_reference and changed
                    lengths = [sum(t.is_alpha for t in s) for s in parser(prediction["output"]).sents]
                    old_lengths = [sum(t.is_alpha for t in s) for s in old.sents]
                    counts[f"{name}_shorter_splits"] += bool(lengths and old_lengths) and len(lengths) > len(old_lengths) and max(lengths) < max(old_lengths)
                differs = baseline["output"] != candidate["output"]
                counts["differences"] += differs
                counts["recovered_from_baseline_copy"] += (
                    differs and normalize(baseline["output"]) == normalize(source)
                    and normalize(candidate["output"]) != normalize(source))
                counts["baseline_edits_lost"] += (
                    differs and normalize(baseline["output"]) != normalize(source)
                    and normalize(candidate["output"]) == normalize(source))
                scores = {}
                if reference is not None:
                    scores = {name: sari(source, prediction["output"], reference)
                              for name, prediction in [("baseline", baseline), ("candidate", candidate)]}
                    delta = scores["candidate"] - scores["baseline"]
                    deltas.append(delta)
                    counts["overlap_gains"] += delta > 1e-9
                    counts["overlap_losses"] += delta < -1e-9
                for proposal in candidate.get("split_candidates", candidate["candidates"]):
                    failures.update(proposal["checks"]["failed"])
                stream.write(json.dumps({"panel": record["snapshot"], "id": row["id"],
                                         "source": source, "reference": reference,
                                         "same_raw_candidates": proposals,
                                         "baseline": baseline, "candidate": candidate,
                                         "different": differs, "overlap_proxy": scores}) + "\n")
                stream.flush()
            summary[record["snapshot"]] = {
                "counts": dict(counts), "candidate_check_failures": dict(failures),
                "reference_overlap_mean_delta": sum(deltas) / len(deltas) if deltas else None,
                "reference_overlap_n": len(deltas),
            }
    if manifest["code_sha256"] != code_hashes():
        raise ValueError("Source changed during replay; partial records retained")
    write(output / "summary.json", {"panels": summary,
          "caution": "Development coverage and mechanical/reference diagnostics, not semantic correctness. Every difference is retained in records.jsonl."})
    return summary


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=["freeze", "replay"])
    cli.add_argument("--input", type=Path, action="append", default=[])
    cli.add_argument("--run", type=Path, required=True)
    args = cli.parse_args()
    result = freeze(args.input, args.run) if args.stage == "freeze" else replay(args.run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
