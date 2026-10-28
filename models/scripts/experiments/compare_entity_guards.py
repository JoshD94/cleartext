"""Compare old and aligned entity guards on every frozen development candidate."""

import argparse
from collections import Counter
from functools import lru_cache
import importlib.util
import importlib.metadata
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cleartext.audit import sha256
from cleartext.data import ROOT


def export_cases():
    paths = [ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")]
    senses = ROOT / "outputs/detail-senses-final-20261002/records.jsonl"
    starts = {r["id"]: r["start"] for r in map(json.loads, senses.read_text().splitlines())}
    cases = []
    for path in paths:
        with path.open("rb") as stream:
            cases.extend(pickle.load(stream)[1])
    # Labels are not needed by the native guard scan.
    rows = [{"id": c["id"], "text": c["text"], "target": c["target"], "start": starts[c["id"]],
             "candidates": [{k: r[k] for k in ("word", "guard", "guard_failed")} for r in c["candidates"]]}
            for c in cases]
    return {"input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths + [senses]}, "rows": rows}


def compare(args, payload):
    from cleartext import lexical
    from cleartext.features import nlp, target_token
    from cleartext.ensemble_pipeline import substitute

    spec = importlib.util.spec_from_file_location("cleartext.lexical_legacy", args.legacy_source)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    parser = lru_cache(2)(nlp())
    # Both implementations receive the identical original and candidate parses.
    # The two-entry cache avoids parsing a candidate twice, without changing it.
    lexical.nlp = legacy.nlp = lambda: parser
    manifest = {"input_sha256": payload["input_sha256"], "export_sha256": sha256(args.input),
                "source_sha256": {"scanner": sha256(Path(__file__)), "aligned": sha256(Path(lexical.__file__)),
                                  "legacy": sha256(args.legacy_source)},
                "packages": {n: importlib.metadata.version(n) for n in ("spacy", "en-core-web-sm")},
                "parse_cache_entries": 2}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    counts = Counter()
    with (args.output / "records.jsonl").open("x") as stream:
        for index, case in enumerate(payload["rows"]):
            doc = parser(case["text"])
            token = target_token(doc, case["target"], case["start"])
            if case["candidates"] and token is None:
                raise ValueError(f"unaligned target {case['id']}")
            rows = []
            for candidate in case["candidates"]:
                text = substitute(doc, token, candidate["word"])
                old = legacy.guardrails(doc.text, text, doc)
                new = lexical.guardrails(doc.text, text, doc)
                assert {k: v for k, v in old["checks"].items() if k != "entity_roles"} == {
                    k: v for k, v in new["checks"].items() if k != "entity_roles"}
                rows.append({"word": candidate["word"], "cached": candidate["guard"],
                             "legacy": old["pass"], "aligned": new["pass"],
                             "legacy_failed": old["failed"], "aligned_failed": new["failed"]})
                counts["candidates"] += 1
                counts["cached_legacy_disagreements"] += int(candidate["guard"] != old["pass"])
                counts["newly_eligible"] += int(not old["pass"] and new["pass"])
                counts["newly_blocked"] += int(old["pass"] and not new["pass"])
            stream.write(json.dumps({"id": case["id"], "rows": rows}) + "\n")
            stream.flush()
            counts["cases"] += 1
            if index % 100 == 0:
                print(json.dumps(dict(counts)), flush=True)
    summary = {"counts": dict(counts), "nonentity_checks_identical": True,
               "semantic_accuracy_measured": False, "records_sha256": sha256(args.output / "records.jsonl")}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--legacy-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.export_only:
        with args.output.open("x") as stream:
            stream.write(json.dumps(export_cases()) + "\n")
        return
    if args.input is None or args.legacy_source is None:
        parser.error("native comparison requires --input and --legacy-source")
    args.output.mkdir(parents=True, exist_ok=False)
    compare(args, json.loads(args.input.read_text()))


if __name__ == "__main__":
    main()
