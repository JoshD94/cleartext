"""Trace frozen listed-candidate losses without assigning semantic correctness."""

import argparse
from collections import Counter
import importlib.metadata
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cleartext.audit import sha256
from cleartext.data import ROOT


def export_losses():
    paths = [ROOT / f"data/cache/refine-v7-benchls-{part}.pkl"
             for part in ("dev", "holdout")]
    sense_path = ROOT / "outputs/detail-senses-final-20261002/records.jsonl"
    senses = {r["id"]: r for r in map(json.loads, sense_path.read_text().splitlines())}
    cases = []
    for path in paths:
        with path.open("rb") as stream:
            cases.extend(pickle.load(stream)[1])
    rows = []
    for case in cases:
        listed = [r for r in case["candidates"] if r["gold"]]
        guarded = [r for r in listed if r["guard"]]
        kind = "guard" if listed and not guarded else (
            "meaning" if guarded and not any(r["x"][0] >= .10 for r in guarded) else None)
        if kind is None:
            continue
        rows.append({
            "id": case["id"], "kind": kind, "text": case["text"],
            "target": case["target"], "target_info": case["target_info"],
            "start": senses[case["id"]]["start"],
            "sense_distribution": senses[case["id"]]["sense_distribution"],
            "candidates": [{k: r[k] for k in ("word", "lemma", "source", "guard", "guard_failed")}
                           | {"sense_fit": float(r["x"][0])} for r in listed],
        })
    return {"scope": "All reused BenchLS development, listed candidates only; no semantic labels assigned.",
            "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths + [sense_path]},
            "rows": rows}


def native_audit(payload, output):
    from cleartext.features import nlp, target_token
    from cleartext.ensemble_pipeline import substitute
    from cleartext import lexical
    from cleartext.generation import relation_map

    def roles(doc):
        return [{"i": t.i, "text": t.text, "dep": t.dep_, "head": t.head.i,
                 "head_text": t.head.text} for t in doc]

    def entities(doc):
        return [{"text": e.text, "label": e.label_, "dep": e.root.dep_,
                 "head": e.root.head.i} for e in doc.ents]

    manifest = {
        "source_sha256": {"audit": sha256(Path(__file__)), "lexical": sha256(Path(lexical.__file__))},
        "input_sha256": payload["input_sha256"],
        "packages": {n: importlib.metadata.version(n) for n in ("spacy", "en-core-web-sm", "nltk")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    counts = Counter()
    max_sense_delta = 0.0
    with (output / "records.jsonl").open("x") as stream:
        for case in payload["rows"]:
            record = dict(case)
            if case["kind"] == "guard":
                doc = nlp()(case["text"])
                token = target_token(doc, case["target"], case["start"])
                if token is None:
                    raise ValueError(f"unaligned target {case['id']}")
                record.update(original_roles=roles(doc), original_entities=entities(doc))
                record["native"] = []
                for candidate in case["candidates"]:
                    text = substitute(doc, token, candidate["word"])
                    parsed = nlp()(text)
                    checks = lexical.guardrails(doc.text, text, doc)
                    record["native"].append({"word": candidate["word"], "output": text,
                                             "checks": checks, "roles": roles(parsed),
                                             "entities": entities(parsed)})
                    counts.update("native_guard_candidate_" + r for r in checks["failed"])
                counts["native_guard_cases_with_passing_listed_candidate"] += int(
                    any(r["checks"]["pass"] for r in record["native"]))
            else:
                lemma, pos = case["target_info"][:2]
                relations = {name.casefold(): senses for name, (_, senses) in relation_map(lemma, pos).items()}
                record["sense_trace"] = []
                for candidate in case["candidates"]:
                    support = relations[candidate["lemma"].replace(" ", "_").casefold()]
                    score = sum(case["sense_distribution"].get(n, 0) for n in support)
                    delta = abs(score - candidate["sense_fit"])
                    max_sense_delta = max(max_sense_delta, delta)
                    record["sense_trace"].append({"word": candidate["word"], "support": support,
                                                   "recomputed": score, "absolute_delta": delta})
            counts[case["kind"] + "_cases"] += 1
            stream.write(json.dumps(record) + "\n")
            stream.flush()
    summary = {"counts": dict(counts), "maximum_sense_fit_disagreement": max_sense_delta,
               "semantic_accuracy_measured": False,
               "records_sha256": sha256(output / "records.jsonl")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text()) if args.input else export_losses()
    if args.export_only:
        with args.output.open("x") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    native_audit(payload, args.output)


if __name__ == "__main__":
    main()
