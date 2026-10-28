"""Audit verb target round trips using local resources, without model imports."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cleartext.candidate_morphology import repair_target
from cleartext.coverage_lexicon import (
    VerbInventory, dictionary_verb_forms, verb_inflections, verb_lemmas,
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=ROOT / "outputs/bert-candidate-coverage-20261002/records.jsonl")
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--wordnet", type=Path, default=Path.home() / "nltk_data/corpora/wordnet.zip")
    args = parser.parse_args()
    if args.resources is None:
        paths = list((ROOT / ".venv/lib").glob("python*/site-packages/lemminflect/resources"))
        if len(paths) != 1:
            raise ValueError("supply --resources for the installed LemmInflect dictionary")
        args.resources = paths[0]
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), ROOT / "src/cleartext/candidate_morphology.py", ROOT / "src/cleartext/coverage_lexicon.py"]
    inputs = [args.records, args.wordnet] + [args.resources / name for name in (
        "lemma_lu.csv.gz", "lemma_overrides.csv", "infl_lu.csv.gz", "infl_overrides.csv")]
    manifest = {"scope": "All 929 BenchLS cases are reused development. Dictionary support and list matches do not grade full-sentence meaning.",
                "input_sha256": {str(p.resolve()): digest(p) for p in inputs},
                "source_sha256": {str(p.resolve()): digest(p) for p in sources},
                "model_imports": False, "new_labels": False, "tsar_used": False, "default_changed": False}
    write_json(args.output / "manifest.json", manifest)
    for path in sources:
        dest = args.output / "source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    rows = [json.loads(line) for line in args.records.open()]
    if len(rows) != 929 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("expected all 929 unique reused development cases")
    analyses = verb_lemmas(args.resources, [r["target"] for r in rows])
    keys = {r["target_info"][0] for r in rows if r["target_info"]}
    keys.update(base for bases in analyses.values() for base in bases)
    forms = verb_inflections(args.resources, keys)
    inventory = VerbInventory(args.wordnet)
    records, counts = [], Counter()
    for row in rows:
        result = repair_target(row["target"], row["target_info"],
            inflections=lambda lemma, tag: dictionary_verb_forms(forms, lemma, tag),
            lemmas=lambda word: analyses.get(word, ()), senses=inventory.senses)
        counts[result.reason] += 1
        records.append({"id": row["id"], "text": row["text"], "surface": row["target"],
                        "start": row["start"], **asdict(result), "changed": result.changed})
    # Coverage labels enter only after the complete label-free repair decisions.
    changed = [r for r in records if r["changed"]]
    names = {name.split("_")[0] for r in changed for name in inventory.relations(r["target"][0])}
    proposal_forms = verb_inflections(args.resources, names)
    covered = 0
    for record, row in zip(records, rows):
        if not record["changed"]:
            continue
        lemma, pos, tag, fallback = record["target"]
        evidence = []
        for name, (source, origins) in inventory.relations(lemma).items():
            if "_" in name or not name.isalpha():
                continue
            supported = dictionary_verb_forms(proposal_forms, name, tag)
            if supported:
                evidence.append({"word": supported[0], "lemma": name, "source": source,
                                 "origin_verb_offsets": origins})
        record["dictionary_supported_single_word_proposals"] = evidence
        record["listed_matches"] = sorted({r["word"].casefold() for r in evidence} & set(row["listed_gold"]))
        record["previously_covered"] = row["baseline_gold_covered"]
        covered += bool(record["listed_matches"]) and not row["baseline_gold_covered"]
    with (args.output / "records.jsonl").open("x") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    summary = {"cases": len(rows), "changed_cases": len(changed), "reason_counts": dict(counts),
               "changed_ids": [r["id"] for r in changed],
               "newly_dictionary_covered_generation_misses": covered,
               "native_generation_parity_checked": False,
               "meaning_accuracy": None, "accepted_edits": 0,
               "records_sha256": digest(args.output / "records.jsonl"),
               "manifest_sha256": digest(args.output / "manifest.json")}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
