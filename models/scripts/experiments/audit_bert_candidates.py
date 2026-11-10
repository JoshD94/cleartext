"""Measure masked-BERT proposal coverage on existing BenchLS development labels.

No acceptance policy, fitting, threshold search, generated gold or TSAR reads.
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cleartext.audit import sha256
from cleartext.bert_candidates import masked_predictions, proposals
from cleartext.data import ROOT, BENCHLS_SHA256
from cleartext.features import nlp, target_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output must be new")
    args.output.mkdir(parents=True)
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    cases = sum([pickle.load(path.open("rb"))[1] for path in paths], [])
    archive = ROOT / "data/raw/BenchLS.zip"
    assert sha256(archive) == BENCHLS_SHA256
    with zipfile.ZipFile(archive) as source:
        offsets = {
            f"benchls-{index}": int(line.split("\t")[2])
            for index, line in enumerate(
                source.read("BenchLS/BenchLS.txt").decode().splitlines()
            )
        }
    records = []
    with (args.output / "records.jsonl").open("x") as stream:
        for begin in range(0, len(cases), 16):
            batch = cases[begin : begin + 16]
            docs = list(nlp().pipe([case["text"] for case in batch]))
            tokens = []
            requests = []
            for case, doc in zip(batch, docs):
                offset = sum(
                    len(word) + 1
                    for word in case["text"].split(" ")[: offsets[case["id"]]]
                )
                token = target_token(doc, case["target"], offset)
                if token is None:
                    raise ValueError(f"unresolved target {case['id']}")
                tokens.append(token)
                requests.append((doc.text, token.idx, token.idx + len(token)))
            predictions = masked_predictions(requests, top_k=20)
            for case, token, raw in zip(batch, tokens, predictions):
                target, candidates = proposals(token, raw)
                existing = {
                    candidate["word"].lower() for candidate in case["candidates"]
                }
                added = [
                    candidate
                    for candidate in candidates
                    if candidate["word"].lower() not in existing
                ]
                gold = set(case["gold"])
                row = {
                    "id": case["id"],
                    "text": case["text"],
                    "target": case["target"],
                    "start": token.idx,
                    "target_info": target,
                    "raw_predictions": raw,
                    "proposals": added,
                    "baseline_gold_covered": bool(existing & gold),
                    "added_gold": sorted(
                        {candidate["word"].lower() for candidate in added} & gold
                    ),
                    "listed_gold": case["gold"],
                }
                records.append(row)
                stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps({"cases_completed": len(records)}), flush=True)
    baseline = sum(row["baseline_gold_covered"] for row in records)
    rescued = sum(
        not row["baseline_gold_covered"] and bool(row["added_gold"]) for row in records
    )
    added = sum(len(row["proposals"]) for row in records)
    hits = sum(len(row["added_gold"]) for row in records)
    sources = [Path(__file__).resolve()] + [
        ROOT / "src/cleartext" / name
        for name in (
            "bert_candidates.py",
            "complete_slot.py",
            "contextual.py",
            "generation.py",
        )
    ]
    for path in sources:
        destination = args.output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    result = {
        "scope": "Reused BenchLS development candidate recall only. Listed-gold matches are not human precision. "
        "Proposals are unaccepted; no fitting, TSAR, or synthetic gold.",
        "cases": len(cases),
        "top_k": 20,
        "baseline_covered_cases": baseline,
        "union_covered_cases": baseline + rescued,
        "rescued_cases": rescued,
        "added_candidate_rows": added,
        "added_listed_gold_rows": hits,
        "novel_rows_without_wordnet_senses": sum(
            not candidate["senses"] for row in records for candidate in row["proposals"]
        ),
        "default_changed": False,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in paths + [archive]
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
