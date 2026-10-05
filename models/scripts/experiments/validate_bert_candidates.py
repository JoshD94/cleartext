"""Replay cached BERT proposals with exact dictionary checks, without inference."""

import argparse
from collections import Counter
import json
import os
import pickle
import shutil
import sys
import time
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
from cleartext.bert_candidates import proposals
from cleartext.candidate_validation import validate_proposal
from cleartext.data import ROOT
from cleartext.features import nlp, target_token
from cleartext.proposal_evidence import wordnet_evidence


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = ROOT / "outputs/proposal-review-20261003"
    prior_summary = json.loads((prior / "summary.json").read_text())
    prior_manifest = json.loads((prior / "manifest.json").read_text())
    assert sha256(prior / "records.jsonl") == prior_summary["records_sha256"]
    proposal_path = ROOT / "outputs/bert-candidate-coverage-20261002/records.jsonl"
    sense_path = ROOT / "outputs/detail-senses-final-20261002/records.jsonl"
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    for path in (proposal_path, sense_path):
        assert (
            sha256(path) == prior_manifest["input_sha256"][str(path.relative_to(ROOT))]
        )
    for name, digest in prior_manifest["artifact_sha256"].items():
        assert sha256(ROOT / name) == digest, name
    # Cached cosine and mechanical checks retain the exact prior source version.
    for name, digest in prior_manifest["source_sha256"].items():
        assert sha256(prior / "source" / name) == digest
    raw, old, senses = (
        records(proposal_path),
        records(prior / "records.jsonl"),
        records(sense_path),
    )
    cases = sum([pickle.load(path.open("rb"))[1] for path in paths], [])
    ids = [row["id"] for row in raw]
    assert (
        ids
        == [row["id"] for row in old]
        == [row["id"] for row in senses]
        == [row["id"] for row in cases]
    )
    sources = [Path(__file__).resolve()] + [
        ROOT / "src/cleartext" / name
        for name in (
            "candidate_validation.py",
            "proposal_evidence.py",
            "bert_candidates.py",
            "generation.py",
        )
    ]
    sources += [ROOT / "tests/test_candidate_validation.py"]
    inputs = paths + [
        proposal_path,
        sense_path,
        prior / "records.jsonl",
        prior / "summary.json",
        prior / "manifest.json",
    ]
    manifest = {
        "scope": "Cached BERT candidate validation on reused BenchLS development. No inference, fitting or acceptance.",
        "inventory": "Exact dictionary lemma; no automatic morphological alternatives.",
        "selection": "Validate all cached novel proposals. Recompute evidence on the earlier label-blind top-three shortlist, without backfilling.",
        "contextual_cosine_scope": "Original-slot insertion before article adjustment, reused unchanged from prior review.",
        "mechanical_checks_source": str(prior.relative_to(ROOT)),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "artifact_sha256": prior_manifest["artifact_sha256"],
        "default_changed": False,
        "tsar_used": False,
        "new_labels": False,
        "accepted_proposals": 0,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    counts, rows, review = Counter(), [], []
    started = time.monotonic()
    with (
        (args.output / "records.jsonl").open("x") as stream,
        (args.output / "review.jsonl").open("x") as review_stream,
    ):
        for index, (raw_case, old_case, sense_case, case, doc) in enumerate(
            zip(raw, old, senses, cases, nlp().pipe([row["text"] for row in raw]))
        ):
            assert (
                raw_case["text"] == case["text"]
                and raw_case["target"] == case["target"]
            )
            token = target_token(doc, raw_case["target"], raw_case["start"])
            assert token is not None and token.idx == raw_case["start"]
            target, generated = proposals(
                token, raw_case["raw_predictions"], raw_case["target_info"]
            )
            existing = {row["word"].lower() for row in case["candidates"]}
            novel = [
                candidate
                for candidate in generated
                if candidate["word"].lower() not in existing
            ]
            assert novel == raw_case["proposals"], case["id"]
            audited = [
                {**candidate, "validation": validate_proposal(candidate, target[1])}
                for candidate in novel
            ]
            kept = [
                candidate
                for candidate in audited
                if candidate["validation"]["dictionary_valid"]
            ]
            assert all(candidate["senses"] == [] for candidate in novel)
            record = {
                "id": case["id"],
                "text": case["text"],
                "target": case["target"],
                "start": token.idx,
                "proposals": audited,
            }
            stream.write(json.dumps(record) + "\n")
            rows.append(record)
            counts["novel_before"] += len(novel)
            counts["novel_after"] += len(kept)
            counts["dictionary_rejected"] += len(novel) - len(kept)
            assert (
                bool(existing & set(case["gold"])) == raw_case["baseline_gold_covered"]
            )
            if not raw_case["baseline_gold_covered"]:
                counts["generation_misses_rescued_before"] += bool(
                    {c["word"].lower() for c in novel} & set(case["gold"])
                )
                counts["generation_misses_rescued_after"] += bool(
                    {c["word"].lower() for c in kept} & set(case["gold"])
                )
            corrected = []
            for candidate in old_case["proposals"]:
                validation = validate_proposal(candidate, target[1])
                evidence = wordnet_evidence(
                    sense_case["sense_distribution"], target[1], candidate["lemma"]
                )
                row = {
                    **candidate,
                    "validation": validation,
                    "wordnet_evidence": evidence,
                }
                corrected.append(row)
                counts["review_before"] += 1
                counts["review_after"] += validation["dictionary_valid"]
                counts["review_rejected"] += not validation["dictionary_valid"]
                for relation, value in evidence["source_support_by_relation"].items():
                    counts[f"exact_any_{relation}_connection"] += value > 0
                if (
                    validation["dictionary_valid"]
                    and candidate["mechanical_checks_pass"]
                    and candidate["complexity_delta"] > 0
                ):
                    counts["review_dictionary_mechanical_lower_complexity"] += 1
            new_review = {
                "id": case["id"],
                "text": case["text"],
                "target": case["target"],
                "start": token.idx,
                "proposals": corrected,
                "accepted": False,
            }
            review_stream.write(json.dumps(new_review) + "\n")
            review.append(new_review)
            if (index + 1) % 128 == 0 or index + 1 == len(cases):
                stream.flush()
                review_stream.flush()
                print(
                    json.dumps(
                        {
                            "completed_cases": index + 1,
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    # Join list membership only after validation. It does not guide the filter.
    matched = Counter()
    for case, before, after in zip(raw, old, review):
        hits = [
            candidate
            for candidate in after["proposals"]
            if candidate["word"].lower() in case["listed_gold"]
        ]
        matched["listed_rows_before"] += len(hits)
        matched["listed_rows_after"] += sum(
            candidate["validation"]["dictionary_valid"] for candidate in hits
        )
        if not case["baseline_gold_covered"]:
            matched["top3_generation_misses_rescued_before"] += bool(hits)
            matched["top3_generation_misses_rescued_after"] += any(
                candidate["validation"]["dictionary_valid"] for candidate in hits
            )
            matched[
                "top3_generation_misses_with_valid_mechanical_lower_complexity_listed_option"
            ] += any(
                candidate["validation"]["dictionary_valid"]
                and candidate["mechanical_checks_pass"]
                and candidate["complexity_delta"] > 0
                for candidate in hits
            )
    result = {
        **manifest,
        "cases": len(cases),
        "counts": dict(counts),
        "listed_match_diagnostics": dict(matched),
        "records_sha256": sha256(args.output / "records.jsonl"),
        "review_sha256": sha256(args.output / "review.jsonl"),
        "elapsed_seconds": time.monotonic() - started,
        "meaning_accuracy": None,
        "limits": "Dictionary validation is not meaning validation. Novel proposals retain empty target-origin senses.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "counts": dict(counts),
                "listed_match_diagnostics": dict(matched),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
