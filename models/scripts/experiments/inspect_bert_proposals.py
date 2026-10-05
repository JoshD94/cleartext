"""Inspect three cached novel BERT proposals per case; never accept or fit them."""

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

import numpy as np
from cleartext.audit import sha256
from cleartext.candidate_diagnostics import review_shortlist
from cleartext.context_similarity import replacement_similarities
from cleartext.data import ROOT
from cleartext.ensemble_pipeline import after_article, mass_noun, substitute
from cleartext.features import nlp, target_token, word_features
from cleartext.grammar_validation import GrammarValidator
from cleartext.lexical import guardrails
from cleartext.proposal_evidence import wordnet_evidence
from cleartext.technical_terms import detect_terms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proposal_path = ROOT / "outputs/bert-candidate-coverage-20261002/records.jsonl"
    sense_path = ROOT / "outputs/detail-senses-final-20261002/records.jsonl"
    word_model_path = ROOT / "runs/initial-20260924/word_model.pkl"
    proposal_summary = json.loads((proposal_path.parent / "summary.json").read_text())
    sense_summary = json.loads((sense_path.parent / "summary.json").read_text())
    assert sha256(sense_path) == sense_summary["records_sha256"]
    artifact_manifest = json.loads(
        (ROOT / "outputs/six-blocks-final-20261002/manifest.json").read_text()
    )
    assert (
        sha256(word_model_path)
        == artifact_manifest["artifact_sha256"][str(word_model_path.relative_to(ROOT))]
    )
    for name, digest in proposal_summary["input_sha256"].items():
        assert sha256(ROOT / name) == digest
    raw = [json.loads(line) for line in proposal_path.read_text().splitlines()]
    senses = [json.loads(line) for line in sense_path.read_text().splitlines()]
    assert [row["id"] for row in raw] == [row["id"] for row in senses]
    # Whitelist scoring inputs. Benchmark labels never reach feature computation.
    inputs = [
        {
            "id": row["id"],
            "text": row["text"],
            "target": row["target"],
            "start": row["start"],
            "target_info": row["target_info"],
            "proposals": review_shortlist(row["proposals"]),
            "sense_distribution": sense["sense_distribution"],
        }
        for row, sense in zip(raw, senses)
    ]
    sources = list((ROOT / "src/cleartext").glob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "tests/test_proposal_evidence.py",
    ]
    paths = [
        proposal_path,
        proposal_path.parent / "summary.json",
        sense_path,
        sense_path.parent / "summary.json",
        word_model_path,
    ]
    manifest = {
        "scope": "Unaccepted proposals on reused BenchLS development. No acceptance, fitting, new labels or TSAR.",
        "selection": "First three unique novel surfaces by cached BERT rank, independent of labels.",
        "candidate_limit": 3,
        "encoder_window": 256,
        "default_changed": False,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "artifact_sha256": artifact_manifest["artifact_sha256"],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    model = pickle.load(word_model_path.open("rb"))
    words = sorted(
        {row["target"] for row in inputs}
        | {c["word"] for row in inputs for c in row["proposals"]}
    )
    difficulty = dict(
        zip(words, model.predict([word_features(word) for word in words]))
    )
    validator, counts, checks, records = GrammarValidator(), Counter(), [], []
    started = time.monotonic()
    with (args.output / "records.jsonl").open("x") as stream:
        for index, (case, doc) in enumerate(
            zip(inputs, nlp().pipe([row["text"] for row in inputs]))
        ):
            token = target_token(doc, case["target"], case["start"])
            assert token is not None and token.idx == case["start"]
            protected = any(
                span.protected
                and span.start < token.idx + len(token)
                and token.idx < span.end
                for span in detect_terms(doc)
            )
            proposals = case["proposals"]
            similarities = replacement_similarities(
                doc.text,
                token.idx,
                token.idx + len(token),
                [c["word"] for c in proposals],
            )
            evidence = []
            for candidate, similarity in zip(proposals, similarities):
                output = substitute(doc, token, candidate["word"])
                guard = guardrails(doc.text, output, doc)
                grammar = validator.validate(doc, output)
                relation = wordnet_evidence(
                    case["sense_distribution"],
                    case["target_info"][1],
                    candidate["lemma"],
                )
                mass_block = (
                    case["target_info"][1] == "n"
                    and after_article(token)
                    and mass_noun(candidate["word"])
                )
                gain = float(difficulty[case["target"]] - difficulty[candidate["word"]])
                mechanical = (
                    guard["pass"]
                    and grammar["pass"]
                    and not protected
                    and not mass_block
                )
                origin_support = sum(
                    case["sense_distribution"].get(name, 0.0)
                    for name in candidate["senses"]
                )
                row = {
                    **candidate,
                    "output": output,
                    "complexity_delta": gain,
                    "original_complexity": float(difficulty[case["target"]]),
                    "replacement_complexity": float(difficulty[candidate["word"]]),
                    "contextual_cosine": float(similarity),
                    "guardrails": guard,
                    "grammar": grammar,
                    "protected_target": protected,
                    "mass_noun_after_article": bool(mass_block),
                    "mechanical_checks_pass": bool(mechanical),
                    "wordnet_evidence": relation,
                    "legacy_origin_support": float(origin_support),
                    "legacy_meaning_floor_pass": origin_support >= 0.10,
                    "accepted": False,
                    "human_meaning_rating": None,
                }
                evidence.append(row)
                counts["proposals"] += 1
                counts["mechanical_checks_pass"] += bool(mechanical)
                counts["lower_predicted_complexity"] += gain > 0
                counts["mechanical_and_lower_complexity"] += bool(
                    mechanical and gain > 0
                )
                counts["legacy_meaning_floor_pass"] += origin_support >= 0.10
                counts["wordnet_senses_available"] += relation[
                    "candidate_senses_available"
                ]
                for name, value in relation["source_support_by_relation"].items():
                    counts[f"any_{name}_connection"] += value > 0
            record = {
                "id": case["id"],
                "text": doc.text,
                "target": case["target"],
                "start": token.idx,
                "proposals": evidence,
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            records.append(record)
            if index in (0, len(inputs) // 2, len(inputs) - 1) and proposals:
                native = replacement_similarities(
                    doc.text,
                    token.idx,
                    token.idx + len(token),
                    [c["word"] for c in proposals],
                    batch_size=1,
                )
                error = float(np.max(np.abs(native - similarities)))
                assert error < 1e-5
                checks.append({"id": case["id"], "batch_difference": error})
            if (index + 1) % 32 == 0 or index + 1 == len(inputs):
                print(
                    json.dumps(
                        {
                            "completed_cases": index + 1,
                            "cases": len(inputs),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    # Benchmark membership is joined after scoring, for a separate report only.
    listed_counts = Counter()
    with (args.output / "listed-match-report.jsonl").open("x") as stream:
        for raw_case, case in zip(raw, records):
            listed = set(raw_case["listed_gold"])
            matches = [
                {
                    "word": c["word"],
                    "listed_match": c["word"].lower() in listed,
                    "mechanical_and_lower_complexity": c["mechanical_checks_pass"]
                    and c["complexity_delta"] > 0,
                }
                for c in case["proposals"]
            ]
            hits = [c for c in matches if c["listed_match"]]
            listed_counts["listed_rows"] += len(hits)
            listed_counts["listed_rows_mechanical_and_lower_complexity"] += sum(
                c["mechanical_and_lower_complexity"] for c in hits
            )
            if not raw_case["baseline_gold_covered"]:
                listed_counts["generation_misses_rescued_top3"] += bool(hits)
                listed_counts[
                    "generation_misses_with_mechanical_lower_complexity_listed_option"
                ] += any(c["mechanical_and_lower_complexity"] for c in hits)
            stream.write(json.dumps({"id": case["id"], "proposals": matches}) + "\n")
    result = {
        **manifest,
        "cases": len(inputs),
        "counts": dict(counts),
        "listed_match_diagnostics": dict(listed_counts),
        "native_checks": checks,
        "records_sha256": sha256(args.output / "records.jsonl"),
        "listed_report_sha256": sha256(args.output / "listed-match-report.jsonl"),
        "elapsed_seconds": time.monotonic() - started,
        "accepted_proposals": 0,
        "meaning_accuracy": None,
        "limits": "Checks, cosine, dictionary connections and complexity predictions do not establish retained meaning. "
        "The current origin-based meaning floor rejects novel proposals. No threshold was changed.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "cases",
                    "counts",
                    "listed_match_diagnostics",
                    "elapsed_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
