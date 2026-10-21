"""Decompose saved BenchLS development misses and candidate-pool ceilings."""

import argparse
from collections import Counter
import json
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from cleartext.audit import sha256
from cleartext.candidate_diagnostics import STAGES, diagnose_case, review_shortlist
from cleartext.data import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    baseline = ROOT / "outputs/detail-preservation-final-20261002"
    proposals_path = ROOT / "outputs/bert-candidate-coverage-20261002/records.jsonl"
    proposal_summary_path = proposals_path.parent / "summary.json"
    proposal_summary = json.loads(proposal_summary_path.read_text())
    for path in paths:
        assert (
            sha256(path)
            == proposal_summary["input_sha256"][str(path.relative_to(ROOT))]
        )
    cases = sum([pickle.load(path.open("rb"))[1] for path in paths], [])
    proposals = [json.loads(line) for line in proposals_path.read_text().splitlines()]
    assert [case["id"] for case in cases] == [row["id"] for row in proposals]
    keys = [(case["id"], row["word"]) for case in cases for row in case["candidates"]]
    assert json.loads((baseline / "feature-keys.json").read_text()) == [
        list(key) for key in keys
    ]
    assert len(keys) == len(set(keys))
    sources = [
        Path(__file__).resolve(),
        ROOT / "src/cleartext/candidate_diagnostics.py",
    ]
    inputs = paths + [
        proposals_path,
        proposal_summary_path,
        baseline / "summary.json",
        baseline / "feature-keys.json",
    ]
    inputs += [baseline / f"similarity-seed{seed}.npz" for seed in range(3)]
    inputs += [baseline / f"similarity-selections-seed{seed}.json" for seed in range(3)]
    manifest = {
        "scope": "All 929 BenchLS cases are reused development. Listed-match ceilings are not semantic accuracy.",
        "threshold": 0.45,
        "meaning_floor": 0.10,
        "fitted": False,
        "tsar_used": False,
        "default_changed": False,
        "new_labels": False,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for path in sources:
        destination = args.output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    repeats, records = [], []
    previous = json.loads((baseline / "summary.json").read_text())
    for seed in range(3):
        saved = np.load(baseline / f"similarity-seed{seed}.npz", allow_pickle=False)
        assert saved["probability"].shape == (len(keys),)
        scores = dict(zip(keys, saved["probability"]))
        folds = dict(zip(keys, saved["folds"]))
        grouped = {}
        for case in cases:
            group = case["target"].lower()
            for candidate in case["candidates"]:
                fold = int(folds[case["id"], candidate["word"]])
                assert 0 <= fold < 10 and grouped.setdefault(group, fold) == fold
        selections = json.loads(
            (baseline / f"similarity-selections-seed{seed}.json").read_text()
        )
        counts, outcomes, ceilings = Counter(), Counter(), Counter()
        with (args.output / f"records-seed{seed}.jsonl").open("x") as stream:
            for case, proposal in zip(cases, proposals):
                assert (
                    case["text"] == proposal["text"]
                    and case["target"] == proposal["target"]
                )
                assert set(case["gold"]) == set(proposal["listed_gold"])
                row = diagnose_case(
                    case,
                    {
                        candidate["word"]: scores[case["id"], candidate["word"]]
                        for candidate in case["candidates"]
                    },
                )
                expected = selections[case["id"]]["plain"]
                assert (
                    None if row["selected"] is None else row["selected"]["word"]
                ) == (None if expected is None else expected["word"])
                counts[row["stage"]] += 1
                outcomes[row["outcome"]] += 1
                for stage, count in row["listed_pool_counts"].items():
                    ceilings[stage] += bool(count)
                existing = {
                    candidate["word"].lower() for candidate in case["candidates"]
                }
                gold = set(case["gold"])
                novel = {
                    candidate["word"].lower() for candidate in proposal["proposals"]
                }
                shortlist = review_shortlist(proposal["proposals"])
                assert not existing & novel
                record = {
                    "id": case["id"],
                    "text": case["text"],
                    "target": case["target"],
                    "start": proposal["start"],
                    **row,
                    "missing_listed_substitutes": sorted(gold - existing),
                    "novel_bert_listed_matches": sorted(gold & novel),
                    "top3_bert_listed_matches": sorted(
                        gold & {c["word"].lower() for c in shortlist}
                    ),
                    "human_meaning_rating": None,
                }
                stream.write(json.dumps(record) + "\n")
                if seed == 0:
                    records.append(record)
        assert (
            counts["selected_listed"]
            == previous["repeats"][seed]["similarity"]["correct"]
        )
        assert (
            outcomes["listed_edit"] + outcomes["unlisted_edit"]
            == previous["repeats"][seed]["similarity"]["edits"]
        )
        repeats.append(
            {
                "seed": seed,
                "stage_counts": {stage: counts[stage] for stage in STAGES},
                "outcomes": dict(outcomes),
                "listed_match_ceilings": dict(ceilings),
            }
        )
    generation_misses = [row for row in records if row["stage"] == "generation"]
    rescued = sum(bool(row["novel_bert_listed_matches"]) for row in generation_misses)
    assert rescued == proposal_summary["rescued_cases"]
    result = {
        **manifest,
        "cases": len(cases),
        "candidate_rows": len(keys),
        "repeats": repeats,
        "bert_top20_rescued_generation_misses": rescued,
        "bert_top3_rescued_generation_misses": sum(
            bool(row["top3_bert_listed_matches"]) for row in generation_misses
        ),
        "bert_top20_union_listed_ceiling": proposal_summary["union_covered_cases"],
        "novel_bert_sense_linked_rows": sum(
            bool(c["senses"]) for row in proposals for c in row["proposals"]
        ),
        "records_sha256": {
            f"records-seed{seed}.jsonl": sha256(
                args.output / f"records-seed{seed}.jsonl"
            )
            for seed in range(3)
        },
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Candidate limits",
        "",
        manifest["scope"],
        "",
        "| First blocking stage | Seed 0 | Seed 1 | Seed 2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {stage} | "
        + " | ".join(str(row["stage_counts"][stage]) for row in repeats)
        + " |"
        for stage in STAGES
    ]
    lines += [
        "",
        f"WordNet contains a listed option in {repeats[0]['listed_match_ceilings']['generation']}/929 cases. "
        f"After cached guards and the meaning floor, the ceiling is {repeats[0]['listed_match_ceilings']['meaning_floor']}/929.",
        "",
        f"BERT top-20 proposals rescue {rescued} generation misses. A label-blind top-three shortlist rescues "
        f"{result['bert_top3_rescued_generation_misses']}. None of the novel proposals has the target-sense links required by the current policy.",
        "",
        "These are benchmark-list diagnostics. An unlisted edit can still be acceptable, and a listed edit can lose detail. "
        "Safety checks are ordered here to expose bottlenecks. Their counts do not justify weakening them.",
        "",
    ]
    (args.output / "report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "cases",
                    "repeats",
                    "bert_top20_rescued_generation_misses",
                    "bert_top3_rescued_generation_misses",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
