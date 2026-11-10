"""Native domain comparison of the base, disabled gate and explicit shadow run."""

import argparse
import json
import math
import os
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

from cleartext.audit import audit_case, load_cases, sha256, summarize
from cleartext.building_blocks import BuildingBlockClearText
from cleartext.data import ROOT
from cleartext.novel_pipeline import NovelCandidateClearText


def assert_retained_edits(base, other):
    """Exact text and edit identity; allow only roundoff in saved score fields."""
    assert base["output"] == other["output"]
    assert len(base["edits"]) == len(other["edits"])
    difference = 0.0
    for original, retained in zip(base["edits"], other["edits"]):
        fields = {"fit", "gain", "utility"}
        assert {key: value for key, value in original.items() if key not in fields} == {
            key: value for key, value in retained.items() if key not in fields
        }
        for field in fields:
            assert (field in original) == (field in retained)
            if field in original:
                assert math.isclose(
                    original[field], retained[field], rel_tol=0, abs_tol=1e-12
                )
                difference = max(difference, abs(original[field] - retained[field]))
    return difference


def recovered_record(probe, case, cached_base):
    """Reuse a captured unfinished case only when all strict retention checks pass."""
    assert probe["case"] == case and probe["cached_baseline"] == cached_base
    variants = probe["variants"].copy()
    fresh_base = variants["baseline"]
    assert_retained_edits(cached_base, fresh_base)
    for name in ("safe", "shadow"):
        assert_retained_edits(fresh_base, variants[name])
    variants["baseline"] = cached_base
    return {
        **case,
        "variants": variants,
        "shadow_changed_from_baseline": False,
        "safe_changed_from_baseline": False,
        "native_baseline_recheck": fresh_base,
        "human_meaning_rating": None,
        "semantic_certified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validator-run", type=Path, required=True)
    parser.add_argument("--shadow-run", type=Path, required=True)
    parser.add_argument(
        "--reuse-baseline",
        type=Path,
        help="Verified previous native audit; retain its unchanged baseline outputs.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Preserve completed records from an interrupted audit in a new output directory.",
    )
    parser.add_argument(
        "--recovered-case",
        type=Path,
        help="Saved strict native replay of the first unfinished case; requires --resume-from.",
    )
    args = parser.parse_args()
    paths = [
        ROOT / "diagnostics/domain-audit.jsonl",
        ROOT / "outputs/parliament-input-20261003/input.jsonl",
    ]
    cases = sum([load_cases(path, limit=100) for path in paths], [])
    assert len({case["id"] for case in cases}) == len(cases)
    safe_config = json.loads((args.validator_run / "config.json").read_text())
    shadow_config = json.loads((args.shadow_run / "config.json").read_text())
    assert shadow_config["shadow_only"] and shadow_config["context_threshold"] == 0.50
    base_run = ROOT / safe_config["base_run"]
    artifacts = set(base_run.glob("*.pkl")) | {base_run / "config.json"}
    base_config = json.loads((base_run / "config.json").read_text())
    for field in ("sense_model", "checker", "decision", "detector", "sentence_model"):
        if field in base_config:
            artifacts.add(ROOT / base_config[field])
    if "bert_sense" in base_config:
        artifacts.add(ROOT / base_config["bert_sense"]["vectors"])
    artifacts.update(
        ROOT / path
        for path in (
            "runs/initial-20260924/word_model.pkl",
            "runs/initial-20260924/context_model.pkl",
            "runs/context-20260924/phrase_model.pkl",
        )
    )
    cached_baseline = None
    if args.reuse_baseline:
        cached_directory = args.reuse_baseline.resolve()
        cached_summary = json.loads((cached_directory / "summary.json").read_text())
        cached_path = cached_directory / "records.jsonl"
        assert sha256(cached_path) == cached_summary["records_sha256"]
        for path in paths:
            assert (
                sha256(path)
                == cached_summary["input_sha256"][str(path.relative_to(ROOT))]
            )
        for path in artifacts:
            assert (
                sha256(path)
                == cached_summary["artifact_sha256"][str(path.relative_to(ROOT))]
            )
        cached_baseline = [
            json.loads(line) for line in cached_path.read_text().splitlines()
        ]
        assert len(cached_baseline) == len(cases)
        for case, row in zip(cases, cached_baseline):
            assert all(row[key] == value for key, value in case.items())
        paths += [cached_path, cached_directory / "summary.json"]
    for run in (args.validator_run, args.shadow_run):
        artifacts.update({run / "config.json", run / "validator.pkl"})
    sources = set((ROOT / "src/cleartext").glob("*.py")) | {Path(__file__).resolve()}
    manifest = {
        "scope": "Ungraded native domain audit. Reused diagnostics and component test text.",
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "artifact_sha256": {
            str(path.resolve().relative_to(ROOT)): sha256(path) for path in artifacts
        },
        "cases": cases,
        "selection": "All 13 existing diagnostics and first 12 existing distinct Europarl sentences.",
        "missing_real_domain_data": [
            "computer_science/arXiv",
            "corporate/SEC10-K",
            "political/GovInfo",
        ],
        "baseline_run": str(base_run.relative_to(ROOT)),
        "safe_config": safe_config,
        "shadow_config": shadow_config,
        "shadow_only": True,
        "max_word_edits": 1,
        "offline": True,
        "new_labels": False,
        "fitting": False,
        "tsar_used": False,
        "default_changed": False,
        "semantic_certified": False,
    }
    manifest["native_baseline_reused"] = cached_baseline is not None
    completed = []
    if args.resume_from:
        previous = args.resume_from.resolve()
        previous_manifest = json.loads((previous / "manifest.json").read_text())
        assert not (previous / "summary.json").exists(), (
            "audit already complete; do not rerun it"
        )
        for field in (
            "input_sha256",
            "artifact_sha256",
            "cases",
            "baseline_run",
            "safe_config",
            "shadow_config",
        ):
            assert previous_manifest[field] == manifest[field], field
        for name, digest in previous_manifest["source_sha256"].items():
            assert sha256(previous / "source" / name) == digest, name
            if name != str(Path(__file__).resolve().relative_to(ROOT)):
                assert sha256(ROOT / name) == digest, name
        completed = [
            json.loads(line)
            for line in (previous / "records.jsonl").read_text().splitlines()
        ]
        assert len(completed) < len(cases), (
            "all cases saved; reconcile rather than rerun"
        )
        for case, row in zip(cases, completed):
            assert all(row[key] == value for key, value in case.items())
        manifest["resumed_from"] = str(previous.relative_to(ROOT))
        manifest["reused_completed_cases"] = len(completed)
        for path in (previous / "manifest.json", previous / "records.jsonl"):
            manifest["input_sha256"][str(path.relative_to(ROOT))] = sha256(path)
    if args.recovered_case:
        assert args.resume_from and cached_baseline is not None
        index = len(completed)
        probe = json.loads(args.recovered_case.read_text())
        completed.append(
            recovered_record(
                probe, cases[index], cached_baseline[index]["variants"]["baseline"]
            )
        )
        manifest["recovered_case_path"] = str(args.recovered_case.resolve())
        manifest["recovered_case_sha256"] = sha256(args.recovered_case)
        manifest["reused_native_recovery_cases"] = 1
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    pipes = {
        "safe": NovelCandidateClearText.load(args.validator_run.resolve()),
        "shadow": NovelCandidateClearText.load(
            args.shadow_run.resolve(), allow_shadow=True
        ),
    }
    if cached_baseline is None:
        pipes["baseline"] = BuildingBlockClearText.load(base_run)
    started, records, maximum_roundoff = time.monotonic(), list(completed), 0.0
    with (args.output / "records.jsonl").open("x") as stream:
        for row in completed:
            base = row["variants"]["baseline"]
            for variant in ("safe", "shadow"):
                if (
                    base["edits"]
                    or base["output"] != row["text"]
                    or variant == "safe"
                    and safe_config["context_threshold"] is None
                ):
                    maximum_roundoff = max(
                        maximum_roundoff,
                        assert_retained_edits(base, row["variants"][variant]),
                    )
            stream.write(json.dumps(row, default=float) + "\n")
        stream.flush()
        for index in range(len(completed), len(cases)):
            case = cases[index]
            variants = {name: audit_case(pipe, case) for name, pipe in pipes.items()}
            if cached_baseline is not None:
                variants["baseline"] = cached_baseline[index]["variants"]["baseline"]
            base, safe, shadow = (
                variants[name] for name in ("baseline", "safe", "shadow")
            )
            # Save native values before assertions so a failed check can be
            # diagnosed and resumed without repeating completed inference.
            (args.output / f"native-case-{index:03}.json").write_text(
                json.dumps(
                    {"case": case, "cached_baseline": base, "variants": variants},
                    default=float,
                )
                + "\n"
            )
            if (
                safe_config["context_threshold"] is None
                or base["edits"]
                or base["output"] != case["text"]
            ):
                maximum_roundoff = max(
                    maximum_roundoff, assert_retained_edits(base, safe)
                )
            if base["edits"] or base["output"] != case["text"]:
                maximum_roundoff = max(
                    maximum_roundoff, assert_retained_edits(base, shadow)
                )
            record = {
                **case,
                "variants": variants,
                "shadow_changed_from_baseline": shadow["output"] != base["output"],
                "safe_changed_from_baseline": safe["output"] != base["output"],
                "human_meaning_rating": None,
                "semantic_certified": False,
            }
            stream.write(json.dumps(record, default=float) + "\n")
            stream.flush()
            records.append(record)
            print(
                json.dumps(
                    {
                        "id": case["id"],
                        "baseline": base["output"],
                        "shadow": shadow["output"],
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    summary = {
        **manifest,
        "variants": {
            name: summarize([row["variants"][name] for row in records])
            for name in ("baseline", "safe", "shadow")
        },
        "shadow_changes": [
            row["id"] for row in records if row["shadow_changed_from_baseline"]
        ],
        "safe_changes": [
            row["id"] for row in records if row["safe_changed_from_baseline"]
        ],
        "safe_output_identity_count": sum(
            not row["safe_changed_from_baseline"] for row in records
        ),
        "retained_score_maximum_roundoff": maximum_roundoff,
        "retained_baseline_changed_sentences": sum(
            row["variants"]["baseline"]["output"] != row["text"] for row in records
        ),
        "automatic_failures": {
            name: sum(
                not row["variants"][name]["automatic_checks"]["pass"]
                or not row["variants"][name]["grammar_validation"]["pass"]
                or not row["variants"][name]["document_consistency"]["pass"]
                for row in records
            )
            for name in ("baseline", "safe", "shadow")
        },
        "records_sha256": sha256(args.output / "records.jsonl"),
        "elapsed_seconds": time.monotonic() - started,
        "human_precision": None,
        "deployment_enabled": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Novel fallback domain audit",
        "",
        "Ungraded diagnostic outputs. The fixed 0.50 shadow threshold is diagnostic only.",
        "",
        "Meaning accuracy is unmeasured. Europarl text is reused CompLex component test data.",
        "",
    ]
    for row in records:
        lines += [
            f"## {row['id']}",
            "",
            row["source"]["attribution"],
            "",
            "Original: " + row["text"],
            "",
            "Base: " + row["variants"]["baseline"]["output"],
            "",
            "Safe: " + row["variants"]["safe"]["output"],
            "",
            "Shadow: " + row["variants"]["shadow"]["output"],
            "",
            "Meaning correctness remains unmeasured.",
            "",
        ]
    (args.output / "report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "shadow_changes",
                    "safe_output_identity_count",
                    "retained_baseline_changed_sentences",
                    "automatic_failures",
                    "elapsed_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
