"""Audit attributed JSONL sentences with frozen models, offline and without fitting."""

import argparse
import importlib.metadata
import json
import os
import platform
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.audit import audit_case, load_cases, render_report, sha256, summarize
from cleartext.data import ROOT
from cleartext.ensemble_pipeline import EnsembleClearText, LATEST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "diagnostics/domain-audit.jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing outputs are never replaced",
    )
    parser.add_argument("--run", type=Path, default=LATEST)
    parser.add_argument(
        "--limit", type=int, default=30, help="First N records, maximum 100"
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Also run identity and the existing dictionary baseline",
    )
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument(
        "--preserve-detail",
        action="store_true",
        help="Opt-in experiment: reject direct WordNet hypernyms after scoring",
    )
    variant.add_argument(
        "--building-blocks",
        action="store_true",
        help="Run the optional blocks enabled in the run config",
    )
    args = parser.parse_args()
    cases = load_cases(args.input, args.limit)
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    artifacts = set(run.glob("*.pkl")) | {run / "config.json"}
    for field in ("sense_model", "checker", "decision", "detector", "sentence_model"):
        if field in config:
            artifacts.add(ROOT / config[field])
    if "bert_sense" in config:
        artifacts.add(ROOT / config["bert_sense"]["vectors"])
    artifacts.update(
        ROOT / path
        for path in (
            "runs/initial-20260924/word_model.pkl",
            "runs/initial-20260924/context_model.pkl",
            "runs/context-20260924/phrase_model.pkl",
            "data/cache/brown-bigrams.pkl",
        )
    )
    if args.baselines:
        artifacts.update((ROOT / "runs/initial-20260924").glob("*.pkl"))
        artifacts.add(ROOT / "runs/initial-20260924/fit_selection.json")
    source_files = set((ROOT / "src/cleartext").glob("*.py")) | {
        Path(__file__).resolve()
    }
    bert_cache = {}
    if config.get("bert_fit") or "bert_sense" in config:
        from huggingface_hub import try_to_load_from_cache
        from cleartext.contextual import NAME

        for filename in (
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
        ):
            path = try_to_load_from_cache(NAME, filename)
            if isinstance(path, str):
                bert_cache[filename] = {"path": path, "sha256": sha256(path)}
    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "host": platform.node(),
        "argv": sys.argv,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "selected_cases": cases,
        "selection": "first N records in input order",
        "model_run": str(run),
        "config": config,
        "artifact_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in sorted(artifacts)
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in sorted(source_files)
        },
        "bert_local_cache": bert_cache,
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "scikit-learn", "spacy", "nltk", "torch", "transformers")
        },
        "offline": True,
        "fitting": False,
        "human_labels": False,
        "experimental_detail_policy": args.preserve_detail,
        "max_word_edits": 1,
    }
    manifest["building_blocks_enabled"] = args.building_blocks
    for path in source_files:
        destination = args.output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    manifest["source_snapshot"] = "source/"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    if args.preserve_detail:
        from cleartext.preservation import ConservativeDetailClearText

        pipe = ConservativeDetailClearText.load(run)
    elif args.building_blocks:
        from cleartext.building_blocks import BuildingBlockClearText

        pipe = BuildingBlockClearText.load(run)
    else:
        pipe = EnsembleClearText.load(run)
    manifest["pipeline_load_seconds"] = time.perf_counter() - started
    dictionary = None
    if args.baselines:
        from cleartext.pipeline import ClearText

        dictionary = ClearText()
    records = []
    with (args.output / "records.jsonl").open("x") as stream:
        for case in cases:
            record = audit_case(pipe, case)
            if dictionary is not None:
                record["baselines"] = {
                    "identity": {"output": case["text"]},
                    "dictionary": dictionary.analyze(case["text"], mode="dictionary"),
                }
                if args.building_blocks:
                    from cleartext.edit_quality import EditQualityEvaluator

                    for baseline in record["baselines"].values():
                        baseline["edit_quality"] = EditQualityEvaluator().evaluate(
                            case["text"], baseline["output"], pipe
                        )
            stream.write(json.dumps(record, default=float) + "\n")
            stream.flush()
            records.append(record)
            print(f"{case['id']}: {record['output']}", flush=True)
    summary = summarize(records)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "report.md").write_text(render_report(records, summary))
    manifest["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary["overall"]), flush=True)


if __name__ == "__main__":
    main()
