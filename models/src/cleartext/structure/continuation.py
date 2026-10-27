"""Compare a completed continuation with its initializer using identical checks."""

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np

from ..features import nlp
from ..rewrites import structural
from .common import normalize
from .corpus import write_json
from .expanded import digest
from .experiment import paired_bootstrap, source_provenance, summarize
from .seq2seq import SentenceGenerator, capacity_check


def compare(run, baseline, split, device="auto", output=None):
    run, baseline = Path(run), Path(baseline)
    output = Path(output) if output else run / f"{split}-evaluation"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output must be a fresh directory")
    if split not in {"dev", "test"}:
        raise ValueError("Only explicit dev/test evaluation is allowed")
    training = json.loads((run / "seq2seq_training.json").read_text())
    if training["steps_completed"] != training["max_steps"]:
        raise ValueError("Finish the bounded training run before comparison")
    data = json.loads((run / "data.json").read_text())
    rows = data[split]
    if not rows:
        raise ValueError("No evaluation rows")
    output.mkdir(parents=True, exist_ok=True)
    freeze = {
        **source_provenance(run, f"continuation-{split}"),
        "split": split,
        "baseline_sha256": digest(baseline / "model.safetensors"),
        "continued_sha256": digest(run / "seq2seq/model.safetensors"),
        "training_record_sha256": digest(run / "seq2seq_training.json"),
        "selection": training["selected"],
        "settings": "Identical deterministic two-beam generation and current checks for both models",
    }
    write_json(output / "freeze.json", freeze)
    results = [
        {**r, "identity": r["source"], "rules": structural(r["source"])[0]}
        for r in rows
    ]
    latency = {}
    for name, path in [("baseline", baseline), ("continued", run / "seq2seq")]:
        capacity_check(output)
        model = SentenceGenerator(path, device)
        latency[name] = []
        for i, row in enumerate(results):
            if i % 8 == 0:
                capacity_check(output)
            started = time.perf_counter()
            prediction = model.predict(row["source"], nlp()(row["source"]))
            latency[name].append(time.perf_counter() - started)
            row[name] = prediction["output"]
            row[f"{name}_raw"] = prediction["raw"]
            row[f"{name}_details"] = prediction
            # Isolate the added T5 checks using the same raw text.
            legacy_pass = all(
                v
                for k, v in prediction["raw_checks"]["checks"].items()
                if k not in {"predicate_tense", "complement_scope"}
            )
            row[f"{name}_previous_checks"] = (
                prediction["raw"] if legacy_pass else row["source"]
            )
            if i % 8 == 0 or i + 1 == len(results):
                write_json(
                    output / "progress.json",
                    {
                        "system": name,
                        "completed": i + 1,
                        "total": len(results),
                        "split": split,
                    },
                )
                print(name, split, i + 1, len(results), flush=True)
        del model
        gc.collect()
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    metrics = {}
    for name in (
        "identity",
        "rules",
        "baseline_raw",
        "baseline_previous_checks",
        "baseline",
        "continued_raw",
        "continued_previous_checks",
        "continued",
    ):
        metrics[name] = summarize(results, name)
        copies = [
            r for r in results if normalize(r["source"]) == normalize(r["reference"])
        ]
        splits = [
            r for r in results if normalize(r["source"]) != normalize(r["reference"])
        ]
        metrics[name]["annotated_copy_n"] = len(copies)
        metrics[name]["annotated_copy_accuracy"] = (
            sum(normalize(r[name]) == normalize(r["source"]) for r in copies)
            / len(copies)
            if copies
            else None
        )
        metrics[name]["changed_reference_n"] = len(splits)
        metrics[name]["changed_reference_sari"] = (
            summarize(splits, name)["sari"] if splits else None
        )
    for name in ("baseline", "continued"):
        metrics[name].update(
            latency_median_seconds=float(np.median(latency[name])),
            latency_p95_seconds=float(np.quantile(latency[name], 0.95)),
            raw_model_checks_passed=sum(
                r[f"{name}_details"]["raw_checks"]["pass"] for r in results
            ),
            tense_rejections=sum(
                "predicate_tense" in r[f"{name}_details"]["raw_checks"]["failed"]
                for r in results
            ),
        )
    comparison = {
        "split": split,
        "n": len(results),
        "host": platform.node(),
        "metrics": metrics,
        "continued_vs_baseline": paired_bootstrap(results, "continued", "baseline"),
        "continued_vs_baseline_previous_checks": paired_bootstrap(
            results, "continued", "baseline_previous_checks"
        ),
        "label_status": "Published crowdsourced references, not project human model ratings",
        "limitations": [
            "One bounded seed and selected sample",
            "Set-based single-reference SARI measures overlap, not correctness",
            "Published references may delete facts or relations",
            "Near duplicates, document overlap and pretraining overlap not excluded",
            "No independent human judgments or project domain quality claim",
        ],
    }
    with (output / "outputs.jsonl").open("x") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output / "comparison.json", comparison)
    write_json(output / "progress.json", {"stage": "complete", "split": split})
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    result = compare(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "split": result["split"],
                "metrics": {n: r["sari"] for n, r in result["metrics"].items()},
                "continued_vs_baseline": result["continued_vs_baseline"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
