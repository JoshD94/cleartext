"""Train and compare three opt-in structural simplification approaches."""

import argparse
import gc
import json
import pickle
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..features import nlp
from ..rewrites import structural
from .common import normalize, sari, structure_checks
from .corpus import prepare, write_json
from .learning import parse_sources, train_editor, train_ranker

DIAGNOSTICS = [
    "The server checks each request and sends a response to the client.",
    "The server, which stores customer records, stopped responding.",
    "The report was reviewed by the committee.",
    "The company may reduce spending, but it will not close the factory.",
    "The neural network processed 500 records, and the analyst checked the results.",
    "Only users who have administrator access can delete the records.",
]


def summarize(rows, output_key):
    outputs = [r[output_key] for r in rows]
    changed = [normalize(r["source"]) != normalize(o) for r, o in zip(rows, outputs)]
    checks = [structure_checks(r["source"], o) for r, o in zip(rows, outputs)]
    failure_counts = Counter(k for c in checks for k in c["failed"])
    scores = [sari(r["source"], o, r["reference"]) for r, o in zip(rows, outputs)]
    lengths, source_lengths, sentence_counts = [], [], []
    for row, out in zip(rows, outputs):
        doc = nlp()(out)
        lens = [sum(t.is_alpha for t in s) for s in doc.sents]
        lengths.append(max(lens, default=0))
        source_lengths.append(sum(t.is_alpha for t in nlp()(row["source"])))
        sentence_counts.append(len(lens))
    return {
        "n": len(rows),
        "sari": float(np.mean(scores)),
        "changed": sum(changed),
        "change_rate": float(np.mean(changed)),
        "split_rate": float(np.mean(np.array(sentence_counts) > 1)),
        "max_sentence_words": float(np.mean(lengths)),
        "source_words": float(np.mean(source_lengths)),
        "check_pass_rate": float(np.mean([c["pass"] for c in checks])),
        "content_recall": float(np.mean([c["content_recall"] for c in checks])),
        "check_failures": dict(failure_counts),
        "semantic_certified": False,
    }


def paired_bootstrap(rows, key, baseline="identity", repeats=1000):
    differences = np.array(
        [
            sari(r["source"], r[key], r["reference"])
            - sari(r["source"], r[baseline], r["reference"])
            for r in rows
        ]
    )
    rng = np.random.default_rng(4701)
    values = [
        float(rng.choice(differences, len(rows), replace=True).mean())
        for _ in range(repeats)
    ]
    return {
        "mean_delta": float(differences.mean()),
        "ci95": [float(v) for v in np.quantile(values, [0.025, 0.975])],
        "resamples": repeats,
    }


def evaluate(data, run, device="auto", classic_only=False):
    run = Path(run)
    if (run / "comparison.json").exists():
        raise FileExistsError(
            "Final comparison already exists; use a fresh run for a new experiment"
        )
    models = {}
    for name in ["ranker", "editor"]:
        with (run / f"{name}.pkl").open("rb") as stream:
            models[name] = pickle.load(stream)
    if not classic_only:
        from .seq2seq import SentenceGenerator

        models["seq2seq"] = SentenceGenerator(run / "seq2seq", device)
    all_outputs, results = {}, {}
    # Freeze models and thresholds before reading any test reference scores.
    for split in ["dev", "test"]:
        rows = []
        latency = {name: [] for name in models}
        for i, row in enumerate(data[split]):
            doc = nlp()(row["source"])
            result = {
                **row,
                "identity": row["source"],
                "existing_rules": structural(row["source"])[0],
            }
            for name, model in models.items():
                start = time.perf_counter()
                prediction = model.predict(row["source"], doc)
                latency[name].append(time.perf_counter() - start)
                result[f"{name}_raw"] = prediction["raw"]
                result[name] = prediction["output"]
                result[f"{name}_details"] = prediction
            rows.append(result)
            if i % 16 == 0:
                write_json(
                    run / "progress.json",
                    {
                        "stage": "evaluate",
                        "split": split,
                        "completed": i + 1,
                        "total": len(data[split]),
                        "host": platform.node(),
                    },
                )
        keys = ["identity", "existing_rules"] + [
            key for name in models for key in [f"{name}_raw", name]
        ]
        results[split] = {key: summarize(rows, key) for key in keys}
        for name in models:
            results[split][name]["latency_median_seconds"] = float(
                np.median(latency[name])
            )
            results[split][name]["latency_p95_seconds"] = float(
                np.quantile(latency[name], 0.95)
            )
            results[split][name]["sari_delta_vs_identity"] = paired_bootstrap(
                rows, name
            )
            results[split][name]["sari_delta_vs_existing_rules"] = paired_bootstrap(
                rows, name, "existing_rules"
            )
        with (run / f"{split}_outputs.jsonl").open("w") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        all_outputs[split] = rows
    diagnostic_results = []
    for text in DIAGNOSTICS:
        doc = nlp()(text)
        diagnostic_results.append(
            {
                "source": text,
                "label_status": "Authored diagnostic, not training data or scored gold",
                **{name: model.predict(text, doc) for name, model in models.items()},
            }
        )
    write_json(run / "diagnostics.json", diagnostic_results)
    comparison = {
        "metrics": results,
        "host": platform.node(),
        "python": platform.python_version(),
        "limitations": [
            "Bounded general-domain WikiSplit pilot, not technical-domain validation",
            "Single-reference overlap scores do not establish meaning or fluency",
            "Checks are parser-dependent and conservative; raw and filtered outputs both reported",
            "No model automatically adopted into the v7 demo",
            "Pretraining contamination not ruled out",
        ],
    }
    write_json(run / "comparison.json", comparison)
    lines = [
        "# Sentence structure experiment",
        "",
        "Bounded WikiSplit pilot. Test references were not used for training, thresholds, or checkpoint selection.",
        "",
        "| System | Test SARI | Changed | Split rate | Checks passed | Max sentence words |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in results["test"].items():
        lines.append(
            f"| {name} | {values['sari']:.2f} | {values['changed']}/{values['n']} | {values['split_rate']:.1%} | {values['check_pass_rate']:.1%} | {values['max_sentence_words']:.1f} |"
        )
    lines.extend(
        [
            "",
            "SARI is our documented single-reference set-based implementation. Checks do not certify semantic equivalence.",
            "Raw rows show each model's proposals. Rows without the raw suffix include checks and development-selected abstention.",
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {s}" for s in comparison["limitations"])
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json(
        run / "progress.json",
        {
            "stage": "complete",
            "host": platform.node(),
            "comparison": str(run / "comparison.json"),
        },
    )
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "train-classic",
            "train-seq2seq",
            "evaluate",
            "all",
            "predict",
        ],
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--dev-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--scan-limit", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--classic-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--model", choices=["ranker", "editor", "seq2seq"], default="ranker"
    )
    parser.add_argument("--text")
    args = parser.parse_args()
    if any(
        n <= 0
        for n in [
            args.train_size,
            args.dev_size,
            args.test_size,
            args.scan_limit,
            args.max_steps,
            args.batch_size,
        ]
    ):
        parser.error("Sizes, scan limit, steps, and batch size must be positive")
    run = args.run.resolve()
    if args.stage == "predict":
        if not args.text:
            parser.error("--text is required for predict")
        if args.model == "seq2seq":
            from .seq2seq import SentenceGenerator

            model = SentenceGenerator(run / "seq2seq", args.device)
        else:
            with (run / f"{args.model}.pkl").open("rb") as stream:
                model = pickle.load(stream)
        print(json.dumps(model.predict(args.text), indent=2))
        return
    run.mkdir(parents=True, exist_ok=True)
    if args.stage in {"prepare", "all"}:
        if args.resume and (run / "data.json").exists():
            data = json.loads((run / "data.json").read_text())
        else:
            data, _ = prepare(
                run,
                args.cache or run.parent / "wikisplit-cache",
                args.train_size,
                args.dev_size,
                args.test_size,
                args.scan_limit,
            )
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit = "source archive, see source_commit.txt"
        if not (run / "invocation.json").exists():
            write_json(
                run / "invocation.json",
                {
                    "args": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                    "source_commit": commit,
                    "host": platform.node(),
                    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
    else:
        data = json.loads((run / "data.json").read_text())
    if args.stage in {"train-classic", "all"}:
        docs = parse_sources(data)
        for name, train in [("ranker", train_ranker), ("editor", train_editor)]:
            if args.resume and (run / f"{name}_training.json").exists():
                continue
            write_json(
                run / "progress.json",
                {"stage": f"train-{name}", "host": platform.node()},
            )
            print("training", name, flush=True)
            _, result = train(data, docs, run)
            print(name, json.dumps(result["selected"]), flush=True)
        del docs
        gc.collect()
    if (
        args.stage in {"train-seq2seq", "all"}
        and not args.classic_only
        and not (args.resume and (run / "seq2seq_training.json").exists())
    ):
        from .seq2seq import train_seq2seq

        train_seq2seq(data, run, args.max_steps, args.batch_size, args.device)
    if args.stage in {"evaluate", "all"}:
        comparison = evaluate(data, run, args.device, args.classic_only)
        print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
