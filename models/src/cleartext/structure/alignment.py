"""Continue T5 on eligible published targets with decoded development selection."""

import argparse
import gc
import json
import math
import platform
from pathlib import Path

import numpy as np

from ..features import nlp
from .common import normalize, sari
from .corpus import pair_keys, write_json
from .expanded import (
    DEFAULT_CONFIG,
    basic_pair,
    development_group,
    digest,
    filtered_pairs,
    hash_order,
    read_parallel,
    verified_downloads,
)
from .experiment import paired_bootstrap, source_provenance
from .review import excluded_keys
from .seq2seq import SentenceGenerator, capacity_check, generation_checks, train_seq2seq

PRIOR_DATA = (
    "structural-20261026/run/data.json",
    "structure-followup-20261026/run/data.json",
    "structure-repair-20261026/run/data.json",
    "structure-expanded-20261026/run/data.json",
    "structure-search-20261027/run/data.json",
    "structure-search-20261027/run-v2/data.json",
    "structure-expanded-20261026/initial-preparation/data.json",
    "structure-expanded-20261026/desse-only-preparation/data.json",
)


def sample_unseen(rows, blocked, count):
    """Input hash ordering; no parser/reference-quality selection."""
    result, seen = [], set(blocked)
    for row in sorted(rows, key=hash_order):
        if basic_pair(row) and not pair_keys(row) & seen:
            result.append(row)
            seen.update(pair_keys(row))
        if len(result) == count:
            break
    return result


def prepare(state, run):
    state, run = Path(state), Path(run)
    if run.exists():
        raise FileExistsError("Preparation requires a fresh directory")
    paths = [state / p for p in PRIOR_DATA]
    blocked = excluded_keys(paths)
    prior = json.loads(paths[3].read_text())
    files = verified_downloads(
        json.loads(DEFAULT_CONFIG.read_text()),
        state / "structure-expanded-20261026/cache",
    )
    released = {
        split: read_parallel(files[a], files[b], split)
        for split, a, b in (
            ("train", "ACL2021/train.complex", "ACL2021/train.simple"),
            ("dev", "ACL2021/valid.complex", "ACL2021/valid.simple"),
            ("test", "ACL2021/test.complex.txt", "ACL2021/test.simple.txt"),
        )
    }
    # Every released test key is excluded from training, not only selected rows.
    test_keys = {k for r in released["test"] for k in pair_keys(r)}
    train, train_counts = filtered_pairs(prior["train"], test_keys)
    dev, dev_counts = filtered_pairs(prior["dev"], set())
    broad = sample_unseen(
        released["dev"] + [r for r in released["train"] if development_group(r)],
        blocked,
        96,
    )
    test = sample_unseen(released["test"], blocked, 96)
    if len(broad) != 96 or len(test) < 32:
        raise ValueError(
            f"Insufficient unseen examples: dev={len(broad)}, test={len(test)}"
        )
    evaluation_keys = {k for v in [*dev, *broad, *test] for k in pair_keys(v)}
    if any(pair_keys(r) & evaluation_keys for r in train):
        raise ValueError("Train/evaluation overlap")
    # Compatibility is annotated AFTER input sampling. Never resample failures.
    for row in broad:
        row["reference_compatible"] = generation_checks(
            row["source"], row["reference"]
        )["pass"]
    eligible = {r["id"] for r in train}
    quarantined = [r for r in prior["train"] if r["id"] not in eligible]
    run.mkdir(parents=True)
    data = {"train": train, "dev": dev, "broad_dev": broad}
    write_json(run / "data.json", data)
    write_json(run / "reserved_test.json", test)
    write_json(run / "quarantined_training.json", quarantined)
    # Independently assert the contract for every retained pair, including copies.
    conflicts = [
        r["id"]
        for r in train + dev
        if not generation_checks(r["source"], r["reference"])["pass"]
    ]
    if conflicts:
        raise ValueError(f"Retained target/check conflicts: {conflicts[:5]}")
    manifest = {
        "prior_data_sha256": {p: digest(state / p) for p in PRIOR_DATA},
        "source_config_sha256": digest(DEFAULT_CONFIG),
        "data_sha256": digest(run / "data.json"),
        "test_sha256": digest(run / "reserved_test.json"),
        "counts": {k: len(v) for k, v in data.items()},
        "reserved_test_n": len(test),
        "train_filters": train_counts,
        "dev_filters": dev_counts,
        "quarantined_train": len(quarantined),
        "retained_target_conflicts": len(conflicts),
        "broad_dev_reference_compatible": sum(r["reference_compatible"] for r in broad),
        "selection": "Two equal panels: aligned prior dev and compatible subset of input-sampled broad dev. Changed-reference scoring substitutes identity for non-splitting outputs. Copy-reference scoring uses actual outputs. Higher mean overlap wins; lower loss breaks ties.",
        "test_sampling": "Up to96 previously unused official test sources/full targets, seed4701 source-hash order, basic length/markup filters only; no grammar/reference-quality filter. Minimum32 required.",
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "accumulation_steps": 4,
            "learning_rate": 0.0001,
            "seed": 4701,
        },
        "human_ratings": 0,
    }
    write_json(run / "manifest.json", manifest)
    source_provenance(run, "alignment-prepare")
    return manifest


def shorter_split(source, output):
    old, new = nlp()(source), nlp()(output)
    a, b = list(old.sents), list(new.sents)
    return len(b) > len(a) and max(sum(t.is_alpha for t in s) for s in b) < max(
        sum(t.is_alpha for t in s) for s in a
    )


def structural_score(row, output):
    source, reference = row["source"], row["reference"]
    if normalize(source) != normalize(reference) and not shorter_split(source, output):
        output = source
    return sari(source, output, reference)


def metrics(rows, name):
    changes = [r for r in rows if normalize(r[name]) != normalize(r["source"])]
    copies = [r for r in rows if normalize(r["source"]) == normalize(r["reference"])]
    eligible = [r for r in rows if r["reference_compatible"]]
    return {
        "n": len(rows),
        "changed": len(changes),
        "shorter_splits": sum(shorter_split(r["source"], r[name]) for r in changes),
        "copy_reference_n": len(copies),
        "copy_references_kept": sum(
            normalize(r[name]) == normalize(r["source"]) for r in copies
        ),
        "all_reference_sari": float(
            np.mean([sari(r["source"], r[name], r["reference"]) for r in rows])
        ),
        "compatible_n": len(eligible),
        "compatible_structural_score": float(
            np.mean([structural_score(r, r[name]) for r in eligible])
        )
        if eligible
        else None,
    }


def decoded_evaluator(run, panels):
    count = 0

    def evaluate(model, tokenizer, device):
        nonlocal count
        generator = SentenceGenerator.__new__(SentenceGenerator)
        generator.model, generator.tokenizer, generator.device = (
            model,
            tokenizer,
            device,
        )
        reports, results = {}, {}
        old_cache = model.config.use_cache
        model.config.use_cache = True
        try:
            for panel, examples in panels.items():
                rows = []
                for i, row in enumerate(examples):
                    if i % 16 == 0:
                        capacity_check(run)
                    prediction = generator.predict(row["source"])
                    rows.append(
                        {
                            **row,
                            "reference_compatible": row.get(
                                "reference_compatible", True
                            ),
                            "output": prediction["output"],
                            "details": prediction,
                        }
                    )
                reports[panel] = metrics(rows, "output")
                reports[panel]["raw_changed_rejections"] = sum(
                    normalize(r["details"]["raw"]) != normalize(r["source"])
                    and not r["details"]["raw_checks"]["pass"]
                    for r in rows
                )
                results[panel] = rows
            score = float(
                np.mean([r["compatible_structural_score"] for r in reports.values()])
            )
            write_json(Path(run) / f"decoded-dev-{count}.json", results)
            count += 1
            return {"score": score, "panels": reports}
        finally:
            model.config.use_cache = old_cache

    return evaluate


def train(run, initial):
    run, initial = Path(run), Path(initial)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["data_sha256"] != digest(run / "data.json"):
        raise ValueError("Prepared data changed")
    data = json.loads((run / "data.json").read_text())
    interval = math.ceil(len(data["train"]) / 8)
    source_provenance(run, "alignment-train")
    write_json(
        run / "training_freeze.json",
        {
            "initial_sha256": digest(initial / "model.safetensors"),
            "manifest_sha256": digest(run / "manifest.json"),
            "max_steps": 2 * interval,
            "validation_interval": interval,
            "protocol": manifest["selection"],
        },
    )
    return train_seq2seq(
        data,
        run,
        max_steps=2 * interval,
        batch_size=2,
        accumulation_steps=4,
        initial_checkpoint=initial,
        target_learning_rate=0.0001,
        device="cpu",
        cpu_threads=2,
        validation_interval=interval,
        selection_evaluator=decoded_evaluator(
            run, {"aligned_dev": data["dev"], "broad_dev": data["broad_dev"]}
        ),
    )


def compare(run, baseline, output, split):
    run, output = Path(run), Path(output)
    training = json.loads((run / "seq2seq_training.json").read_text())
    if training["steps_completed"] != training["max_steps"]:
        raise ValueError("Finish training before comparison")
    manifest = json.loads((run / "manifest.json").read_text())
    if output.exists():
        raise FileExistsError("Comparison needs a fresh directory")
    if split == "test":
        if manifest["test_sha256"] != digest(run / "reserved_test.json"):
            raise ValueError("Reserved test changed")
        examples = json.loads((run / "reserved_test.json").read_text())
    else:
        if manifest["data_sha256"] != digest(run / "data.json"):
            raise ValueError("Prepared data changed")
        examples = json.loads((run / "data.json").read_text())["broad_dev"]
    output.mkdir(parents=True)
    freeze = {
        **source_provenance(run, f"alignment-{split}"),
        "baseline_sha256": digest(Path(baseline) / "model.safetensors"),
        "continued_sha256": digest(run / "seq2seq/model.safetensors"),
        "selection": training["selected"],
        "split": split,
    }
    write_json(output / "freeze.json", freeze)
    rows = [
        {
            **r,
            "identity": r["source"],
            "reference_compatible": generation_checks(r["source"], r["reference"])[
                "pass"
            ],
        }
        for r in examples
    ]
    for name, checkpoint in [("baseline", baseline), ("continued", run / "seq2seq")]:
        capacity_check(run)
        model = SentenceGenerator(checkpoint, "cpu")
        import torch

        torch.set_num_threads(2)
        for i, row in enumerate(rows):
            if i % 16 == 0:
                capacity_check(run)
                print(name, split, i, len(rows), flush=True)
            result = model.predict(row["source"])
            row[name], row[name + "_details"] = result["output"], result
        del model
        gc.collect()
    report = {
        "split": split,
        "n": len(rows),
        "host": platform.node(),
        "metrics": {
            name: metrics(rows, name) for name in ["identity", "baseline", "continued"]
        },
        "overlap_delta_bootstrap": paired_bootstrap(rows, "continued", "baseline"),
        "limitations": [
            "Single bounded seed",
            "Published decomposition references and parser compatibility are not human quality ratings",
            "Near duplicates and pretraining overlap not excluded",
            "No domain test prediction or default promotion",
        ],
    }
    write_json(output / "outputs.json", rows)
    write_json(output / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "compare"])
    parser.add_argument("--state", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.state, args.run)
    elif args.stage == "train":
        result = train(args.run, args.initial)
    else:
        result = compare(args.run, args.initial, args.output, args.split)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
