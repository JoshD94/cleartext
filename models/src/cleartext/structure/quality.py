"""Audit cached grammar warnings and continue training on screened published pairs.

All decoded text and training rows belong in the local experiment store, not Git.
The complete development panels remain unchanged. A missing warning is not a
grammar certificate, and differences between warnings are only audit candidates.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from ..grammar_preprocessing import GrammarCorrector, POLICY_VERSION
from ..grammar_screening import SCREEN_POLICY, screen_result
from .alignment import decoded_evaluator, metrics
from .common import normalize
from .corpus import pair_keys, write_json
from .expanded import digest
from .seq2seq import capacity_check, generation_checks, train_seq2seq

TRAINING_POLICY = "training-grammar-v2"
# Review-only spelling, unfamiliar names, dialect and style are retained. These
# additional, identified rule families concern grammar or sentence orthography.
TRAINING_ERROR_RULES = {
    "IVE_CONTRACTION",
    "UPPERCASE_SENTENCE_START",
    "POSSESSIVE_APOSTROPHE",
    "EVERYDAY_EVERY_DAY",
}


def training_exclusion(grammar):
    screen = screen_result(grammar)
    if screen["status"] in {"flagged", "unavailable"}:
        return screen["status"]
    if TRAINING_ERROR_RULES.intersection(screen["review_rules"]):
        return "selected_review_rule"
    return None


def text_key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_cache(path):
    cache = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row["sha256"] != text_key(row["grammar"]["original"]):
                raise ValueError("Grammar cache text/hash mismatch")
            if row["grammar"].get("policy") != POLICY_VERSION:
                raise ValueError("Grammar cache policy mismatch")
            cache[row["sha256"]] = row["grammar"]
    return cache


def scan_texts(texts, cache_path, server):
    cache = read_cache(cache_path)
    corrector = GrammarCorrector(server)
    with Path(cache_path).open("a") as stream:
        for i, text in enumerate(dict.fromkeys(texts)):
            key = text_key(text)
            if key not in cache:
                grammar = corrector.correct(text)
                if screen_result(grammar)["status"] == "unavailable":
                    raise RuntimeError(f"Grammar check unavailable for {key}")
                stream.write(json.dumps({"sha256": key, "grammar": grammar}) + "\n")
                stream.flush()
                cache[key] = grammar
            if i % 100 == 0:
                capacity_check(Path(cache_path).parent)
                print("grammar scan", i, "cached", len(cache), flush=True)
    return cache


def warning_keys(grammar):
    """Selected error rule + matched token, independent of shifted offsets."""
    selected = set(screen_result(grammar)["error_rules"])
    return Counter(
        (s["rule"], s.get("source", "").casefold())
        for s in grammar["suggestions"]
        if s["rule"] in selected and s.get("reason") != "protected_span_or_insertion"
    )


def audit_rows(rows, cache, output_name):
    records = []
    for row in rows:
        source, output = row["source"], row[output_name]
        before, after = cache[text_key(source)], cache[text_key(output)]
        original, returned = warning_keys(before), warning_keys(after)
        records.append(
            {
                "id": row["id"],
                "changed": normalize(source) != normalize(output),
                "input_screen": screen_result(before),
                "output_screen": screen_result(after),
                "new_warning_candidates": [
                    list(key) + [count] for key, count in (returned - original).items()
                ],
                "resolved_warning_candidates": [
                    list(key) + [count] for key, count in (original - returned).items()
                ],
                "shared_warning_candidates": [
                    list(key) + [count] for key, count in (original & returned).items()
                ],
                "current_checks": generation_checks(source, output),
            }
        )
    return {
        "n": len(records),
        "changed": sum(r["changed"] for r in records),
        "input_status_counts": dict(
            Counter(r["input_screen"]["status"] for r in records)
        ),
        "output_status_counts": dict(
            Counter(r["output_screen"]["status"] for r in records)
        ),
        "new_warning_candidate_inputs": sum(
            bool(r["new_warning_candidates"]) for r in records
        ),
        "resolved_warning_candidate_inputs": sum(
            bool(r["resolved_warning_candidates"]) for r in records
        ),
        "shared_warning_candidate_inputs": sum(
            bool(r["shared_warning_candidates"]) for r in records
        ),
        "current_check_failures": dict(
            Counter(k for r in records for k in r["current_checks"]["failed"])
        ),
        "limits": "Warning changes are candidates for inspection, not measured grammar regressions or repairs. Zero warnings does not certify grammar. Every input retained.",
        "records": records,
    }


def select_training(rows, cache, evaluation_rows):
    blocked = {k for row in evaluation_rows for k in pair_keys(row)}
    retained, quarantine = [], []
    for row in rows:
        if pair_keys(row) & blocked:
            raise ValueError("Training/evaluation overlap")
        reasons = []
        for field in ("source", "reference"):
            reason = training_exclusion(cache[text_key(row[field])])
            if reason:
                reasons.append(field + ":" + reason)
        checks = generation_checks(row["source"], row["reference"])
        reasons.extend(checks["failed"])
        if reasons:
            quarantine.append({**row, "quarantine_reasons": reasons})
        else:
            retained.append(row)
    if not retained:
        raise ValueError("No screened training examples remain")
    return retained, quarantine


def scan(prior, saved_outputs, run, server):
    prior, saved_outputs, run = map(Path, (prior, saved_outputs, run))
    run.mkdir(parents=True, exist_ok=True)
    freeze_path = run / "scan_freeze.json"
    freeze = {
        "prior_data_sha256": digest(prior),
        "saved_outputs_sha256": digest(saved_outputs),
        "grammar_policy": POLICY_VERSION,
        "screen_policy": SCREEN_POLICY,
        "limits": "Already-inspected development material. No test prediction, label correction, or benchmark filtering.",
    }
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != freeze:
        raise ValueError("Scan inputs or policies changed")
    write_json(freeze_path, freeze)
    data, rows = json.loads(prior.read_text()), json.loads(saved_outputs.read_text())
    texts = [r[k] for r in rows for k in ("source", "continued")]
    # Inspect evaluation pairs first, then the existing eligible training pool.
    texts += [
        r[k]
        for split in ("dev", "broad_dev", "train")
        for r in data[split]
        for k in ("source", "reference")
    ]
    cache = scan_texts(texts, run / "grammar-cache.jsonl", server)
    audit = audit_rows(rows, cache, "continued")
    write_json(run / "baseline_audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "records"}), flush=True)


def prepare(prior, reserved, run):
    prior, reserved, run = map(Path, (prior, reserved, run))
    if (run / "data.json").exists():
        raise FileExistsError("Prepared data already exists")
    freeze = json.loads((run / "scan_freeze.json").read_text())
    if freeze["prior_data_sha256"] != digest(prior):
        raise ValueError("Prior data changed")
    data = json.loads(prior.read_text())
    cache = read_cache(run / "grammar-cache.jsonl")
    evaluation = data["dev"] + data["broad_dev"] + json.loads(reserved.read_text())
    retained, quarantine = select_training(data["train"], cache, evaluation)
    prepared = {**data, "train": retained}
    write_json(run / "data.json", prepared)
    write_json(run / "quarantined_training.json", quarantine)
    summary = {
        "training_policy": TRAINING_POLICY,
        "additional_training_error_rules": sorted(TRAINING_ERROR_RULES),
        "prior_data_sha256": digest(prior),
        "reserved_exclusion_sha256": digest(reserved),
        "grammar_cache_sha256": digest(run / "grammar-cache.jsonl"),
        "data_sha256": digest(run / "data.json"),
        "counts": {k: len(v) for k, v in prepared.items()},
        "quarantined_training": len(quarantine),
        "quarantine_reasons": dict(
            Counter(k for r in quarantine for k in r["quarantine_reasons"])
        ),
        "retained_training_datasets": dict(
            Counter(r.get("dataset", "unknown") for r in retained)
        ),
        "retained_copies": sum(
            normalize(r["source"]) == normalize(r["reference"]) for r in retained
        ),
        "retained_pairs_with_review_only_warnings": sum(
            any(
                screen_result(cache[text_key(r[k])])["status"] == "review"
                for k in ("source", "reference")
            )
            for r in retained
        ),
        "evaluation_panels_unchanged": all(
            prepared[k] == data[k] for k in ("dev", "broad_dev")
        ),
        "training": {
            "epochs": 1,
            "learning_rate": 0.0001,
            "effective_batch_size": 8,
            "seed": 4701,
        },
        "selection": "Existing equal-panel decoded structural overlap, with original reference compatibility annotations, then lower dev loss; initializer included. No evaluation input removed. One complete screened-data epoch only.",
        "limits": "Quarantined pairs retain original published labels. No fabricated copies or grammar-corrected labels. No-covered-issue training rows can still have errors. This continuation is not a causal ablation of screening versus extra training.",
    }
    write_json(run / "manifest.json", summary)
    return summary


def train(run, initial):
    run, initial = Path(run), Path(initial)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["data_sha256"] != digest(run / "data.json"):
        raise ValueError("Prepared data changed")
    data = json.loads((run / "data.json").read_text())
    interval = math.ceil(len(data["train"]) / 8)
    write_json(
        run / "training_freeze.json",
        {
            "initial_weights_sha256": digest(initial / "model.safetensors"),
            "manifest_sha256": digest(run / "manifest.json"),
            "max_steps": interval,
            "selection": manifest["selection"],
            "source_sha256": {
                p.name: digest(p) for p in Path(__file__).parent.glob("*.py")
            },
        },
    )
    return train_seq2seq(
        data,
        run,
        max_steps=interval,
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


def compare(run, server):
    run = Path(run)
    training = json.loads((run / "seq2seq_training.json").read_text())
    if training["steps_completed"] != training["max_steps"]:
        raise ValueError("Training did not finish")
    history = training["history"]
    best_index = next(
        i for i, r in enumerate(history) if r["step"] == training["selected"]["step"]
    )
    baseline = json.loads((run / "decoded-dev-0.json").read_text())
    selected = json.loads((run / f"decoded-dev-{best_index}.json").read_text())
    final = json.loads((run / f"decoded-dev-{len(history) - 1}.json").read_text())
    texts = [
        r["output"]
        for panels in (baseline, selected, final)
        for panel in panels.values()
        for r in panel
    ]
    cache = scan_texts(texts, run / "comparison-grammar-cache.jsonl", server)
    prior_cache = read_cache(run / "grammar-cache.jsonl")
    cache.update(prior_cache)
    report = {"selected_step": training["selected"]["step"], "panels": {}}
    for panel in baseline:
        ids = [r["id"] for r in baseline[panel]]
        if any([r["id"] for r in p[panel]] != ids for p in (selected, final)):
            raise ValueError("Development panels changed")
        report["panels"][panel] = {}
        for name, panels in (
            ("baseline_repaired_checks", baseline),
            ("selected", selected),
            ("final_trained", final),
        ):
            rows = panels[panel]
            audit = audit_rows(rows, cache, "output")
            write_json(run / f"{panel}-{name}-audit.json", audit)
            report["panels"][panel][name] = {
                "metrics": metrics(rows, "output"),
                "grammar": {k: v for k, v in audit.items() if k != "records"},
            }
        report["panels"][panel]["selected_outputs_differ"] = sum(
            a["output"] != b["output"]
            for a, b in zip(baseline[panel], selected[panel], strict=True)
        )
    report["limits"] = (
        "Inspected development panels used for selection, not fresh tests. Screening and additional training confounded. Parser and LanguageTool checks are incomplete. No automatic promotion or benchmark exclusions. Final trained outputs reported even if initializer wins."
    )
    write_json(run / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["scan", "prepare", "train", "compare"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--saved-outputs", type=Path)
    parser.add_argument("--reserved", type=Path)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8786")
    args = parser.parse_args()
    if args.stage == "scan":
        result = scan(args.prior, args.saved_outputs, args.run, args.server)
    elif args.stage == "prepare":
        result = prepare(args.prior, args.reserved, args.run)
    elif args.stage == "train":
        result = train(args.run, args.initial)
    else:
        result = compare(args.run, args.server)
    if result is not None:
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
