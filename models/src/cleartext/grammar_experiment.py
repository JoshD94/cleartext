"""Compare optional input correction on inspected development data, without fitting."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from .grammar_preprocessing import GrammarCorrector, POLICY_VERSION
from .structure.alignment import shorter_split
from .structure.seq2seq import SentenceGenerator

# Authored mechanism checks, not published test data or training examples.
# None means report only; there is no prescribed rewrite for those inputs.
DIAGNOSTICS = [
    ("She go to school every day.", "She goes to school every day."),
    ("They is ready.", "They are ready."),
    ("He have a new car.", "He has a new car."),
    ("We was ready.", "We were ready."),
    ("She were ready yesterday.", "She was ready yesterday."),
    ("She does not knows the answer.", "She does not know the answer."),
    ("They has 25 neural networks.", "They have 25 neural networks."),
    ("This is an useful tool.", "This is a useful tool."),
    ("A hour passed.", "An hour passed."),
    ("We dont understand the result.", "We don't understand the result."),
    ("This is a useful tool.", "This is a useful tool."),
    ("The data are stored in PostgreSQL.", "The data are stored in PostgreSQL."),
    (
        "The server may not process 25 samples.",
        "The server may not process 25 samples.",
    ),
    ("The token is `dont`.", "The token is `dont`."),
    (
        "There are several schools offer evening classes.",
        None,
    ),
    ("The houses were painted and where ready for sale.", None),
    ("He dont know the result.", None),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def evaluate(saved_outputs, model_path, output, server):
    saved_outputs, model_path, output = map(Path, (saved_outputs, model_path, output))
    output.mkdir(parents=True, exist_ok=False)
    corrector = GrammarCorrector(server)
    rows = json.loads(saved_outputs.read_text())
    # Record choices before any outputs. These are already inspected dev rows.
    freeze = {
        "policy": POLICY_VERSION,
        "saved_outputs_sha256": digest(saved_outputs),
        "weights_sha256": digest(model_path / "model.safetensors"),
        "corrector_source_sha256": digest(
            Path(__file__).with_name("grammar_preprocessing.py")
        ),
        "experiment_source_sha256": digest(__file__),
        "n": len(rows),
        "diagnostics": DIAGNOSTICS,
        "scope": "Reused, inspected broad development panel plus authored diagnostics. No training or fresh test claim. All dev rows included, unchanged inputs reuse saved outputs after two exact reproduction controls.",
        "inference": "Frozen selected step906 checkpoint, greedy=False, two beams, no sampling; source and guards receive corrected input.",
    }
    write(output / "freeze.json", freeze)
    records = []
    for row in rows:
        start = time.monotonic()
        grammar = corrector.correct(row["source"])
        if not grammar["available"] or grammar.get("reason"):
            raise RuntimeError(grammar)
        records.append(
            {
                "id": row["id"],
                "source": row["source"],
                "baseline": row["continued"],
                "grammar": grammar,
                "grammar_seconds": time.monotonic() - start,
            }
        )
    diagnostics = []
    for index, (source, expected) in enumerate(DIAGNOSTICS):
        grammar = corrector.correct(source)
        if not grammar["available"] or grammar.get("reason"):
            raise RuntimeError(grammar)
        diagnostics.append(
            {
                "id": f"authored-{index}",
                "source": source,
                "expected_grammar": expected,
                "grammar": grammar,
            }
        )
    # Persist corrections before loading the generator, so failure is reviewable.
    write(output / "corrections.json", records)
    write(output / "diagnostics.json", diagnostics)
    generator = SentenceGenerator(model_path, "cpu")
    controls = [r for r in records if not r["grammar"]["changed"]][:2]
    for row in controls:
        if generator.predict(row["source"])["output"] != row["baseline"]:
            raise ValueError("Frozen baseline reproduction failed")
    for row in records:
        if row["grammar"]["changed"]:
            # Independently reproduce each affected baseline before comparing.
            if generator.predict(row["source"])["output"] != row["baseline"]:
                raise ValueError(f"Affected baseline reproduction failed: {row['id']}")
            row["prediction"] = generator.predict(row["grammar"]["output"])
            row["output"] = row["prediction"]["output"]
            row["remaining_matches"] = corrector.check(row["grammar"]["output"])[
                "matches"
            ]
            print("corrected", row["id"], flush=True)
        else:
            row["output"] = row["baseline"]
    for row in diagnostics:
        row["baseline"] = generator.predict(row["source"])["output"]
        row["prediction"] = generator.predict(row["grammar"]["output"])
        row["output"] = row["prediction"]["output"]
        row["remaining_matches"] = corrector.check(row["grammar"]["output"])["matches"]
        print("diagnostic", row["id"], flush=True)
    write(output / "outputs.json", records)
    write(output / "diagnostics.json", diagnostics)
    changed = [r for r in records if r["grammar"]["changed"]]
    expected = [r for r in diagnostics if r["expected_grammar"] is not None]
    summary = {
        "scope": freeze["scope"],
        "n": len(records),
        "software": records[0]["grammar"]["software"],
        "policy": POLICY_VERSION,
        "corrected_inputs": len(changed),
        "grammar_edits": sum(len(r["grammar"]["edits"]) for r in records),
        "applied_rules": dict(
            Counter(e["rule"] for r in records for e in r["grammar"]["edits"])
        ),
        "returned_outputs_differ": sum(r["output"] != r["baseline"] for r in records),
        "baseline_shorter_splits": sum(
            shorter_split(r["source"], r["baseline"]) for r in records
        ),
        "preprocessed_shorter_splits": sum(
            shorter_split(r["grammar"]["output"], r["output"]) for r in records
        ),
        "corrected_inputs_with_remaining_matches": sum(
            bool(r["remaining_matches"]) for r in changed
        ),
        "grammar_latency_mean_seconds": sum(r["grammar_seconds"] for r in records)
        / len(records),
        "diagnostic_n": len(diagnostics),
        "diagnostic_expected_n": len(expected),
        "diagnostic_expected_matches": sum(
            r["grammar"]["output"] == r["expected_grammar"] for r in expected
        ),
        "diagnostic_expected_mismatch_ids": [
            r["id"] for r in expected if r["grammar"]["output"] != r["expected_grammar"]
        ],
        "baseline_reproduction_controls": len(controls) + len(changed),
        "limits": "Correction counts and remaining checker matches are not human grammar or meaning ratings. Some malformed inputs have no LanguageTool matches. No claim of broad quality improvement. Original raw inputs and correction history retained. No weight or demo change.",
    }
    write(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-outputs", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8786")
    args = parser.parse_args()
    evaluate(args.saved_outputs, args.model, args.output, args.server)


if __name__ == "__main__":
    main()
