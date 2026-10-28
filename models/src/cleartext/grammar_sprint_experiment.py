"""Paired rule-only correction trial on the unchanged 96-input development panel.

No fitting, exclusions, generated semantic labels or server lifecycle actions.
Run corrections first, then schedule the optional native generator comparison.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .grammar_preprocessing import GrammarCorrector, POLICY_VERSION


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def panel(rows):
    if len(rows) != 96 or len({r["id"] for r in rows}) != 96:
        raise ValueError("Keep all 96 unique development inputs")
    for row in rows:
        if (not isinstance(row["source"], str) or not row["source"].strip()
                or row["grammar"]["original"] != row["source"]):
            raise ValueError("Saved grammar must describe the unchanged source")
    return rows


def corrections(saved, output, server):
    rows = panel(json.loads(saved.read_text()))
    output.mkdir(parents=True, exist_ok=False)
    freeze = {
        "policy": POLICY_VERSION, "saved_outputs_sha256": sha(saved),
        "corrector_source_sha256": sha(Path(__file__).with_name("grammar_preprocessing.py")),
        "experiment_source_sha256": sha(__file__),
        "input_ids": [r["id"] for r in rows], "n": 96,
        "input_sha256": [hashlib.sha256(r["source"].encode()).hexdigest() for r in rows],
        "scope": "All 96 reused development inputs, in original order. One pass versus contraction-only follow-up. No grammar filter, fitting or sentence-accuracy labels.",
    }
    write(output / "freeze.json", freeze)
    clients = {"one_pass": GrammarCorrector(server, contraction_followup=False),
               "followup": GrammarCorrector(server, contraction_followup=True)}
    records = []
    with (output / "corrections.jsonl").open("x") as stream:
        for row in rows:
            record = {"id": row["id"], "source": row["source"], "conditions": {}}
            # Persist failed checks too. Never drop a troublesome benchmark row.
            for name, client in clients.items():
                record["conditions"][name] = client.correct(row["source"])
            record["archived_one_pass_reproduced"] = (
                record["conditions"]["one_pass"]["output"] == row["grammar"]["output"]
            )
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
    summary = {
        "n": len(records), "retained_n": len(records),
        "archived_one_pass_reproduced_n": sum(r["archived_one_pass_reproduced"] for r in records),
        "different_preprocessed_inputs": sum(r["conditions"]["one_pass"]["output"] != r["conditions"]["followup"]["output"] for r in records),
        "unavailable": {name: sum(not r["conditions"][name]["available"] for r in records) for name in clients},
        "corrections_sha256": sha(output / "corrections.jsonl"),
        "limits": "Correction counts are mechanism diagnostics. No general grammar or semantic accuracy estimate. Failed checks remain in the denominator.",
    }
    write(output / "correction_summary.json", summary)
    return summary


def diagnostics(saved, output, server):
    rows = json.loads(saved.read_text())
    if len(rows) != 17 or len({r["id"] for r in rows}) != 17:
        raise ValueError("Keep all 17 unique original authored diagnostics")
    if any(not isinstance(r["source"], str)
           or r["grammar"]["original"] != r["source"]
           or (r["expected_grammar"] is not None
               and not isinstance(r["expected_grammar"], str)) for r in rows):
        raise ValueError("Keep diagnostic sources and original expectations unchanged")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "freeze.json", {
        "policy": POLICY_VERSION, "saved_diagnostics_sha256": sha(saved),
        "corrector_source_sha256": sha(Path(__file__).with_name("grammar_preprocessing.py")),
        "experiment_source_sha256": sha(__file__),
        "input_ids": [r["id"] for r in rows], "n": 17,
        "scope": "All 17 original authored diagnostics, separate from the unchanged 96-input panel. Original expected strings and report-only cases retained. Mechanism checks, no semantic accuracy labels.",
    })
    clients = {"one_pass": GrammarCorrector(server, contraction_followup=False),
               "followup": GrammarCorrector(server, contraction_followup=True)}
    records = []
    with (output / "diagnostics.jsonl").open("x") as stream:
        for row in rows:
            record = {"id": row["id"], "source": row["source"],
                      "expected_grammar": row["expected_grammar"], "conditions": {}}
            for name, client in clients.items():
                record["conditions"][name] = client.correct(row["source"])
            record["archived_one_pass_reproduced"] = (
                record["conditions"]["one_pass"]["output"] == row["grammar"]["output"]
            )
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
    expected = [r for r in records if r["expected_grammar"] is not None]
    summary = {
        "n": 17, "retained_n": len(records), "mechanism_expected_n": len(expected),
        "report_only_n": len(records) - len(expected),
        "archived_one_pass_reproduced_n": sum(r["archived_one_pass_reproduced"] for r in records),
        "different_preprocessed_ids": [r["id"] for r in records
            if r["conditions"]["one_pass"]["output"] != r["conditions"]["followup"]["output"]],
        "conditions": {name: {
            "unavailable_ids": [r["id"] for r in records if not r["conditions"][name]["available"]],
            "changed_ids": [r["id"] for r in records if r["conditions"][name]["output"] != r["source"]],
            "mechanism_expected_matches": sum(r["conditions"][name]["available"]
                and r["conditions"][name]["output"] == r["expected_grammar"] for r in expected),
            "mechanism_expected_mismatch_ids": [r["id"] for r in expected
                if not r["conditions"][name]["available"]
                or r["conditions"][name]["output"] != r["expected_grammar"]],
        } for name in clients},
        "diagnostics_sha256": sha(output / "diagnostics.jsonl"),
        "limits": "Authored grammar mechanism expectations only, no semantic or general grammar accuracy claim. All 17 cases and failed checks retained. The three original report-only cases remain report-only.",
    }
    write(output / "summary.json", summary)
    return summary


def predict(trial, model, output):
    freeze = json.loads((trial / "freeze.json").read_text())
    receipt = json.loads((trial / "correction_summary.json").read_text())
    if receipt["corrections_sha256"] != sha(trial / "corrections.jsonl"):
        raise ValueError("Correction records changed after checking")
    for key, path in [("corrector_source_sha256", Path(__file__).with_name("grammar_preprocessing.py")),
                      ("experiment_source_sha256", Path(__file__))]:
        if freeze[key] != sha(path):
            raise ValueError("Correction policy code changed; use a new trial")
    rows = [json.loads(line) for line in (trial / "corrections.jsonl").read_text().splitlines()]
    if [r["id"] for r in rows] != freeze["input_ids"] or len(rows) != 96:
        raise ValueError("Incomplete or reordered correction panel")
    if [hashlib.sha256(r["source"].encode()).hexdigest() for r in rows] != freeze["input_sha256"]:
        raise ValueError("Correction source text changed")
    if any(not r["archived_one_pass_reproduced"] for r in rows):
        raise ValueError("Archived one-pass correction drifted; inspect before native comparison")
    if any(not g["available"] or g.get("reason") for r in rows for g in r["conditions"].values()):
        raise ValueError("Correction failures must be resolved before paired inference")
    output.mkdir(parents=True, exist_ok=False)
    # Fingerprint dependencies before model import; code/weights are never fitted.
    source_dir = Path(__file__).parent
    write(output / "freeze.json", {
        "corrections_sha256": sha(trial / "corrections.jsonl"),
        "trial_freeze_sha256": sha(trial / "freeze.json"),
        "artifacts": {str(p): sha(p) for p in [
            *model.rglob("*"), *source_dir.rglob("*.py")
        ] if p.is_file()},
        "n": 96, "conditions": ["original", "one_pass", "followup"],
        "scope": "Native frozen sentence generator, all 96 paired inputs. This does not rerun the ten-system full ensemble comparison or measure semantic accuracy.",
    })
    from .structure.seq2seq import SentenceGenerator

    generator = SentenceGenerator(model, "cpu")
    cache, changed = {}, {name: 0 for name in ("original", "one_pass", "followup")}
    output_differences = 0
    with (output / "outputs.jsonl").open("x") as stream:
        for row in rows:
            record = {"id": row["id"], "source": row["source"], "predictions": {}}
            inputs = {"original": row["source"], **{
                name: g["output"] for name, g in row["conditions"].items()
            }}
            for name, text in inputs.items():
                if text not in cache:
                    cache[text] = generator.predict(text)
                record["predictions"][name] = {"input": text, **cache[text]}
                changed[name] += cache[text]["output"] != row["source"]
            output_differences += record["predictions"]["one_pass"]["output"] != record["predictions"]["followup"]["output"]
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
    summary = {"n": 96, "predicted_input_n": len(cache), "changed_vs_original": changed,
               "one_pass_vs_followup_output_differences": output_differences,
               "limits": "No sentence accuracy labels; unchanged rows and failures are never silently excluded."}
    write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    correction = commands.add_parser("correct")
    correction.add_argument("--saved-outputs", type=Path, required=True)
    correction.add_argument("--output", type=Path, required=True)
    correction.add_argument("--server", default="http://127.0.0.1:8786")
    diagnostic = commands.add_parser("diagnostics")
    diagnostic.add_argument("--saved-diagnostics", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    diagnostic.add_argument("--server", default="http://127.0.0.1:8786")
    native = commands.add_parser("predict")
    native.add_argument("--trial", type=Path, required=True)
    native.add_argument("--model", type=Path, required=True)
    native.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "correct":
        result = corrections(args.saved_outputs, args.output, args.server)
    elif args.command == "diagnostics":
        result = diagnostics(args.saved_diagnostics, args.output, args.server)
    else:
        result = predict(args.trial, args.model, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
