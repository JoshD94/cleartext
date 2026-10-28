"""Replay code preservation on saved outputs, without model inference or labels."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cleartext.protected_literals import literal_spans, preserved_literals


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(rows, output_path):
    counts = Counter(inputs=len(rows))
    records = []
    for row in rows:
        source = row["source"]
        corrected = row["grammar"]["output"]
        counts["inputs_with_code"] += bool(literal_spans(source))
        for variant, original, output in [
            ("baseline", source, row["baseline"]),
            ("grammar", corrected, row["output"]),
        ]:
            check = preserved_literals(original, output)
            revised = output if check["pass"] else original
            counts[variant + "_code_rollbacks"] += not check["pass"]
            counts[variant + "_changed_outputs_before"] += output != original
            counts[variant + "_changed_outputs_after"] += revised != original
            counts[variant + "_noncode_output_changes"] += (
                not literal_spans(original) and revised != output
            )
            records.append({"id": row["id"], "variant": variant,
                            "original": original, "before": output, "after": revised,
                            "check": check})
    output_path.write_text(json.dumps(records, indent=2) + "\n")
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-panel", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [args.saved_panel, args.diagnostics, args.candidate_input]
    manifest = {"input_sha256": {str(p.resolve()): digest(p) for p in inputs},
                "source_sha256": {str(p.resolve()): digest(p) for p in [
                    Path(__file__), Path(__file__).resolve().parents[2]
                    / "src/cleartext/protected_literals.py"]}}
    summary = {"scope": "Saved-output code preservation replay, not native generation or semantic evaluation.",
               "panel": replay(json.loads(args.saved_panel.read_text()), args.output / "panel-records.json"),
               "authored_diagnostics": replay(json.loads(args.diagnostics.read_text()), args.output / "diagnostic-records.json")}
    cases = json.loads(args.candidate_input.read_text())["rows"]
    summary["candidate_pool"] = {"inputs": len(cases),
        "candidates": sum(len(case["candidates"]) for case in cases),
        "inputs_with_code": sum(bool(literal_spans(case["text"])) for case in cases)}
    for name in ("panel", "authored_diagnostics"):
        for variant in ("baseline", "grammar"):
            assert summary[name][variant + "_noncode_output_changes"] == 0
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
