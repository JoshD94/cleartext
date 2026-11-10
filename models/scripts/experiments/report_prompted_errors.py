"""Describe paired development errors using existing substitute-list labels."""

import argparse
from collections import Counter
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cleartext.audit import sha256
from cleartext.data import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads((args.experiment / "summary.json").read_text())
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    for path in paths:
        assert sha256(path) == summary["input_sha256"][str(path.relative_to(ROOT))]
    cases = sum([pickle.load(path.open("rb"))[1] for path in paths], [])
    selections = {}
    for name in ("baseline", "prompted"):
        selections[name] = []
        for row in summary["repeats"]:
            path = args.experiment / f"{name}-selections-seed{row['seed']}.json"
            selections[name].append(json.loads(path.read_text()))
            paths.append(path)
    transitions = Counter()
    errors = []

    def state(candidate):
        return (
            "abstain"
            if candidate is None
            else "listed"
            if candidate["gold"]
            else "unlisted"
        )

    for case in cases:
        choices = []
        for repeat, row in enumerate(summary["repeats"]):
            before = selections["baseline"][repeat][case["id"]]
            after = selections["prompted"][repeat][case["id"]]
            transitions[f"{state(before)} -> {state(after)}"] += 1
            if after and not after["gold"]:
                candidate = next(
                    c for c in case["candidates"] if c["word"] == after["word"]
                )
                choices.append(
                    {
                        "seed": row["seed"],
                        "before": before,
                        "after": after,
                        "source": candidate["source"],
                        "sense_fit": float(candidate["x"][0]),
                    }
                )
        if choices:
            errors.append(
                {
                    "id": case["id"],
                    "target": case["target"],
                    "text": case["text"],
                    "unlisted_in_repeats": len(choices),
                    "choices": choices,
                    "listed_candidates": [
                        c["word"] for c in case["candidates"] if c["gold"]
                    ],
                }
            )
    errors.sort(key=lambda row: (-row["unlisted_in_repeats"], row["id"]))
    result = {
        "scope": "Reused BenchLS candidate labels. Unlisted choices are not new semantic error labels. Repeats are dependent.",
        "transitions": dict(transitions),
        "cases": errors,
        "persistent_unlisted_cases": sum(
            row["unlisted_in_repeats"] == len(summary["repeats"]) for row in errors
        ),
        "source_sha256": sha256(Path(__file__)),
        "input_sha256": {str(path): sha256(path) for path in paths},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("transitions", "persistent_unlisted_cases")}
        )
    )


if __name__ == "__main__":
    main()
