"""Count where listed substitutions disappear from cached development cases."""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cleartext.audit import sha256
from cleartext.data import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [ROOT / f"data/cache/refine-v7-benchls-{part}.pkl"
             for part in ("dev", "holdout")]
    cases = []
    for path in paths:
        with path.open("rb") as stream:
            cases.extend(pickle.load(stream)[1])
    result = {
        "scope": "Post-hoc description of all reused BenchLS development with the target supplied. "
                 "A missing listed substitute does not prove no valid simplification exists. "
                 "No threshold selection or adoption-bar change.",
        "meaning_floor": 0.10,
        "cases": len(cases),
        "with_candidates": sum(bool(c["candidates"]) for c in cases),
        "with_listed_candidate": sum(any(r["gold"] for r in c["candidates"]) for c in cases),
        "with_guarded_listed_candidate": sum(
            any(r["gold"] and r["guard"] for r in c["candidates"]) for c in cases),
        "with_eligible_listed_candidate": sum(
            any(r["gold"] and r["guard"] and r["x"][0] >= .10
                for r in c["candidates"]) for c in cases),
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        "source_sha256": sha256(Path(__file__)),
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
