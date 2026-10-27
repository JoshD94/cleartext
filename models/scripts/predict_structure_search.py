#!/usr/bin/env python3
"""Run the opt-in candidate search with an existing local T5 checkpoint."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.structure.search import CandidateGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--details", action="store_true")
    parser.add_argument("text")
    args = parser.parse_args()
    result = CandidateGenerator(args.checkpoint, args.device).predict(args.text)
    print(json.dumps(result, indent=2) if args.details else result["output"])


if __name__ == "__main__":
    main()
