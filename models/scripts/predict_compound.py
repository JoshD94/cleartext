#!/usr/bin/env python3
"""Split compound sentences using the local small parser, without an LLM."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.structure.compound import DependencyCompoundSplitter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--min-sentence-words", type=int, default=0)
    parser.add_argument("--min-clause-words", type=int, default=0)
    parser.add_argument("text")
    args = parser.parse_args()
    try:
        splitter = DependencyCompoundSplitter(args.min_sentence_words, args.min_clause_words)
    except ValueError as exc:
        parser.error(str(exc))
    result = splitter.predict(args.text)
    print(json.dumps(result, indent=2) if args.details else result["output"])


if __name__ == "__main__":
    main()
