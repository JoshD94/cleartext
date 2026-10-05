"""Run the opt-in six-block ClearText pipeline on local text."""

import argparse
import json
import os
import sys
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.building_blocks import BuildingBlockClearText
from cleartext.data import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("runs/ensemble-building-blocks-20261002")
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input", type=Path, help="local UTF-8 text file")
    parser.add_argument("--output", type=Path, help="new JSON output file")
    args = parser.parse_args()
    text = args.text if args.text is not None else args.input.read_text()
    if not text.strip():
        parser.error("input text must be nonempty")
    if len(text) > 20000:
        parser.error(
            "split inputs longer than 20,000 characters into smaller documents"
        )
    if args.output and args.output.exists():
        parser.error("output file must be new")
    pipe = BuildingBlockClearText.load(
        args.run if args.run.is_absolute() else ROOT / args.run
    )
    result = pipe.analyze(text)
    serialized = json.dumps(result, indent=2, default=float) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
