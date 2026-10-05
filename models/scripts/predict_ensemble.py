import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.ensemble_pipeline import EnsembleClearText, LATEST

p = argparse.ArgumentParser()
p.add_argument("text")
p.add_argument(
    "--run", type=Path, default=LATEST, help="Local model run (default: current v7)"
)
p.add_argument(
    "--max-edits",
    type=int,
    default=None,
    help="Word edits per sentence (default: the run config)",
)
p.add_argument(
    "--full",
    action="store_true",
    help="Also apply experimental phrase and structure rules",
)
args = p.parse_args()
print(
    json.dumps(
        EnsembleClearText.load(args.run.resolve()).analyze(
            args.text,
            structure=args.full,
            phrases=args.full,
            max_word_edits=args.max_edits,
        ),
        indent=2,
        default=float,
    )
)
