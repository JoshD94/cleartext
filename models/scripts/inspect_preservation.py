"""Inspect contextual detail evidence and term spans without rewriting text."""

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

from cleartext import ensemble as E
from cleartext.data import ROOT
from cleartext.detail_preservation import DetailPreservationScorer
from cleartext.ensemble_pipeline import EnsembleClearText
from cleartext.features import nlp, target_token
from cleartext.technical_terms import detect_terms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("runs/ensemble-context-similarity-20261002")
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="exact character offset, including for repeated words",
    )
    args = parser.parse_args()
    if (
        args.start < 0
        or args.text[args.start : args.start + len(args.target)].casefold()
        != args.target.casefold()
    ):
        parser.error("target must match the exact character offset")
    doc = nlp()(args.text)
    token = target_token(doc, args.target, args.start)
    if (
        token is None
        or token.idx != args.start
        or token.text.casefold() != args.target.casefold()
    ):
        parser.error("target must be a complete parser token")
    pipe = EnsembleClearText.load(
        args.run if args.run.is_absolute() else ROOT / args.run
    )
    target, candidates = pipe.generator.generate(token)
    distribution = (
        E.target_senses(
            E.Slot(doc, token, target=target), pipe.fit.members[0].sense_model
        )
        if target
        else {}
    )
    scorer = DetailPreservationScorer()
    spans = detect_terms(doc)
    result = {
        "text": args.text,
        "target": token.text,
        "start": token.idx,
        "scope": "Context-weighted WordNet evidence; no rewrite or calibrated semantic correctness score.",
        "sense_distribution": {
            name: float(mass) for name, mass in distribution.items()
        },
        "protected_target": any(
            span.protected
            and span.start < token.idx + len(token)
            and token.idx < span.end
            for span in spans
        ),
        "term_spans": [span.record() for span in spans],
        "candidates": [
            {**candidate, "detail_evidence": scorer.evidence(distribution, candidate)}
            for candidate in candidates
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
