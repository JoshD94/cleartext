"""CLI entry point: run the ClearText eval pipeline on sample passages.

Usage:
    python run_evals.py [--passages data/sample_passages.json]
                        [--systems original,rule_based]
                        [--out results]
                        [--embedding-model all-MiniLM-L6-v2]

Evaluates each system on readability + meaning preservation and writes a
JSON results file plus a markdown results table.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.runner import (  # noqa: E402
    default_ablation_order,
    register_system,
    run_evaluation,
    save_results,
    to_markdown_table,
)


def main():
    ap = argparse.ArgumentParser(description="Run the ClearText evaluation pipeline.")
    ap.add_argument("--passages", default="data/sample_passages.json")
    ap.add_argument(
        "--systems",
        default="original,rule_based",
        help="Comma-separated system names. 'original' and 'rule_based' are "
             "built in; team modules (lexical_only, structural_only, "
             "lexical_structural, full_no_guardrails, full_pipeline) can be "
             "registered in this file once implemented.",
    )
    ap.add_argument("--out", default="results")
    ap.add_argument("--embedding-model", default=None,
                    help="sentence-transformers model for embedding similarity "
                         "(optional dependency)")
    ap.add_argument("--nli-model", default=None,
                    help="transformers MNLI model for NLI entailment scores "
                         "(optional dependency)")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    passages = json.loads((base / args.passages).read_text(encoding="utf-8"))
    system_names = [s.strip() for s in args.systems.split(",") if s.strip()]

    # --- Team hook: register implemented pipeline variants here. ---
    # Example:
    #   from cleartext.lexical import LexicalPipeline
    #   register_system("lexical_only", LexicalPipeline().simplify)
    # The full 7-way ladder from the proposal is:
    #   default_ablation_order()
    _ = default_ablation_order  # keep import used / documented

    results = run_evaluation(system_names, passages,
                             embedding_model=args.embedding_model,
                             nli_model=args.nli_model)
    json_path, md_path = save_results(results, base / args.out)
    print(f"Wrote {json_path}\nWrote {md_path}\n")
    print(to_markdown_table(results))
    try:
        from evals.build_status import build as build_status_page
        html_path = build_status_page()
        print(f"Wrote {html_path}")
    except Exception as exc:  # never break an eval run over the status page
        print(f"(status page rebuild skipped: {exc})")


if __name__ == "__main__":
    main()
