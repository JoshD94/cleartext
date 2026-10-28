#!/usr/bin/env python3
"""Refit Stage B sentence weights with the GBM-based S_lex (threshold 0.3).

Standalone so a killed full training run doesn't block it. Updates the
stage_b_gbm section of results/word_model.json in place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from wordfreq import zipf_frequency  # noqa: E402

from evaluate_base import clamp, jargon_simplicity, readability_score, words  # noqa: E402
from train_weights import stage_b_sentence_weights  # noqa: E402
from word_models import lexical_simplicity_gbm  # noqa: E402


def gbm_features(original, simplified):
    ow, sw = words(original), words(simplified)
    # Threshold 0.3 tuned on Stage B train (0.5 collapses: GBM scores are
    # well-calibrated, so most news words fall below 0.5 -> zero counts).
    s_lex = lexical_simplicity_gbm(ow, sw, threshold=0.3)
    s_read = clamp(readability_score(simplified) - readability_score(original) + 0.5)
    technical_terms = {w for w in ow if zipf_frequency(w, "en") < 3.0}
    s_jargon = jargon_simplicity(ow, sw, technical_terms)
    return {"S_lex": s_lex, "S_read": s_read, "S_jargon": s_jargon}


def main() -> int:
    w_corr, w_corr_comb, sb_results = stage_b_sentence_weights(feature_fn=gbm_features)
    path = HERE / "results" / "word_model.json"
    out = json.loads(path.read_text())
    out["stage_b_gbm"] = {
        "sentence_weights": w_corr,
        "sentence_weights_combined": w_corr_comb,
        "table": sb_results,
        "s_lex_threshold": 0.3,
        "s_lex_threshold_note": "tuned on Stage B train (0.5 gives test r=-0.06)",
    }
    path.write_text(json.dumps(out, indent=2))
    print(f"updated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
