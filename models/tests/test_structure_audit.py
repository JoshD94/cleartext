"""Audit accounting must separate copying, rejection, and candidate availability."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.structure.audit import beam_audit, exposure, funnel

SOURCE = "Alice reviewed the report and Bob checked the figures."
SPLIT = "Alice reviewed the report. Bob checked the figures."


def test_funnel_does_not_count_failed_identity_as_rejected_rewrite():
    rows = [
        {
            "source": SOURCE,
            "reference": SPLIT,
            "out": SOURCE,
            "details": {"raw": SOURCE, "raw_checks": {"failed": ["complete_clauses"]}},
        },
        {
            "source": SOURCE,
            "reference": SPLIT,
            "out": SOURCE,
            "details": {"raw": SPLIT, "raw_checks": {"failed": ["complement_scope"]}},
        },
    ]
    report = funnel(rows, "out", "details")
    assert report["counts"] == {"raw_copy": 1, "raw_change": 1, "blocked_change": 1}
    assert report["failed_checks_on_changed_raw_only"] == {"complement_scope": 1}


def test_exposure_accounts_for_partial_batches_across_epochs():
    rows = [
        {"source": SOURCE, "reference": SPLIT, "dataset": "fixture"} for _ in range(5)
    ]
    metadata = {
        "length_filtered_training_pairs": 0,
        "training_pairs": 5,
        "effective_batch_size": 4,
        "selected": {"step": 3},
        "seed": 4701,
        "history": [],
    }
    report = exposure(rows, metadata)
    assert report["examples_seen_at_selected_step"] == 9
    assert report["distinct_rows_seen_at_selected_step"] == 5
    assert report["unseen_in_this_continuation"] == 0


def test_exposure_refuses_to_guess_token_filtered_training_order():
    with pytest.raises(ValueError):
        exposure([], {"length_filtered_training_pairs": 1})


def test_audit_checks_lower_beams_even_after_selected_copy():
    rows = [
        {
            "id": "fixture",
            "source": SOURCE,
            "reference": SPLIT,
            "search": SOURCE,
            "search_details": {
                "model_candidates": [
                    {"text": SOURCE, "terminated": True},
                    {"text": SPLIT, "terminated": True},
                ]
            },
        }
    ]
    report, records = beam_audit(rows)
    assert report["counts"]["top_copy_hides_passing_change"] == 1
    assert len(records[0]["candidates"]) == 2
    assert (
        report["sari_overlap_proxy"]["checked_oracle_with_identity"]
        > report["sari_overlap_proxy"]["selected"]
    )
