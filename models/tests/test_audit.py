import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.audit import audit_case, explain_rank, load_cases, summarize


def write_cases(tmp_path, rows):
    path = tmp_path / "input.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def case(**fields):
    return {
        "id": "example",
        "domain": "diagnostic",
        "text": "The bank faced the river bank.",
        "source": {
            "kind": "authored_diagnostic",
            "attribution": "Test fixture; ungraded.",
        },
        **fields,
    }


def test_repeated_target_requires_exact_offset(tmp_path):
    with pytest.raises(ValueError, match="exact start"):
        load_cases(write_cases(tmp_path, [case(target="bank")]))
    rows = load_cases(write_cases(tmp_path, [case(target="bank", start=25)]))
    assert rows[0]["text"][rows[0]["start"] : 29] == "bank"
    with pytest.raises(ValueError, match="does not match"):
        load_cases(write_cases(tmp_path, [case(target="bank", start=20)]))


def test_audit_requires_corpus_attribution_and_unique_ids(tmp_path):
    with pytest.raises(ValueError, match="url or path"):
        load_cases(
            write_cases(
                tmp_path, [case(source={"kind": "corpus", "attribution": "A corpus"})]
            )
        )
    with pytest.raises(ValueError, match="duplicate id"):
        load_cases(write_cases(tmp_path, [case(), case()]))


def test_input_limit_stops_before_unselected_records(tmp_path):
    path = write_cases(tmp_path, [case()])
    with path.open("a") as stream:
        stream.write("invalid unselected record\n")
    assert len(load_cases(path, limit=1)) == 1
    with pytest.raises(ValueError, match="between 1 and 100"):
        load_cases(path, limit=0)


def test_unreached_candidate_is_not_a_guardrail_pass():
    chosen = {"word": "simple", "rejections": []}
    rank = {
        "selected": chosen,
        "candidates": [
            chosen,
            {"word": "plain", "rejections": []},
            {"word": "easy", "rejections": ["low acceptance probability"]},
            {"word": "wrong", "rejections": ["argument_roles"]},
        ],
    }
    result = explain_rank(rank)
    assert [r["guard_status"] for r in result["candidates"]] == [
        "passed",
        "not_evaluated",
        "not_evaluated",
        "failed",
    ]
    assert result["candidates"][1]["selection_status"] == "not_selected_for_target"
    assert "guard_status" not in chosen


def test_coverage_does_not_claim_accuracy():
    rows = [
        {
            "original": "x",
            "output": "y",
            "domain": "cs",
            "latency_seconds": 0.2,
            "source": {"kind": "authored_diagnostic"},
            "edits": [{}],
            "trace": [],
        },
        {
            "original": "x",
            "output": "x",
            "domain": "cs",
            "latency_seconds": 0.4,
            "source": {"kind": "authored_diagnostic"},
            "edits": [],
            "trace": [],
        },
    ]
    result = summarize(rows)
    assert result["overall"]["change_coverage"] == 0.5
    assert result["overall"]["human_precision"] is None
    assert result["overall"]["latency_median_seconds"] == pytest.approx(0.3)


def test_applied_trace_distinguishes_repeated_targets():
    first = {
        "word": "shore",
        "output": "The shore faced the river bank.",
        "rejections": [],
    }
    second = {
        "word": "shore",
        "output": "The bank faced the river shore.",
        "rejections": [],
    }

    class Pipe:
        def analyze(self, text, **options):
            assert options == {
                "structure": False,
                "phrases": False,
                "max_word_edits": 1,
            }
            return {
                "original": text,
                "output": second["output"],
                "edits": [{"stage": "word", "source": "bank", "replacement": "shore"}],
                "trace": [
                    {
                        "target": "bank",
                        "start": 4,
                        "selected": first,
                        "candidates": [first],
                    },
                    {
                        "target": "bank",
                        "start": 25,
                        "selected": second,
                        "candidates": [second],
                    },
                ],
            }

    result = audit_case(Pipe(), case())
    assert [r["applied_to_output"] for r in result["trace"]] == [False, True]
    assert result["human_correctness"] is None
