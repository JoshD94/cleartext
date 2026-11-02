"""Authored mechanism regressions, not held-out semantic quality labels."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.structure.adapter import rewrite
from cleartext.structure.compound import DependencyCompoundSplitter


@pytest.fixture(scope="module")
def splitter():
    return DependencyCompoundSplitter()


@pytest.mark.parametrize("source,expected", [
    ("The store closed early, and the workers went home.",
     "The store closed early. The workers went home."),
    ("The store closed early and the workers went home.",
     "The store closed early. The workers went home."),
    ("The store closed early, but the workers stayed outside.",
     "The store closed early. But the workers stayed outside."),
    ("The store closed early but the workers stayed outside.",
     "The store closed early. But the workers stayed outside."),
    ("The store closed early, so the workers went home.",
     "The store closed early. So the workers went home."),
    ("The server checks each request and sends a response to the client.",
     "The server checks each request. The server sends a response to the client."),
    ("Alice read the report and checked the figures.",
     "Alice read the report. Alice checked the figures."),
    ("The neural network processed 500 records, and the analyst checked the results.",
     "The neural network processed 500 records. The analyst checked the results."),
    ("The server checks each request but sends a response.",
     "The server checks each request. But the server sends a response."),
    ("The report was reviewed, and the figures were checked.",
     "The report was reviewed. The figures were checked."),
])
def test_actual_parser_splits_and_retains_source_details(splitter, source, expected):
    result = splitter.predict(source)
    assert result["output"] == expected, result
    assert result["checks"]["pass"]
    assert not result["semantic_certified"]
    assert result["edits"]


@pytest.mark.parametrize("source", [
    "Alice bought bread and milk.",
    "Alice and Bob packed the bags.",
    "Alice opened the door and Bob.",
    "If the store closes early, Alice leaves and Bob stays.",
    "Alice said that Bob left and Carol stayed.",
    "Alice hopes to leave and Bob stays.",
    "The server, which stores records, checks requests and sends responses.",
    "The server may check requests and send responses.",
    "The store closed, but the workers did not leave.",
    "Either Alice leaves or Bob stays.",
    "Alice left so that Bob could stay.",
    "Alice worked so hard that Bob stayed.",
    "A scientist reviewed the report and checked the figures.",
    "Someone reviewed the report and checked the figures.",
    "The server has checked the request and sent the response.",
    "The server carefully checks requests and sends responses.",
    "Alice bought and read the book.",
    "Alice bought the book and read.",
    # Known parser failure: QUIC is tagged as a participle. Keep the input.
    "The gateway uses Diffie-Hellman keys, and the client stores QUIC packets.",
    "The gateway uses Diffie-Hellman keys, and the client stores each response.",
    "The server checks requests but sends responses.",
    "In 2025, Alice left and Bob stayed.",
    'Alice said "Bob left and Carol stayed."',
    "The server checks `a and b` and sends the response.",
    "The store closed (early), and the workers went home.",
    "The store closed early and the workers went home?",
])
def test_ambiguous_scope_fragments_and_literals_abstain(splitter, source):
    assert splitter.predict(source)["output"] == source


def test_passages_preserve_whitespace_and_exceed_four_sentences(splitter):
    source = "The store closed early, and the workers went home.\n\n" + "Bob waited. " * 5
    result = splitter.predict(source)
    assert result["output"] == source.replace("early, and the", "early. The")
    output, edits, details = rewrite(source, splitter)
    assert output == result["output"]
    assert edits and details["rule"] == "dependency compound split"


def test_rollback_when_final_checks_fail(splitter, monkeypatch):
    original = splitter.validate

    def reject(source, output):
        result = original(source, output)
        result.update({"pass": False, "failed": ["authored rejection probe"]})
        return result

    monkeypatch.setattr(splitter, "validate", reject)
    source = "The store closed early, and the workers went home."
    assert splitter.predict(source)["output"] == source


def test_import_and_predict_do_not_load_generative_libraries(splitter):
    before = set(sys.modules)
    splitter.predict("The store closed early, and the workers went home.")
    added = set(sys.modules) - before
    assert not any(name.split(".")[0] in {"torch", "transformers", "sentence_transformers"} for name in added)


def test_eval_variants_keep_old_ladder_and_use_local_splitter(monkeypatch):
    from evals import model_adapter, runner

    class Pipeline:
        def analyze(self, text, **kwargs):
            assert kwargs["structure"] and kwargs["phrases"] and kwargs["words"]
            assert kwargs["guardrails"] is True
            assert kwargs["structure_rewriter"].backend == "dependency_compound"
            return kwargs["structure_rewriter"].predict(text)

    monkeypatch.setattr(model_adapter, "_load_pipe", lambda run: Pipeline())
    monkeypatch.setattr(runner, "SYSTEMS", dict(runner.SYSTEMS))
    original_toggles = {name: dict(kw) for name, kw in model_adapter._ABLATION_KWARGS.items()}
    model_adapter.register_compound_systems(model_adapter.COMPOUND_SYSTEMS)
    text = "The store closed early, and the workers went home."
    expected = "The store closed early. The workers went home."
    for name in model_adapter.COMPOUND_SYSTEMS:
        assert runner.SYSTEMS[name](text) == expected
    assert model_adapter._ABLATION_KWARGS == original_toggles
