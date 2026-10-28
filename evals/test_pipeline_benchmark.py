"""Authored diagnostics for stage isolation and evaluation failure modes."""

from types import SimpleNamespace

import pytest

from evals.pipeline_benchmark import SentencewiseGenerator, aggregate, validate_passages
from evals.preservation import embedding_similarity, explicit_checks, nli_entailment, tfidf_cosine_similarity


@pytest.mark.parametrize("overlong", ["source", "output", "prompt"])
def test_embedding_does_not_score_only_a_truncated_prefix(monkeypatch, overlong):
    import sys
    import numpy as np
    from evals import preservation

    encoded = []

    def tokenizer(text, add_special_tokens, truncation):
        assert add_special_tokens and not truncation
        return {"input_ids": [1] * (len(text.split()) + 2)}

    tokenizer.model_max_length = 10

    def encode(texts, normalize_embeddings):
        encoded.append(texts)
        return np.array([[1., 0.], [.6, .8]])

    model = SimpleNamespace(tokenizer=tokenizer, max_seq_length=5,
                            prompts={}, default_prompt_name=None, encode=encode)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=lambda name: model
    ))
    monkeypatch.setattr(preservation, "_EMB_MODELS", {})
    assert embedding_similarity("one", "one two", "authored") == .6
    source, output = "one", "one"
    if overlong == "source":
        source = "one two three four"
    elif overlong == "output":
        output = "one two three four"
    else:
        model.prompts, model.default_prompt_name = {"task": "prefix "}, "task"
        source = "one two three"
    with pytest.warns(UserWarning, match="token limit"):
        assert embedding_similarity(source, output, "authored") is None
    assert len(encoded) == 1


def test_changed_only_denominators_do_not_reward_copying():
    def row(changed, value):
        return {"panel": "authored", "changed": changed, "readability": {},
                "reference": {}, "preservation": {"score": value, "missing": None}}

    result = aggregate([row(False, 1.0), row(True, 0.2), row(True, None)])
    assert result["all"]["n"] == 3
    assert result["all"]["preservation.score"]["n"] == 2
    assert result["changed_only"]["n"] == 2
    assert result["changed_only"]["preservation.score"] == {"mean": 0.2, "n": 1}
    assert result["all"]["preservation.missing"] == {"mean": None, "n": 0}


def test_numbers_and_negation_require_complete_items():
    result = explicit_checks("There were not 20 errors.", "There were noteworthy 120 errors.")
    assert result["numbers_preserved"] == 0
    assert result["negations_preserved"] == 0
    assert explicit_checks("The value is 20.", "The value is 20.1.")["numbers_preserved"] == 0
    assert explicit_checks("It happened in 2026.", "It happened in 2025.")["numbers_preserved"] == 0


def test_tfidf_accepts_single_letter_words():
    assert tfidf_cosine_similarity("I.", "I.") == pytest.approx(1)


def test_passage_structure_preserves_spacing_and_caches_both_modes():
    class Generator:
        calls = 0

        def predict(self, text):
            self.calls += 1
            return {"output": text, "raw": text.replace("may ", "")}

    generator = Generator()
    cache = {}
    checked = SentencewiseGenerator(generator, cache=cache)
    unchecked = SentencewiseGenerator(generator, guarded=False, cache=cache)
    source = "Alice may leave.\n\nBob may stay."
    assert checked.predict(source)["output"] == source
    assert unchecked.predict(source)["output"] == "Alice leave.\n\nBob stay."
    assert generator.calls == 2


def test_adapter_ablation_only_bypasses_checks_when_explicit():
    from cleartext.structure.adapter import rewrite

    model = SimpleNamespace(predict=lambda text: {"output": "Alice left."})
    source = "Alice may leave."
    assert rewrite(source, model)[0] == source
    assert rewrite(source, model, guard=False)[0] == "Alice left."


def test_duplicate_passages_fail_before_prediction():
    p = {"id": "one", "panel": "authored", "original": "Alice left."}
    with pytest.raises(ValueError, match="unique"):
        validate_passages([p, p])


def test_http_full_flag_enables_both_stages(monkeypatch):
    import json
    from io import BytesIO
    from evals.model_adapter import make_http_system

    payloads = []
    def respond(request, timeout):
        payloads.append(json.loads(request.data))
        return BytesIO(b'{"output":"Alice left."}')
    monkeypatch.setattr("urllib.request.urlopen", respond)
    assert make_http_system("http://127.0.0.1:8770", full=True)("Alice left.") == "Alice left."
    assert payloads[0]["structure"] is True
    assert payloads[0]["phrases"] is True
    make_http_system("http://127.0.0.1:8770", full=False)("Alice left.")
    assert payloads[1]["structure"] is False
    assert payloads[1]["phrases"] is False


def test_nli_refuses_unknown_label_mapping_and_overlength(monkeypatch):
    from evals import preservation

    class Pipe:
        calls = 0
        tokenizer = SimpleNamespace()
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=5))

        def __init__(self, size):
            class Tokenizer:
                model_max_length = 5

                def __call__(self, *args, **kwargs):
                    return {"input_ids": [1] * size}
            self.tokenizer = Tokenizer()

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return [{"label": f"LABEL_{i}", "score": v} for i, v in enumerate([0.2, 0.7, 0.1])]

    pipe = Pipe(4)
    monkeypatch.setitem(preservation._NLI_PIPES, "authored", pipe)
    with pytest.warns(UserWarning, match="entailment label"):
        assert nli_entailment("Alice left.", "Bob stayed.", "authored")["nli_mean"] is None
    pipe = Pipe(6)
    monkeypatch.setitem(preservation._NLI_PIPES, "authored", pipe)
    with pytest.warns(UserWarning, match="token limit"):
        assert nli_entailment("Alice left.", "Bob stayed.", "authored")["nli_mean"] is None
    assert pipe.calls == 0
