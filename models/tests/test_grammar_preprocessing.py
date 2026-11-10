"""Input grammar correction must preserve tense, protected text and edit history."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.grammar_preprocessing import GrammarCorrector, _python_offset


def match(text, old, choices, rule, start=None):
    start = text.index(old) if start is None else start
    return {
        "offset": len(text[:start].encode("utf-16-le")) // 2,
        "length": len(old.encode("utf-16-le")) // 2,
        "replacements": [{"value": word} for word in choices],
        "rule": {"id": rule},
    }


def corrector(monkeypatch, matches, **kwargs):
    instance = GrammarCorrector(**kwargs)
    monkeypatch.setattr(instance, "check", lambda text: {"matches": matches})
    return instance


@pytest.mark.parametrize(
    "source,old,choices,rule,expected",
    [
        ("She go home.", "go", ["goes", "went"], "HE_VERB_AGR", "She goes home."),
        (
            "They is ready.",
            "is",
            ["are", "were"],
            "PERS_PRONOUN_AGREEMENT",
            "They are ready.",
        ),
        (
            "We was ready.",
            "was",
            ["are", "were"],
            "PERS_PRONOUN_AGREEMENT",
            "We were ready.",
        ),
        ("He have a car.", "have", ["has", "had"], "HE_VERB_AGR", "He has a car."),
        (
            "She does not knows.",
            "knows",
            ["know"],
            "AUXILIARY_DO_WITH_INCORRECT_VERB_FORM",
            "She does not know.",
        ),
        (
            "An useful tool arrived.",
            "An",
            ["A"],
            "EN_A_VS_AN",
            "A useful tool arrived.",
        ),
        (
            "We dont understand.",
            "dont",
            ["don't"],
            "EN_CONTRACTION_SPELLING",
            "We don't understand.",
        ),
        ("A hour passed.", "A", ["An"], "EN_A_VS_AN", "An hour passed."),
    ],
)
def test_supported_edits_preserve_tense_and_have_an_audit_trail(
    monkeypatch, source, old, choices, rule, expected
):
    result = corrector(monkeypatch, [match(source, old, choices, rule)]).correct(source)
    assert result["output"] == expected
    assert result["original"] == source
    assert result["edits"][0]["source"] == old
    assert result["edits"][0]["stage"] == "grammar"
    assert not result["grammar_certified"]


@pytest.mark.parametrize(
    "source,old,choices,rule",
    [
        ("She go home.", "go", ["went"], "HE_VERB_AGR"),
        ("She go home.", "go", ["goes", "go"], "UNKNOWN_RULE"),
        ("They were ready.", "were", ["are"], "PERS_PRONOUN_AGREEMENT"),
        ("They may go home.", "may", ["must"], "HE_VERB_AGR"),
        ("She does not know.", "not", [""], "EN_CONTRACTION_SPELLING"),
        ("An useful tool arrived.", "An", ["A", "a"], "EN_A_VS_AN"),
        ("The neural network runs.", "network", ["networks"], "BASE_FORM"),
        ("There are 25 samples.", "25", ["26"], "UNKNOWN_RULE"),
        ('The text says "dont".', "dont", ["don't"], "EN_CONTRACTION_SPELLING"),
        ("The token is `dont`.", "dont", ["don't"], "EN_CONTRACTION_SPELLING"),
    ],
)
def test_unsupported_ambiguous_and_protected_changes_abstain(
    monkeypatch, source, old, choices, rule
):
    result = corrector(monkeypatch, [match(source, old, choices, rule)]).correct(source)
    assert result["output"] == source
    assert not result["changed"] and not result["edits"]


def test_detected_person_name_is_protected_even_for_article_rule(monkeypatch):
    from spacy.tokens import Span

    from cleartext import grammar_preprocessing as module

    source = "An Nguyen arrived."
    doc = module.nlp()(source)
    # Test the protection policy independently of imperfect name recognition.
    doc.ents = [Span(doc, 0, 2, label="PERSON")]
    instance = corrector(monkeypatch, [match(source, "An", ["A"], "EN_A_VS_AN")])
    monkeypatch.setattr(module, "nlp", lambda: lambda text: doc)
    result = instance.correct(source)
    assert result["output"] == source
    assert result["suggestions"][0]["reason"] == "protected_span_or_insertion"


def test_java_offsets_after_emoji_and_multiple_edits_use_original_positions(
    monkeypatch,
):
    source = "🙂 We dont use an useful tool."
    matches = [
        match(source, "dont", ["don't"], "EN_CONTRACTION_SPELLING"),
        match(source, "an", ["a"], "EN_A_VS_AN"),
    ]
    result = corrector(monkeypatch, matches).correct(source)
    assert result["output"] == "🙂 We don't use a useful tool."
    assert len(result["edits"]) == 2
    with pytest.raises(ValueError):
        _python_offset(source, 1)


def test_malformed_match_cannot_leave_a_partial_correction(monkeypatch):
    source = "We dont use this tool."
    matches = [
        match(source, "dont", ["don't"], "EN_CONTRACTION_SPELLING"),
        {"offset": 999},
    ]
    result = corrector(monkeypatch, matches).correct(source)
    assert result["output"] == source and result["edits"] == []
    assert "Invalid grammar matches" in result["reason"]


def test_server_failure_keeps_original_and_is_visible(monkeypatch):
    instance = GrammarCorrector()

    def unavailable(text):
        raise TimeoutError()

    monkeypatch.setattr(instance, "check", unavailable)
    result = instance.correct("We dont know.")
    assert result["output"] == "We dont know."
    assert not result["available"]
    assert "TimeoutError" in result["reason"]


@pytest.mark.parametrize(
    "server",
    [
        "https://api.languagetool.org",
        "http://example.com",
        "http://127.0.0.1.example.com",
        "http://user:password@localhost:8786",
        "http://localhost:8786/v2/check",
    ],
)
def test_cloud_and_unexpected_endpoint_configuration_is_rejected(server):
    with pytest.raises(ValueError, match="loopback"):
        GrammarCorrector(server)


def test_overlap_and_edit_limit_are_recorded(monkeypatch):
    source = "We dont use an useful tool."
    first = match(source, "dont", ["don't"], "EN_CONTRACTION_SPELLING")
    matches = [first, first, match(source, "an", ["a"], "EN_A_VS_AN")]
    result = corrector(monkeypatch, matches, max_edits=1).correct(source)
    assert result["output"] == "We don't use an useful tool."
    assert [s["reason"] for s in result["suggestions"]] == [
        None,
        "overlapping_edit",
        "edit_limit",
    ]


def test_ensemble_runs_correction_before_structure_and_preserves_original(monkeypatch):
    from cleartext import ensemble_pipeline as module

    source = "They is ready."
    grammar = corrector(
        monkeypatch, [match(source, "is", ["are", "were"], "PERS_PRONOUN_AGREEMENT")]
    )
    received = []

    def structure(text):
        received.append(text)
        return text, []

    monkeypatch.setattr(module, "structural", structure)
    pipe = module.EnsembleClearText(None, None)
    result = pipe.analyze(source, words=False, phrases=False, grammar_corrector=grammar)
    assert received == ["They are ready."]
    assert result["original"] == source and result["output"] == "They are ready."
    assert result["grammar"]["changed"] and result["edits"][0]["stage"] == "grammar"
    baseline = pipe.analyze(source, words=False, phrases=False)
    assert baseline["output"] == source and "grammar" not in baseline


def test_seq2seq_preprocessing_discards_stale_original_parse(monkeypatch):
    from cleartext.structure.seq2seq import SentenceGenerator

    source = "They is ready."
    grammar = corrector(
        monkeypatch, [match(source, "is", ["are", "were"], "PERS_PRONOUN_AGREEMENT")]
    )
    original_predict = SentenceGenerator.predict
    calls = []

    def frozen_predict(self, text, doc=None):
        calls.append((text, doc))
        return {"output": text, "raw": text, "raw_checks": {"pass": True}}

    monkeypatch.setattr(SentenceGenerator, "predict", frozen_predict)
    instance = SentenceGenerator.__new__(SentenceGenerator)
    result = original_predict(instance, source, doc=object(), grammar_corrector=grammar)
    assert calls == [("They are ready.", None)]
    assert result["original"] == source
    assert result["preprocessed"] == "They are ready."
