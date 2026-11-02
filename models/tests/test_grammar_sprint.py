"""Authored policy controls with fake parses; no model loading or quality labels."""

import copy
from http.client import IncompleteRead
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext import grammar_preprocessing as grammar
from cleartext.grammar_screening import screen_result


class Token:
    def __init__(self, text, idx):
        self.text, self.idx = text, idx
        self.pos_ = "AUX" if text.lower() in {
            "do", "does", "did", "is", "are", "was", "were", "has", "have"
        } else "VERB"

    def __len__(self):
        return len(self.text)


class Doc(list):
    ents = ()


def parse(text):
    doc = Doc()
    for m in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text):
        word = m.group()
        if word.endswith(("n't", "n’t")):
            doc.extend([Token(word[:-3], m.start()), Token(word[-3:], m.end() - 3)])
        else:
            doc.append(Token(word, m.start()))
    return doc


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    monkeypatch.setattr(grammar, "nlp", lambda: parse)
    monkeypatch.setattr(grammar, "detect_terms", lambda doc: [])
    forms = [{"do", "does"}, {"have", "has"}, {"go", "goes"}]
    monkeypatch.setattr(grammar, "_present_forms", lambda word: next(
        (f for f in forms if word in f), set()
    ))


def match(text, old, choices, rule="EN_CONTRACTION_SPELLING", start=None):
    start = text.index(old) if start is None else start
    return {"offset": len(text[:start].encode("utf-16-le")) // 2,
            "length": len(old.encode("utf-16-le")) // 2,
            "rule": {"id": rule},
            "replacements": [{"value": c} for c in choices]}


def client(monkeypatch, responses, **kwargs):
    instance = grammar.GrammarCorrector(**kwargs)
    calls = []

    def check(text):
        calls.append(text)
        value = responses[len(calls) - 1]
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)

    monkeypatch.setattr(instance, "check", check)
    return instance, calls


def test_repaired_contraction_gets_present_agreement_and_original_coordinates(monkeypatch):
    source, intermediate = "🙂 He dont know.", "🙂 He don't know."
    instance, calls = client(monkeypatch, [
        {"matches": [match(source, "dont", ["don't"])]},
        {"matches": [match(intermediate, "do", ["does", "did"], "HE_VERB_AGR")]},
    ])
    result = instance.correct(source)
    assert calls == [source, intermediate]
    assert result["output"] == "🙂 He doesn't know."
    assert grammar._apply(source, result["edits"]) == result["output"]
    edit, = result["edits"]
    assert source[edit["start"]:edit["end"]] == edit["source"] == "dont"
    assert edit["first_replacement"] == "don't"
    assert edit["followup_rules"] == ["HE_VERB_AGR"]
    assert result["passes"][1]["input"] == intermediate
    assert [s["rule"] for s in result["suggestions"]] == ["EN_CONTRACTION_SPELLING"]
    assert screen_result(result)["error_rules"] == ["EN_CONTRACTION_SPELLING"]
    assert not result["grammar_certified"]


def test_second_pass_is_bounded_to_new_contractions_and_preserves_tense(monkeypatch):
    source, intermediate = "He dont know. She go home.", "He don't know. She go home."
    instance, calls = client(monkeypatch, [
        {"matches": [match(source, "dont", ["don't"])]},
        {"matches": [match(intermediate, "do", ["did"], "HE_VERB_AGR"),
                     match(intermediate, "go", ["goes"], "HE_VERB_AGR")]},
    ])
    result = instance.correct(source)
    assert result["output"] == intermediate
    assert len(calls) == 2
    assert [s["reason"] for s in result["passes"][1]["suggestions"]] == [
        "unsupported_or_ambiguous_edit", "outside_repaired_contraction"
    ]


@pytest.mark.parametrize("kwargs", [{"contraction_followup": False}, {"max_edits": 1}])
def test_one_pass_control_and_total_edit_budget_skip_followup(monkeypatch, kwargs):
    source = "He dont know."
    instance, calls = client(monkeypatch, [{"matches": [match(source, "dont", ["don't"])]}], **kwargs)
    assert instance.correct(source)["output"] == "He don't know."
    assert calls == [source]


def test_original_offsets_survive_other_length_changes_and_multiple_contractions(monkeypatch):
    source = "A hour. He dont know. They hasnt left."
    intermediate = "An hour. He don't know. They hasn't left."
    instance, _ = client(monkeypatch, [
        {"matches": [match(source, "A", ["An"], "EN_A_VS_AN"),
                     match(source, "dont", ["don't"]), match(source, "hasnt", ["hasn't"])]},
        {"matches": [match(intermediate, "do", ["does"], "HE_VERB_AGR"),
                     match(intermediate, "has", ["have"], "PERS_PRONOUN_AGREEMENT")]},
    ])
    result = instance.correct(source)
    assert result["output"] == "An hour. He doesn't know. They haven't left."
    assert grammar._apply(source, result["edits"]) == result["output"]
    assert all(source[e["start"]:e["end"]] == e["source"] for e in result["edits"])


@pytest.mark.parametrize("failure", [TimeoutError(), IncompleteRead(b"", 10), {"matches": [None]}])
def test_followup_failure_rolls_back_every_edit_and_screen_is_unavailable(monkeypatch, failure):
    source = "He dont know."
    instance, _ = client(monkeypatch, [
        {"matches": [match(source, "dont", ["don't"])]}, failure
    ])
    result = instance.correct(source)
    assert result["output"] == source and not result["changed"]
    assert not result["available"] and not result["edits"]
    assert result["rolled_back"]
    assert not any(s["applied"] for s in result["suggestions"])
    assert screen_result(result)["status"] == "unavailable"


def test_incomplete_http_response_is_unavailable(monkeypatch):
    instance, _ = client(monkeypatch, [IncompleteRead(b"", 10)])
    result = instance.correct("He dont know.")
    assert result["output"] == "He dont know." and not result["available"]
    assert "IncompleteRead" in result["reason"]


def test_followup_protection_failure_rolls_back_first_pass(monkeypatch):
    source, intermediate = "He dont know.", "He don't know."
    instance, _ = client(monkeypatch, [
        {"matches": [match(source, "dont", ["don't"])]},
        {"matches": [match(intermediate, "do", ["does"], "HE_VERB_AGR")]},
    ])

    def parse_once(text):
        if text == intermediate:
            raise OSError("protection parser failed")
        return parse(text)

    monkeypatch.setattr(grammar, "nlp", lambda: parse_once)
    result = instance.correct(source)
    assert result["output"] == source and result["edits"] == []
    assert not result["available"] and result["rolled_back"]


def test_total_budget_counts_followup_operations_across_contractions(monkeypatch):
    source, intermediate = "He dont know. She dont know.", "He don't know. She don't know."
    instance, _ = client(monkeypatch, [
        {"matches": [match(source, "dont", ["don't"]),
                     match(source, "dont", ["don't"], start=source.rindex("dont"))]},
        {"matches": [match(intermediate, "do", ["does"], "HE_VERB_AGR"),
                     match(intermediate, "do", ["does"], "HE_VERB_AGR", start=intermediate.rindex("do"))]},
    ], max_edits=3)
    result = instance.correct(source)
    assert result["output"] == "He doesn't know. She don't know."
    assert sum(len(p["edits"]) for p in result["passes"]) == 3
    assert result["passes"][1]["suggestions"][1]["reason"] == "edit_limit"


@pytest.mark.parametrize("field,value", [
    ("offset", True), ("offset", -1), ("offset", "3"), ("length", True),
    ("length", -1), ("length", 1.5), ("length", None), ("rule", []),
    ("rule", {"id": []}), ("rule", {"id": ""}), ("replacements", None),
    ("replacements", [None]), ("replacements", [{"value": []}]),
    ("replacements", [{"value": None}]), ("offset", 999),
])
def test_malformed_payload_abstains_before_parser_and_never_partially_applies(monkeypatch, field, value):
    source = "He dont know."
    valid = match(source, "dont", ["don't"])
    invalid = {**valid, field: value}
    instance, _ = client(monkeypatch, [{"matches": [valid, invalid]}])
    monkeypatch.setattr(grammar, "nlp", lambda: pytest.fail("Malformed payload reached parser"))
    result = instance.correct(source)
    assert result["output"] == source and result["edits"] == []
    assert not result["available"]
    assert screen_result(result)["status"] == "unavailable"


@pytest.mark.parametrize("payload", [None, [], {"matches": None}, {"matches": [None]}])
def test_malformed_top_level_payloads_do_not_raise(monkeypatch, payload):
    instance, _ = client(monkeypatch, [payload])
    assert instance.correct("Text.")["available"] is False


@pytest.mark.parametrize("payload", [None, [], "invalid"])
def test_malformed_saved_check_is_unavailable(payload):
    assert screen_result(payload)["status"] == "unavailable"


@pytest.mark.parametrize("source,old,new", [
    ("The token is `dont`.", "dont", "don't"),
    ("The token is ``print(`dont`)``.", "dont", "don't"),
    ("Example:\n~~~text\nHe dont know.\n~~~\n", "dont", "don't"),
    ("Example:\n```text\nHe dont know.", "dont", "don't"),
    ('The text says "dont".', "dont", "don't"),
    ("A dontcare marker.", "dont", "don't"),
    ("We dont know.", "dont", "d'o'n't"),
])
def test_protected_partial_and_malformed_contractions_never_edit(monkeypatch, source, old, new):
    instance, calls = client(monkeypatch, [{"matches": [match(source, old, [new])]}])
    result = instance.correct(source)
    assert result["output"] == source and not result["edits"]
    assert len(calls) == 1


def test_followup_rechecks_protected_spans(monkeypatch):
    source, intermediate = "He dont know.", "He don't know."
    instance, _ = client(monkeypatch, [
        {"matches": [match(source, "dont", ["don't"])]},
        {"matches": [match(intermediate, "do", ["does"], "HE_VERB_AGR")]},
    ])

    def protect(text):
        doc = parse(text)
        if text == intermediate:
            doc.ents = [SimpleNamespace(start_char=3, end_char=8, label_="PERSON")]
        return doc

    monkeypatch.setattr(grammar, "nlp", lambda: protect)
    result = instance.correct(source)
    assert result["output"] == intermediate
    assert result["passes"][1]["suggestions"][0]["reason"] == "protected_span_or_insertion"


def test_missing_protection_parser_abstains_instead_of_editing_unprotected_text(monkeypatch):
    source = "He dont know."
    instance, _ = client(monkeypatch, [{"matches": [match(source, "dont", ["don't"])]}])

    def unavailable():
        raise OSError("parser unavailable")

    monkeypatch.setattr(grammar, "nlp", unavailable)
    result = instance.correct(source)
    assert result["output"] == source and not result["available"]
    assert screen_result(result)["status"] == "unavailable"


def test_no_matches_does_not_need_protection_parser(monkeypatch):
    instance, _ = client(monkeypatch, [{"matches": []}])
    monkeypatch.setattr(grammar, "nlp", lambda: pytest.fail("Unneeded parser load"))
    result = instance.correct("A valid sentence.")
    assert result["available"] and result["output"] == "A valid sentence."
