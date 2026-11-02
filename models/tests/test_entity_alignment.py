"""Entity recognition must not determine whether an unchanged name kept its role."""

import pytest
from spacy.tokens import Doc, Span
from spacy.lang.en import English

from cleartext.lexical import entity_roles_preserved, guardrails


VOCAB = English().vocab


def doc(words, heads, deps, entities):
    parsed = Doc(VOCAB, words=words, heads=heads, deps=deps)
    parsed.ents = [Span(parsed, start, end, label=label) for start, end, label in entities]
    return parsed


@pytest.mark.parametrize("new_entities", [[(0, 1, "FAC")], []])
def test_same_name_and_role_survive_ner_label_change_or_disappearance(new_entities):
    old = doc(["Acme", "closed", "."], [1, 1, 1], ["nsubj", "ROOT", "punct"], [(0, 1, "ORG")])
    new = doc(["Acme", "shut", "."], [1, 1, 1], ["nsubj", "ROOT", "punct"], new_entities)
    assert entity_roles_preserved(old, new)


def test_every_occurrence_of_a_repeated_name_is_checked():
    words = ["Alex", "greeted", "Alex", "."]
    entities = [(0, 1, "PERSON"), (2, 3, "PERSON")]
    old = doc(words, [1, 1, 1, 1], ["nsubj", "ROOT", "dobj", "punct"], entities)
    new = doc(words, [1, 1, 1, 1], ["dobj", "ROOT", "dobj", "punct"], entities)
    assert not entity_roles_preserved(old, new)


def test_internal_name_head_can_change_while_the_whole_name_keeps_its_role():
    words = ["Acme", "Labs", "closed"]
    old = doc(words, [1, 2, 2], ["compound", "nsubj", "ROOT"], [(0, 2, "ORG")])
    new = doc(words, [2, 0, 2], ["nsubj", "compound", "ROOT"], [])
    assert entity_roles_preserved(old, new)


def test_role_swap_is_rejected_even_when_both_names_remain():
    old = doc(["Alice", "defeated", "Bob"], [1, 1, 1], ["nsubj", "ROOT", "dobj"],
              [(0, 1, "PERSON"), (2, 3, "PERSON")])
    new = doc(["Bob", "defeated", "Alice"], [1, 1, 1], ["nsubj", "ROOT", "dobj"],
              [(0, 1, "PERSON"), (2, 3, "PERSON")])
    assert not entity_roles_preserved(old, new)


def test_changed_predicate_attachment_is_rejected_without_new_ner():
    old = doc(["Alice", "said", "Bob", "won"], [1, 1, 3, 1],
              ["nsubj", "ROOT", "nsubj", "ccomp"], [(0, 1, "PERSON"), (2, 3, "PERSON")])
    new = doc(["Alice", "said", "Bob", "won"], [1, 1, 1, 1],
              ["nsubj", "ROOT", "dobj", "ccomp"], [])
    assert not entity_roles_preserved(old, new)


def test_guard_uses_original_entity_anchor(monkeypatch):
    from cleartext import lexical

    old = doc(["Acme", "closed", "."], [1, 1, 1], ["nsubj", "ROOT", "punct"], [(0, 1, "ORG")])
    new = doc(["Acme", "shut", "."], [1, 1, 1], ["nsubj", "ROOT", "punct"], [])
    monkeypatch.setattr(lexical, "nlp", lambda: lambda text: new)
    assert guardrails(old.text, new.text, old)["pass"]
