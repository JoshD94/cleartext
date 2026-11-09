"""WikiSplit labelling and feature contracts for the connector split tagger."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.features import nlp
from cleartext.structure.boundary_tagger import BoundaryTagger, features, wikisplit_label, wikisplit_rows


def split(text):
    return text.split()


@pytest.mark.parametrize("complex_text,first,second,expected", [
    ("The store closed early , and the workers went home .",
     "The store closed early .", "The workers went home .", 0),
    ("Bread and milk sold out , but the store stayed open .",
     "Bread and milk sold out .", "But the store stayed open .", 1),
    # A copied subject of up to six words is allowed in front.
    ("Alice read the report and checked the figures .",
     "Alice read the report .", "Alice checked the figures .", 0),
    # The person split somewhere else, or rewrote the halves.
    ("Alice left because Bob stayed and Carol sang .",
     "Alice left .", "Bob stayed and Carol sang .", None),
    ("The store closed early , and the workers went home .",
     "The shop shut early .", "The workers went home .", None),
])
def test_wikisplit_label(complex_text, first, second, expected):
    assert wikisplit_label(split(complex_text), split(first), split(second)) == expected


def test_wikisplit_rows_detokenize_and_skip_sentences_without_connectors(tmp_path):
    path = tmp_path / "tiny.tsv"
    path.write_text(
        "The store closed early , and the workers went home .\t"
        "The store closed early . <::::> The workers went home .\n"
        "Alice left early .\tAlice left . <::::> Early .\n", encoding="utf-8")
    rows = wikisplit_rows(path)
    assert len(rows) == 1
    assert rows[0]["source"] == "The store closed early, and the workers went home."
    assert rows[0]["connector_occurrence"] == 0


def test_features_separate_clause_and_noun_coordination():
    clause = nlp()("The store closed early, and the workers went home.")
    nouns = nlp()("Alice bought bread and milk.")
    a = features(clause, next(t for t in clause if t.lower_ == "and"))
    b = features(nouns, next(t for t in nouns if t.lower_ == "and"))
    assert a["comma_before"] == 1 and a["right_subject"] == 1 and a["matrix_coordination"] == 1
    assert b["comma_before"] == 0 and b["right_subject"] == 0 and b["right_finite"] == 0


def test_tagger_round_trip(tmp_path):
    clause = nlp()("The store closed early, and the workers went home.")
    nouns = nlp()("Alice bought bread and milk.")
    rows = [features(d, next(t for t in d if t.lower_ == "and")) for d in (clause, nouns)]
    vectorizer = DictVectorizer()
    model = LogisticRegression().fit(vectorizer.fit_transform(rows), np.array([1, 0]))
    path = tmp_path / "tagger.joblib"
    BoundaryTagger(vectorizer, model, 0.5).save(path)
    tagger = BoundaryTagger.load(path)
    assert tagger.threshold == 0.5
    assert tagger.score(clause, clause[5]) > tagger.score(nouns, nouns[3])
