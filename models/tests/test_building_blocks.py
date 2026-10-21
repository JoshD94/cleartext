import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.features import nlp
from cleartext.grammar_validation import GrammarValidator
from cleartext.document_consistency import (
    DocumentConsistencyChecker,
    acronym_definitions,
)
from cleartext.edit_quality import EditQualityEvaluator
from cleartext.jargon_explanations import JargonExplainer


def test_grammar_rejects_new_agreement_and_article_errors_but_accepts_good_edits():
    validator = GrammarValidator()
    assert not validator.validate("The dogs eat food.", "The dogs eats food.")["pass"]
    assert not validator.validate("A teacher arrived.", "A apple arrived.")["pass"]
    assert validator.validate("An apple arrived.", "A pear arrived.")["pass"]
    assert validator.validate("The dogs eat food.", "The cats eat food.")["pass"]
    assert validator.validate("The dogs eats food.", "The dogs eats food.")["pass"]
    assert (
        validator.validate("An XQZ device operates.", "An XQZ device operates.")[
            "output_issues"
        ]
        == []
    )
    assert not validator.validate("This result is clear.", "These result is clear.")[
        "pass"
    ]


def test_grammar_handles_pronouns_auxiliaries_and_passive_positive_controls():
    validator = GrammarValidator()
    for text in [
        "I am ready.",
        "You were ready.",
        "The dogs are fed.",
        "The cat and dog eat food.",
    ]:
        assert validator.validate(text, text)["pass"]
        assert not validator.validate(text, text)["output_issues"]
    assert not validator.validate("They are ready.", "They is ready.")["pass"]
    assert validator.validate(
        "This sentence is convoluted.", "This sentence is complex."
    )["pass"]


def test_coordination_is_local_and_existing_errors_do_not_become_new_after_a_word_edit():
    validator = GrammarValidator()
    assert validator.validate(
        "The size and number of links relates to the result.",
        "The size and number of links connects to the result.",
    )["pass"]
    assert validator.validate(
        "The productivity of crops and pastures, as well as health, declines.",
        "The productivity of crops and pastures, as well as health, decreases.",
    )["pass"]
    assert not validator.validate(
        "The cat and dog eat food.", "The cat and dog eats food."
    )["pass"]


def test_document_definitions_detect_conflicts_and_preserve_repeated_entities():
    checker = DocumentConsistencyChecker()
    text = "Hidden Markov model (HMM) predicts states. HMM runs again."
    assert acronym_definitions(nlp()(text))[0]["expansion"] == "Hidden Markov model"
    assert not checker.check(text, text.replace("HMM runs", "model runs"))["pass"]
    conflict = text + " Huge memory machine (HMM) uses RAM."
    assert "conflicting_acronym_definitions" in checker.check(text, conflict)["failed"]
    assert checker.check(conflict, conflict)["pass"]
    assert not checker.check(
        "Alice met Bob. Alice departed.", "Alice met Bob. She departed."
    )["pass"]
    assert not checker.check("The cost is 5 dollars.", "The cost is 50 dollars.")[
        "pass"
    ]


def test_document_consistency_does_not_ban_different_senses_without_evidence():
    checker = DocumentConsistencyChecker()
    edits = [
        {"stage": "word", "source": "bank", "replacement": word}
        for word in ["shore", "lender"]
    ]
    report = checker.check("The bank is here.", "The bank is here.", edits)
    assert report["pass"] and report["replacement_review"]
    for edit in edits:
        edit["sense_id"] = "bank.n.01"
    assert not checker.check("The bank is here.", "The bank is here.", edits)["pass"]


def test_jargon_uses_document_expansions_and_abstains_on_conflicts():
    text = "Hidden Markov model (HMM) predicts states. HMM runs again."
    glosses = JargonExplainer().explain(nlp()(text))
    defined = [row for row in glosses if row["term"] == "HMM"]
    assert len(defined) == 2 and all(
        row["explanation"] == "Hidden Markov model" for row in defined
    )
    conflict = text + " Huge memory machine (HMM) uses RAM."
    assert all(
        row["status"] == "ambiguous" and row["explanation"] is None
        for row in JargonExplainer().explain(nlp()(conflict))
        if row["term"] == "HMM"
    )
    assert text == nlp()(text).text
    varied = text + " hidden Markov model (HMM) predicts again."
    assert all(
        row["status"] == "available"
        for row in JargonExplainer().explain(nlp()(varied))
        if row["term"] == "HMM"
    )


def test_jargon_returns_exact_dictionary_gloss_and_context_uncertainty():
    from nltk.corpus import wordnet as wn

    gloss = next(
        row
        for row in JargonExplainer().explain(nlp()("The operating system runs."))
        if row["term"] == "operating system"
    )
    assert gloss["status"] == "available"
    assert gloss["explanation"] == wn.synset(gloss["sense_id"]).definition()
    sense = SimpleNamespace(
        distribution=lambda words, index, lemma, pos: {
            "bank.n.01": 0.5,
            "depository_financial_institution.n.01": 0.5,
        }
    )
    pipe = SimpleNamespace(
        flagged=lambda doc, token: token.lower_ == "bank",
        fit=SimpleNamespace(members=[SimpleNamespace(sense_model=sense)]),
    )
    uncertain = JargonExplainer().explain(nlp()("The bank is nearby."), pipe)[0]
    assert uncertain["status"] == "ambiguous" and uncertain["explanation"] is None
    sense.distribution = lambda words, index, lemma, pos: {
        "bank.n.01": 0.9,
        "depository_financial_institution.n.01": 0.1,
    }
    certain = JargonExplainer().explain(nlp()("The bank is nearby."), pipe)[0]
    assert certain["sense_id"] == "bank.n.01" and not certain["semantic_certified"]


def test_quality_reports_readability_and_checks_separately():
    pipe = SimpleNamespace(
        difficulty=SimpleNamespace(
            score=lambda words: np.asarray(
                [0.8 if word == "convoluted" else 0.2 for word in words]
            )
        ),
        reading_level=lambda doc: 4.0,
    )
    result = EditQualityEvaluator().evaluate(
        "This sentence is convoluted.", "This sentence is complex.", pipe
    )
    assert result["readability"]["word_complexity_reduction"] > 0
    assert result["grammar"]["pass"] and result["document_consistency"]["pass"]
    assert (
        result["token_difference_ratio"] > 0 and result["human_quality_score"] is None
    )
    assert (
        EditQualityEvaluator().evaluate("Hello.", "Hello.")["token_difference_ratio"]
        == 0
    )
    assert (
        EditQualityEvaluator().evaluate("", "", pipe)["readability"]["before"][
            "mean_word_complexity"
        ]
        is None
    )


def test_grammar_candidate_rejection_falls_back_to_next_candidate():
    from cleartext.building_blocks import BuildingBlockClearText

    pipe = BuildingBlockClearText(None, None, decision_threshold=0.5)
    pipe.blocks = {"grammar_validation": True}
    doc = nlp()("The dogs eat food.")
    rows = [
        {"accept": 0.9, "rejections": [], "output": "The dogs eats food."},
        {"accept": 0.8, "rejections": [], "output": "The dogs consume food."},
    ]
    assert pipe.first_guarded(doc, rows) is rows[1]
    assert "grammar:subject_verb_number" in rows[0]["rejections"]


def test_wrapper_rolls_back_unsafe_output_and_reports_final_quality(monkeypatch):
    from cleartext.building_blocks import BuildingBlockClearText
    from cleartext.ensemble_pipeline import EnsembleClearText

    pipe = BuildingBlockClearText(None, None)
    pipe.blocks = {"grammar_validation": True, "edit_quality": True}
    monkeypatch.setattr(
        EnsembleClearText,
        "analyze",
        lambda self, text, **kwargs: {
            "original": text,
            "output": "The dogs eats food.",
            "edits": [{"stage": "word"}],
            "trace": [],
        },
    )
    pipe.difficulty = SimpleNamespace(score=lambda words: np.zeros(len(words)))
    pipe.reading_level = lambda doc: None
    result = pipe.analyze("The dogs eat food.")
    assert result["output"] == "The dogs eat food." and result["edits"] == []
    assert result["block_rollbacks"] == ["grammar_validation"]
    assert result["grammar_validation"]["pass"]
    assert not result["rejected_block_reports"]["grammar_validation"]["pass"]
    assert result["edit_quality"]["grammar"]["pass"]
