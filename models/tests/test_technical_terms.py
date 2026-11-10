import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.features import nlp
from cleartext.technical_terms import detect_terms, preserved_terms


def protected(text):
    return [span.text for span in detect_terms(nlp()(text)) if span.protected]


def test_known_terms_handle_plural_hyphenation_and_whole_word_boundaries():
    assert "neural networks" in protected("The neural networks classify images.")
    assert "neural-network" in protected("A neural-network classifier runs here.")
    assert "neural-networks" in protected("Several neural-networks classify images.")
    assert not protected("A neural, network example follows.")
    assert not protected("The neural networker arrived.")


def test_acronym_definitions_both_directions_and_repeated_occurrences():
    text = "A support vector machine (SVM) classifies samples. SVM runs again."
    spans = detect_terms(nlp()(text))
    assert sum(span.text == "SVM" and span.protected for span in spans) == 2
    assert preserved_terms(nlp()(text), text.replace("SVM runs", "model runs"), spans)[
        "missing"
    ] == ["SVM"]
    assert "hidden Markov model" in protected(
        "HMM (hidden Markov model) predicts states."
    )
    assert not protected("The bus (car) arrived.")


def test_defined_acronyms_cannot_be_satisfied_by_lowercase_pronouns():
    text = "United States (US) sends reports. US approves them."
    assert not preserved_terms(nlp()(text), text.replace("US approves", "us approves"))[
        "pass"
    ]
    assert preserved_terms(nlp()(text), text)["pass"]


def test_generic_noun_phrases_are_proposals_and_keep_positive_edits_available():
    spans = detect_terms(nlp()("The school bus follows the route."))
    assert any(span.text == "school bus" and not span.protected for span in spans)
    text = "The neural network processes a convoluted input."
    assert preserved_terms(nlp()(text), text.replace("convoluted", "complex"))["pass"]
    assert not preserved_terms(nlp()(text), text.replace("network", "system"))["pass"]
    repeated = "A neural network feeds another neural network."
    assert not preserved_terms(
        nlp()(repeated), repeated.replace("neural network", "system", 1)
    )["pass"]


def test_opt_in_pipeline_protects_target_spans_and_keeps_default_behavior():
    from cleartext.ensemble_pipeline import EnsembleClearText

    pipe = EnsembleClearText(None, None)
    doc = nlp()("A support vector machine (SVM) processes convoluted input.")
    target = next(token for token in doc if token.text == "machine")
    rows = [{"word": "device", "rejections": []}]
    pipe.protect_term_target(doc, target, rows)
    assert rows[0]["rejections"] == []
    pipe.technical_terms = True
    pipe.protect_term_target(doc, target, rows)
    assert rows[0]["rejections"] == ["technical_term_span"]
    ordinary = [{"word": "complex", "rejections": []}]
    pipe.protect_term_target(
        doc, next(t for t in doc if t.text == "convoluted"), ordinary
    )
    assert ordinary[0]["rejections"] == []


def test_pipeline_checks_structure_and_phrase_stages(monkeypatch):
    import cleartext.ensemble_pipeline as module

    text = "A support vector machine (SVM) learns."
    changed = text.replace("machine", "device")
    pipe = module.EnsembleClearText(None, None, phrase_model=object())
    pipe.technical_terms = True
    monkeypatch.setattr(
        module, "structural", lambda value: (changed, [{"stage": "structure"}])
    )
    monkeypatch.setattr(
        module, "phrase_rewrite", lambda value, scorer: (changed, [{"stage": "phrase"}])
    )
    result = pipe.analyze(text, words=False)
    assert result["output"] == text and result["edits"] == []
    assert result["term_preservation"]["blocked_stages"] == ["structure", "phrase"]
    assert result["term_preservation"]["pass"]
    assert not result["term_preservation"]["semantic_certified"]
