import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.generation import inflect, WordNetGenerator
from cleartext.ensemble_pipeline import build_generator
from cleartext.noun_inflection import headed_inflect, HeadedWordNetGenerator
from cleartext.features import nlp


def test_noun_modifiers_stay_unchanged_and_heads_receive_number():
    assert (
        headed_inflect("musical_composition", "NNS", "pieces") == "musical compositions"
    )
    assert (
        headed_inflect("computer_scientist", "NNS", "researchers")
        == "computer scientists"
    )
    assert headed_inflect("court_of_law", "NNS", "courts") == "courts of law"
    assert headed_inflect("arms_race", "NN", "competition") == "arms race"
    assert (
        headed_inflect("musical_composition", "NNS", "Pieces") == "Musical compositions"
    )
    assert (
        headed_inflect("musical_composition", "NNS", "PIECES") == "MUSICAL COMPOSITIONS"
    )


def test_lexical_exception_and_uncertain_or_mass_heads():
    assert headed_inflect("court_martial", "NNS", "tribunals") == "courts martial"
    assert headed_inflect("attorney_general", "NNS", "officials") in {
        None,
        "attorneys general",
    }
    assert headed_inflect("game_equipment", "NNS", "pieces") is None


def test_single_words_and_verbal_phrases_keep_existing_behavior():
    for lemma, tag, original in [
        ("person", "NNS", "people"),
        ("take_off", "VBD", "departed"),
        ("get_down", "VBZ", "starts"),
        ("complex", "JJ", "convoluted"),
    ]:
        assert headed_inflect(lemma, tag, original) == inflect(lemma, tag, original)


def test_generator_preserves_provenance_and_requires_explicit_config():
    assert type(build_generator({"generator": "expanded"})) is WordNetGenerator
    updated = build_generator({"generator": "expanded", "noun_phrase_heads": True})
    assert type(updated) is HeadedWordNetGenerator
    doc = nlp()("The missing pieces were recovered.")
    token = next(token for token in doc if token.text == "pieces")
    _, old = build_generator({"generator": "expanded"}).generate(token)
    _, new = updated.generate(token)
    first = next(
        candidate for candidate in old if candidate["lemma"] == "musical composition"
    )
    second = next(
        candidate for candidate in new if candidate["lemma"] == "musical composition"
    )
    assert first["word"] == "musicals composition"
    assert second["word"] == "musical compositions"
    assert {k: v for k, v in first.items() if k != "word"} == {
        k: v for k, v in second.items() if k != "word"
    }
