"""The opt-in window swap veto: windowing, decisions and the pipeline hook."""

import pytest
from spacy.lang.en import English

from cleartext.ensemble_pipeline import EnsembleClearText
from cleartext.swap_veto import WindowSwapVeto, changed_region, window_pair


def test_window_keeps_the_same_context_on_both_sides():
    before = "Our team found that the gradual decline in sales was not caused by pricing at all."
    after = "Our team found that the slow decline in sales was not caused by pricing at all."
    assert window_pair(before, after, 3) == ("found that the gradual decline in sales",
                                             "found that the slow decline in sales")
    assert window_pair(before, after, 0) == ("gradual", "slow")
    assert window_pair(before, after, None) == (before, after)


def test_window_at_sentence_edges_and_shared_affixes():
    assert window_pair("Evaluated twice.", "Assessed twice.", 6) == ("Evaluated twice.", "Assessed twice.")
    # A shared suffix ("ed.") is part of the changed word, not of the context.
    assert changed_region("It was evaluated.", "It was assessed.") == ("It was ", "evaluated.", "assessed.", "")
    # An article change next to the swap stays inside the changed run.
    assert changed_region("Use a large one.", "Use an enormous one.")[1:3] == ("a large", "an enormous")
    assert window_pair("Same text.", "Same text.", 2) == ("Same text.", "Same text.")


def test_veto_needs_window_nli_and_checks():
    seen = []

    def entail(premise, hypothesis):
        seen.append((premise, hypothesis))
        return 0.99 if "slow" in premise + hypothesis else 0.4

    log = []
    veto = WindowSwapVeto(entail, threshold=0.9, window=2, log=log)
    sentence = "The long and gradual decline in sales was expected."
    assert veto(sentence, sentence.replace("gradual", "slow")) is False
    assert seen == [("long and gradual decline in", "long and slow decline in"),
                    ("long and slow decline in", "long and gradual decline in")]
    assert veto(sentence, sentence.replace("gradual", "nasty")) is True
    assert [r["passes"] for r in log] == [True, False] and log[0]["word"] == "gradual"
    assert log[0]["mean"] == pytest.approx(0.99)
    failing = WindowSwapVeto(lambda p, h: 1.0, 0.9, 2, checks=lambda s, e: False)
    assert failing(sentence, sentence.replace("gradual", "slow")) is True
    unscorable = WindowSwapVeto(lambda p, h: None, 0.9, 2)
    assert unscorable(sentence, sentence.replace("gradual", "slow")) is True
    with pytest.raises(ValueError):
        WindowSwapVeto(entail, 1.5)


def test_pipeline_skips_a_vetoed_swap_and_keeps_the_next(monkeypatch):
    from cleartext import ensemble_pipeline, lexical, rewrites

    parse = English()
    parse.add_pipe("sentencizer")

    def annotated(text):
        doc = parse(text)
        for token in doc:
            if token.text in {"convoluted", "intricate"}:
                token.pos_ = "ADJ"
        return doc

    for module in (ensemble_pipeline, lexical, rewrites):
        monkeypatch.setattr(module, "nlp", lambda: annotated)
    pipe = EnsembleClearText(None, None)
    monkeypatch.setattr(pipe, "flagged", lambda doc, token: True)
    replacement = {"convoluted": ("wrong", 0.9), "intricate": ("detailed", 0.5)}

    def rank(doc, token, original, guard=True):
        word, utility = replacement[token.text]
        return {"target": token.text, "start": token.idx,
                "selected": {"word": word, "utility": utility, "fit": utility, "gain": 0.1}}

    monkeypatch.setattr(pipe, "rank_word", rank)
    log = []
    veto = WindowSwapVeto(lambda p, h: 0.2 if "wrong" in p + h else 0.98, 0.95, 1, log=log)
    result = pipe.analyze("A convoluted and intricate plan.", structure=False, phrases=False, word_veto=veto)
    assert result["output"] == "A convoluted and detailed plan."
    assert [(r["word"], r["passes"]) for r in log] == [("convoluted", False), ("intricate", True)]
    assert log[0]["window_before"] == "A convoluted and"
    # Without the veto the pipeline's default output is unchanged.
    assert pipe.analyze("A convoluted and intricate plan.", structure=False, phrases=False)["output"] == \
        "A wrong and intricate plan."
