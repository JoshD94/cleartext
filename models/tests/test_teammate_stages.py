"""Chhoeng's passive and phrase-finder stages are opt-in and run in a fixed order."""

import pytest
from spacy.lang.en import English

from cleartext.ensemble_pipeline import EnsembleClearText


def parser(monkeypatch):
    from cleartext import ensemble_pipeline, lexical, rewrites

    parse = English()
    parse.add_pipe("sentencizer")
    for module in (ensemble_pipeline, lexical, rewrites):
        monkeypatch.setattr(module, "nlp", lambda: parse)
    return parse


class Recorder:
    def __init__(self, log, name, old, new):
        self.log, self.name, self.old, self.new = log, name, old, new

    def rewrite(self, text):
        self.log.append((self.name, text))
        if self.old not in text:
            return text, []
        return text.replace(self.old, self.new), [{"stage": self.name}]


def test_defaults_leave_new_stages_off(monkeypatch):
    parser(monkeypatch)
    result = EnsembleClearText(None, None).analyze("The cake was eaten by John.", structure=False,
                                                   phrases=False, words=False)
    assert result["output"] == "The cake was eaten by John." and "passive" not in result


def test_passive_runs_before_structure_and_finder_after_authored_phrases(monkeypatch):
    parser(monkeypatch)
    from cleartext import ensemble_pipeline

    log = []
    monkeypatch.setattr(ensemble_pipeline, "structural", lambda text: (log.append(("structure", text)) or text, []))
    monkeypatch.setattr(ensemble_pipeline, "phrase_rewrite",
                        lambda text, complexity: (log.append(("authored", text)) or text, []))
    pipe = EnsembleClearText(None, None, phrase_model=object())
    result = pipe.analyze(
        "The cake was eaten by John prior to noon.", words=False,
        passive_converter=Recorder(log, "passive", "The cake was eaten by John", "John ate the cake"),
        phrase_finder=Recorder(log, "phrase_finder", "prior to", "before"),
    )
    assert [name for name, _ in log] == ["passive", "structure", "authored", "phrase_finder"]
    assert log[1][1] == "John ate the cake prior to noon."
    assert result["output"] == "John ate the cake before noon."
    assert [e["stage"] for e in result["edits"]] == ["passive", "phrase_finder"]
    assert result["passive"]["changed"]


def test_stacked_structure_backends_apply_in_order(monkeypatch):
    parser(monkeypatch)
    from cleartext.structure import adapter

    monkeypatch.setattr(adapter, "structure_checks", lambda s, o: {"pass": True, "checks": {}, "failed": []})

    class Model:
        supports_passages = True

        def __init__(self, backend, old, new):
            self.backend, self.old, self.new = backend, old, new

        def predict(self, text):
            return {"output": text.replace(self.old, self.new)}

    result = EnsembleClearText(None, None).analyze(
        "A and B and C.", phrases=False, words=False,
        structure_rewriter=[Model("first", "A and B", "A. B"), Model("second", "B and C", "B. C")])
    assert result["output"] == "A. B. C."
    assert result["structure"]["backend"] == "first+second" and len(result["structure"]["stack"]) == 2
    assert [e["stage"] for e in result["edits"]] == ["structure", "structure"]


def test_word_veto_drops_an_edit_and_takes_the_next_word(monkeypatch):
    parse = parser(monkeypatch)
    from cleartext import ensemble_pipeline

    def annotated(text):
        doc = parse(text)
        for token in doc:
            if token.text in {"convoluted", "intricate"}:
                token.pos_ = "ADJ"
        return doc

    monkeypatch.setattr(ensemble_pipeline, "nlp", lambda: annotated)
    pipe = EnsembleClearText(None, None)
    monkeypatch.setattr(pipe, "flagged", lambda doc, token: True)
    replacement = {"convoluted": ("wrong", 0.9), "intricate": ("detailed", 0.5)}

    def rank(doc, token, original, guard=True):
        word, utility = replacement[token.text]
        return {"target": token.text, "start": token.idx,
                "selected": {"word": word, "utility": utility, "fit": utility, "gain": 0.1}}

    monkeypatch.setattr(pipe, "rank_word", rank)
    source = "A convoluted and intricate plan."
    assert pipe.analyze(source, structure=False, phrases=False)["output"] == "A wrong and intricate plan."
    seen = []
    veto = lambda sentence, edited: seen.append(edited) or "wrong" in edited
    result = pipe.analyze(source, structure=False, phrases=False, word_veto=veto)
    assert result["output"] == "A convoluted and detailed plan."
    assert seen == ["A wrong and intricate plan.", "A convoluted and detailed plan."]
    assert [r.get("vetoed", {}).get("word") for r in result["trace"]] == ["wrong", None]


def test_chhoeng_passive_stage_follows_his_placeholder_setting():
    from cleartext.teammate_stages import PassiveStage

    text = "The cake was eaten by John. The thief was caught. She was born in 1990."
    output, edits = PassiveStage().rewrite(text)
    assert output == "John ate the cake. Someone caught the thief. She was born in 1990."
    assert [e["before"] for e in edits] == ["The cake was eaten by John.", "The thief was caught."]
    output, edits = PassiveStage(placeholder=None).rewrite(text)
    assert output == "John ate the cake. The thief was caught. She was born in 1990." and len(edits) == 1


def test_chhoeng_simpleppdb_finder_rewrites_and_keeps_hf_evaluate_importable():
    import sys

    from cleartext.teammate_stages import PhraseFinderStage

    stage = PhraseFinderStage(("simpleppdb",))
    output, edits = stage.rewrite("Prior to the meeting, we ate lunch. Nothing here.")
    assert output == "Before the meeting, we ate lunch. Nothing here."
    assert edits[0]["source"] == "simpleppdb" and edits[0]["phrase"] == "Prior to"
    assert "evaluate" not in sys.modules or "pos_tag" not in str(sys.modules["evaluate"].__file__)
    with pytest.raises(ValueError):
        PhraseFinderStage(("ppdb",))
