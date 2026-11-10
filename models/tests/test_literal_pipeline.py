"""Each pipeline stage must preserve code while allowing edits to nearby prose."""

from spacy.lang.en import English

from cleartext.ensemble_pipeline import EnsembleClearText
from cleartext.lexical import guardrails
from cleartext.rewrites import phrase_rewrite


def parser(monkeypatch):
    from cleartext import ensemble_pipeline, lexical, rewrites

    parse = English()
    for module in (ensemble_pipeline, lexical, rewrites):
        monkeypatch.setattr(module, "nlp", lambda: parse)
    return parse


def test_guard_rejects_removing_code_delimiters(monkeypatch):
    parser(monkeypatch)
    result = guardrails("Use `execute()`.", "Use execute().")
    assert not result["checks"]["code_literals"]


def test_phrase_stage_skips_code_and_can_edit_the_same_phrase_in_prose(monkeypatch):
    parser(monkeypatch)
    text = "Use `prior to` prior to release."
    output, edits = phrase_rewrite(text, lambda value: 1.0 if value == "prior to" else 0.0)
    assert output == "Use `prior to` before release."
    assert len(edits) == 1


def test_final_guard_rolls_back_code_damage_from_a_structural_stage(monkeypatch):
    parser(monkeypatch)
    from cleartext import ensemble_pipeline

    source = "Use `execute()` for this task."
    monkeypatch.setattr(ensemble_pipeline, "structural", lambda text: (
        text.replace("`", ""), [{"stage": "structure"}],
    ))
    result = EnsembleClearText(None, None).analyze(source, phrases=False, words=False)
    assert result["output"] == source and result["edits"] == []
    assert result["literal_preservation"]["blocked"]
    assert result["literal_preservation"]["pass"]
    assert not result["literal_preservation"]["attempt"]["pass"]
    assert not result["structure"]["changed"]
    assert result["structure"]["after"] == source
    assert result["structure"]["attempt"]["after"] == source.replace("`", "")
    unguarded = EnsembleClearText(None, None).analyze(
        source, phrases=False, words=False, guardrails=False,
    )
    assert unguarded["output"] == source.replace("`", "")
    assert not unguarded["literal_preservation"]["pass"]


def test_word_stage_never_scores_code_tokens_and_can_edit_nearby_prose(monkeypatch):
    parse = parser(monkeypatch)
    from cleartext import ensemble_pipeline

    def annotated(text):
        doc = parse(text)
        for token in doc:
            if token.text == "convoluted":
                token.pos_ = "ADJ"
        return doc

    monkeypatch.setattr(ensemble_pipeline, "nlp", lambda: annotated)
    pipeline = EnsembleClearText(None, None)
    monkeypatch.setattr(pipeline, "flagged", lambda doc, token: True)
    scored = []

    def rank(doc, token, original, guard=True):
        scored.append(token.idx)
        return {"target": token.text, "start": token.idx, "selected": {
            "word": "complex", "utility": 0.9, "fit": 0.9, "gain": 0.1,
        }}

    monkeypatch.setattr(pipeline, "rank_word", rank)
    source = "Use `convoluted` for convoluted input."
    result = pipeline.analyze(source, structure=False, phrases=False)
    assert scored == [source.rindex("convoluted")]
    assert result["output"] == "Use `convoluted` for complex input."
    assert not result["literal_preservation"]["blocked"]


def test_demo_trace_does_not_apply_discarded_phrase_after_rollback(monkeypatch):
    import ast
    from pathlib import Path
    import re
    from types import SimpleNamespace

    parse = parser(monkeypatch)
    from cleartext import ensemble_pipeline

    source = "Use `execute()` prior to convoluted work."
    monkeypatch.setattr(ensemble_pipeline, "structural", lambda text: (
        text.replace("`", ""), [{"stage": "structure"}],
    ))
    pipeline = EnsembleClearText(None, None)
    complexity = lambda text: 1.0 if text == "prior to" else 0.0
    monkeypatch.setattr(pipeline, "complexity", complexity)
    result = pipeline.analyze(source, phrases=True, words=False)
    assert result["output"] == source and result["edits"] == []

    # Load the response helper without starting the demo's model/server setup.
    path = Path(__file__).resolve().parents[1] / "demo/server.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"trace", "skip_reason"}]
    seen = []

    def tracked_parse(text):
        seen.append(text)
        return parse(text)

    namespace = {"re": re, "PHRASES": [("prior to", "before")], "PROTECTED": [],
                 "CONTENT": {"NOUN", "VERB", "ADJ", "ADV"}, "nlp": lambda: tracked_parse,
                 "PIPE": SimpleNamespace(complexity=complexity, hardness=lambda doc, words: [],
                                         detector={"threshold": .5}, noun_threshold=.8)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    trace = namespace["trace"](source, {"phrases": True}, result["structure"], result)
    match = trace["phrases"]["matches"][0]
    assert not match["applied"] and match["attempted"]
    assert match["blocked_by"] == "code_literals"
    assert seen[0] == result["output"]
