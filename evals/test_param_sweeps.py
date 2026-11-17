"""Param-sweep IDs, panel guards and the opt-in stage knobs, without native models."""

import json

import pytest

from evals import ensemble_study as S
from evals import param_sweeps as P

WINNER = {"words": "off", "phrases": "authored", "structure": "t5+compound_tagger"}


def test_ids_round_trip_and_default_ids_match_the_study():
    assert P.make_id(WINNER) == S.config_id(WINNER)
    name = P.make_id(WINNER, {"t5.waive": ["complete_clauses", "argument_roles"], "t5.min_entailment": 0.8})
    assert name.startswith(S.config_id(WINNER) + "#")
    config, params = P.parse(name)
    assert config == S.normalize(WINNER)
    assert params == {"t5.min_entailment": 0.8, "t5.waive": ["argument_roles", "complete_clauses"]}
    # Order of a list parameter does not create a new config.
    assert name == P.make_id(WINNER, {"t5.waive": ["argument_roles", "complete_clauses"], "t5.min_entailment": 0.8})
    assert P.parse("original") == ("original", {})


@pytest.mark.parametrize("base,params", [
    (WINNER, {"w.threshold": 0.8}),                          # words are off
    ({**WINNER, "words": "v7"}, {"w.veto": 0.9}),            # veto needs contextual_veto
    ({**WINNER, "structure": "compound_tagger"}, {"t5.min_words": 10}),
    ({**WINNER, "structure": "t5"}, {"cmp.threshold": 0.5}),
    (WINNER, {"finder.model_threshold": 0.5}),               # no finder level
    (WINNER, {"t5.candidates": 4}),                          # more than the 2 beams
    (WINNER, {"t5.passage": "loose"}),
    ({**WINNER, "passive": "on"}, {"passive.placeholder": "they"}),
    (WINNER, {"no.such": 1}),
])
def test_parameters_a_config_would_ignore_are_rejected(base, params):
    with pytest.raises(ValueError):
        P.make_id(base, params)


def test_design_names_and_groups(tmp_path):
    design = tmp_path / "d.json"
    design.write_text(json.dumps({"configs": [
        {"base": WINNER},
        {"base": WINNER, "params": {"t5.min_words": 12}, "group": "t5.min_words", "value": 12},
        "original"]}))
    names = P.design_names(design)
    assert names[0] == S.config_id(WINNER) and names[2] == "original"
    assert P.design_groups(design) == {names[1]: ("t5.min_words", 12)}
    design.write_text(json.dumps({"configs": [{"base": WINNER}, {"base": WINNER, "params": {}}]}))
    with pytest.raises(ValueError):
        P.design_names(design)


def test_locked_test_passages_are_refused(tmp_path):
    rows = [{"id": "a", "original": "One. Two."}, {"id": "b", "original": "Three."}]
    ok = tmp_path / "dev.json"
    ok.write_text(json.dumps(rows))
    assert P.guard_panel(ok) == rows
    locked = tmp_path / "test-LOCKED.json"
    locked.write_text(json.dumps(rows))
    with pytest.raises(SystemExit):
        P.guard_panel(locked)
    split = tmp_path / "mixed.json"
    split.write_text(json.dumps(rows + [{"id": "c", "original": "Four.", "split": "test"}]))
    with pytest.raises(SystemExit):
        P.guard_panel(split)
    test_id = (S.ROOT / "evals/data/testset_v3/test_ids.txt").read_text().split()[0]
    renamed = tmp_path / "dev2.json"
    renamed.write_text(json.dumps(rows + [{"id": test_id, "original": "Five."}]))
    with pytest.raises(SystemExit):
        P.guard_panel(renamed)


def test_selection_breaks_fk_ties_by_kea_s():
    base = {"n": 20, "nli_mean": 0.97, "nli_below_080": 0, "numbers_failed": 0, "negations_failed": 0,
            "entities": 0.9}
    summaries = {"a": {**base, "fk_grade": 10.0, "kea_S": 0.4}, "b": {**base, "fk_grade": 10.05, "kea_S": 0.5},
                 "c": {**base, "fk_grade": 9.0, "nli_mean": 0.90, "kea_S": 0.9},
                 "rule_based": {**base, "fk_grade": 12, "kea_S": 0}}
    chosen = P.select(summaries, 0.8)
    assert chosen["winner"] == "b" and chosen["tied"] == ["a", "b"] and "c" not in chosen["passing"]


def test_adoption_needs_gates_over_edit_a_clear_fk_drop_and_no_real_nli_loss():
    row = {"fk_grade": {"mean": -0.3, "ci95": [-0.5, -0.1]}, "nli_mean": {"mean": -0.004, "ci95": [-0.01, 0]}}
    assert P.adoption(row, True, True)["adopt"]
    assert not P.adoption(row, False, True)["adopt"]
    assert not P.adoption(row, True, False)["adopt"]
    assert not P.adoption({**row, "fk_grade": {"mean": -0.05, "ci95": [-0.1, -0.01]}}, True, True)["adopt"]
    assert not P.adoption({**row, "fk_grade": {"mean": -0.3, "ci95": [-0.6, 0.05]}}, True, True)["adopt"]
    assert not P.adoption({**row, "nli_mean": {"mean": -0.02, "ci95": [-0.04, 0]}}, True, True)["adopt"]


def test_phrase_rewrite_defaults_unchanged_and_cap_is_opt_in():
    from cleartext.rewrites import phrase_rewrite
    text = "In order to win, they met at the present time."
    scorer = lambda s: 0.9 if s in ("in order to", "at the present time") else 0.1
    one, edits = phrase_rewrite(text, scorer)
    assert len(edits) == 1 and one != text
    assert phrase_rewrite(text, scorer, min_gain=0.9) == (text, [])
    two, edits = phrase_rewrite(text, scorer, max_edits=2)
    assert len(edits) == 2 and "in order to" not in two.lower() and "present time" not in two


class Beams:
    def __init__(self, table):
        self.table = table

    def candidates(self, sentence):
        return [{"raw": r, "terminated": True} for r in self.table.get(sentence, [sentence])]


def test_tuned_splitter_defaults_follow_the_guarded_t5_contract():
    from cleartext.structure.tuned import TunedSplitter
    source = "The team wrote the code and the team tested it."
    good = "The team wrote the code. The team tested it."
    bad = "The team wrote. Code tested."
    text = source + " " + "It works."
    assert TunedSplitter(Beams({source: [good]})).predict(text)["output"] == good + " It works."
    assert TunedSplitter(Beams({source: [bad]})).predict(text)["output"] == text
    # A lower beam is tried only when asked, and only if the top one fails.
    assert TunedSplitter(Beams({source: [bad, good]})).predict(text)["output"] == text
    assert TunedSplitter(Beams({source: [bad, good]}), candidates=2).predict(text)["output"] == good + " It works."
    assert TunedSplitter(Beams({source: [good]}), min_words=20).predict(text)["output"] == text
    # The NLI floor rejects a split that would otherwise pass.
    strict = TunedSplitter(Beams({source: [good]}), entailment=lambda a, b: 0.5, min_entailment=0.9)
    assert strict.predict(text)["output"] == text
    assert TunedSplitter(Beams({}), passage="off").validate(text, "x")["pass"]
    with pytest.raises(ValueError):
        TunedSplitter(Beams({}), waive=["no_such_check"])


def test_finder_settings_are_validated():
    from cleartext.teammate_stages import FINDER_SETTINGS, PhraseFinderStage
    assert set(FINDER_SETTINGS) == {"model_threshold", "min_entailment", "max_fluency_drop", "min_similarity"}
    with pytest.raises(ValueError):
        PhraseFinderStage(("manual",), settings={"temperature": 1})


def test_unlock_needs_a_committed_finalists_file_and_runs_only_it(tmp_path):
    loose = S.ROOT / "evals" / "_uncommitted_finalists_test.json"
    loose.write_text(json.dumps({"configs": [{"base": WINNER}]}))
    try:
        with pytest.raises(SystemExit):
            P.committed_finalists(loose, tmp_path / "dev-plan.json")
    finally:
        loose.unlink()
    with pytest.raises(SystemExit):
        P.committed_finalists(S.ROOT / "evals/reports/20261116-param-sweeps/design-0-references.json", None)
    committed = S.ROOT / "evals/reports/20261116-param-sweeps/design-0-references.json"
    found = P.committed_finalists(committed, tmp_path / "dev-plan.json")
    assert found["configs"] == P.design_names(committed) and len(found["commit"]) == 40
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"configs": [{"base": WINNER}]}))
    plan = {"panels": {}, "unlock_test": True, "finalists": found}
    with pytest.raises(SystemExit):
        P.preflight(plan, design=other)
