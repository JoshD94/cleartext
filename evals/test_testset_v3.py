"""Test set v3: panel builder helpers, split lock, new metrics and the report maths."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from evals import build_testset_v3 as B
from evals import ensemble_study as E
from evals import testset_v3 as T
from evals.human_score import simplicity, trained_weights
from evals.preservation import explicit_checks, extract_entities, extract_entities_capitalized


# --- entity check -------------------------------------------------------------

def test_entities_drop_sentence_initial_discourse_and_common_words():
    text = "However, the plan failed. Patients should call Dr. Smith in Boston."
    assert extract_entities_capitalized(text) == ["However", "Patients", "Dr", "Smith", "Boston"]
    assert extract_entities(text) == ["Dr", "Smith", "Boston"]


def test_entities_keep_names_at_sentence_start():
    text = "In Washington, the Senate met. Apple said Apple Music grew. Desmopressin is a hormone."
    # "In" is stripped from the span; "Apple" also appears mid-sentence; the
    # rare "Desmopressin" stays an entity at the start of a sentence.
    assert extract_entities(text) == ["Washington", "Senate", "Apple", "Apple Music", "Desmopressin"]
    assert extract_entities('The United States agreed. "Yes," said Biden.') == ["United States", "Biden"]


def test_entity_preservation_no_longer_penalises_a_dropped_however():
    source = "However, the board approved the merger with Contoso."
    output = "The board approved the merger with Contoso."
    checks = explicit_checks(source, output)
    assert checks["entities_preserved"] == 1.0
    assert checks["entities_preserved_capitalized"] == 0.5


# --- Kea-Roy score ------------------------------------------------------------

def test_kea_score_matches_train_weights_features():
    train_weights = pytest.importorskip("evals.train_weights")
    weights = trained_weights()
    source = "The committee postponed the deliberations indefinitely due to unforeseen circumstances."
    output = "The group put off the talks. Something they did not expect happened."
    expected = train_weights.sentence_features(source, output, weights["word"])
    got = simplicity(source, output)
    for key in ("S_lex", "S_read", "S_jargon"):
        assert got[key] == pytest.approx(expected[key])
    assert got["S"] == pytest.approx(train_weights.s_score(expected, weights["sentence"]))


def test_kea_score_rewards_simpler_output_and_is_flat_for_a_copy():
    source = "The committee postponed the deliberations indefinitely due to unforeseen circumstances."
    copy = simplicity(source, source)
    assert copy["S_lex"] == 0 and copy["S_jargon"] == 0 and copy["S_read"] == pytest.approx(0.5)
    assert simplicity(source, "The group put off the talks.")["S"] > copy["S"]


# --- over-edit ----------------------------------------------------------------

def test_changed_token_rate():
    assert T.changed_token_rate("The cat sat.", "The cat sat.") == 0
    assert T.changed_token_rate("The cat sat on the mat.", "The dog sat on the mat.") == pytest.approx(1 / 7)
    # Case is ignored, so a sentence split costs only the inserted period.
    assert T.changed_token_rate("A b c.", "A b. C.") == pytest.approx(1 / 4)
    # A replaced run counts its longer side.
    assert T.changed_token_rate("A b.", "A x y z.") == pytest.approx(3 / 3)


# --- bootstrap and paired deltas ----------------------------------------------

def test_stratified_bootstrap_is_seeded_and_brackets_the_mean():
    values = list(np.random.default_rng(0).normal(1.0, 0.5, 60))
    strata = ["a"] * 30 + ["b"] * 30
    first, second = T.bootstrap(values, strata, 2000), T.bootstrap(values, strata, 2000)
    assert first == second
    assert first["ci95"][0] < first["mean"] < first["ci95"][1]
    assert T.bootstrap([np.nan, 2.0], ["a", "a"], 100)["n"] == 1


def test_paired_deltas_and_gates():
    rows = lambda vals: [{"domain": "d", "fk_grade": v} for v in vals]
    delta = T.paired(rows([10, 11, 12]), rows([11, 11, 13]), "fk_grade")
    assert delta["mean"] == pytest.approx(-2 / 3) and delta["passages_nonzero"] == 2
    summary = {"n": 20, "nli_mean": 0.96, "nli_below_080": 1, "numbers_failed": 0, "negations_failed": 0,
               "entities": 0.9}
    assert all(T.gates(summary, 0.9).values())
    assert not T.gates({**summary, "nli_below_080": 2}, 0.9)["nli_below_080_le_5pct"]
    assert not T.gates({**summary, "entities": 0.89}, 0.9)["entities_ge_rule_based"]
    assert all(T.over_edit_gate({"changed_token_rate": 0.1, "nli_below_080": 0}).values())
    assert not T.over_edit_gate({"changed_token_rate": 0.11, "nli_below_080": 0})["changed_token_rate_le_010"]


# --- panel builder ------------------------------------------------------------

def test_sentences_respect_abbreviations_and_initials():
    text = "The U.S. Government said so. Dr. Smith agreed with J. Doe on No. 5. Yes, it did."
    assert B.sentences(text) == ["The U.S. Government said so.", "Dr. Smith agreed with J. Doe on No. 5.",
                                 "Yes, it did."]


def test_window_filter():
    body = " ".join(["The agency reviewed every comment that it received during the public period."] * 6)
    assert B.acceptable(body, "fedreg")
    assert not B.acceptable(body.replace("agency", "agency (see www.example.gov)"), "fedreg")
    assert not B.acceptable(body[:-1], "fedreg")  # no sentence terminator
    assert not B.acceptable("Short text.", "fedreg")
    assert not B.acceptable(body.replace("reviewed", "re- viewed", 1), "fedreg")


def test_pick_windows_are_disjoint_and_seeded():
    candidates = [(0, i, j, "") for i in range(6) for j in range(i, min(i + 3, 6))]
    a = B.pick_windows(candidates, 2, random.Random(1))
    assert a == B.pick_windows(candidates, 2, random.Random(1))
    (b0, i0, j0, _), (b1, i1, j1, _) = a
    assert b0 != b1 or j0 < i1 or j1 < i0
    assert B.pick_windows(candidates[:1], 2, random.Random(1)) is None


def test_split_is_by_document_and_stratified():
    passages = [{"id": f"x{k}", "domain": d, "source_doc": f"{d}-{k // 2}"} for d in ("a", "b") for k in range(12)]
    plan = B.PLAN
    try:
        B.PLAN = {"a": None, "b": None}
        B.split(passages)
    finally:
        B.PLAN = plan
    for d in ("a", "b"):
        rows = [p for p in passages if p["domain"] == d]
        assert sum(p["split"] == "dev" for p in rows) == 4
        for doc in {p["source_doc"] for p in rows}:
            assert len({p["split"] for p in rows if p["source_doc"] == doc}) == 1


def test_committed_manifest_and_split_lists_agree():
    manifest = json.loads((B.DATA / "manifest.json").read_text())
    rows = manifest["passages"]
    assert len(rows) == 300 and len({r["id"] for r in rows}) == 300
    assert all("original" not in r and len(r["text_sha256"]) == 64 for r in rows)
    for split, n in (("dev", 100), ("test", 200)):
        ids = (B.DATA / f"{split}_ids.txt").read_text().split()
        assert len(ids) == n and set(ids) == {r["id"] for r in rows if r["split"] == split}
    for doc in {r["source_doc"] for r in rows}:
        assert len({r["split"] for r in rows if r["source_doc"] == doc}) == 1
    assert sum(r["kind"] == "simple_control" for r in rows) == 48


# --- test-split lock ----------------------------------------------------------

def test_freeze_and_run_refuse_locked_test_ids(tmp_path, monkeypatch):
    test_id = (B.DATA / "test_ids.txt").read_text().split()[0]
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps([{"id": test_id, "original": "Text."}]))
    assert E.locked_ids([{"id": test_id}, {"id": "cs-s01"}]) == [test_id]
    args = SimpleNamespace(state=tmp_path / "state", panel=[f"x={panel}"], unlock_test=False)
    with pytest.raises(SystemExit, match="locked"):
        E.freeze(args)
    with pytest.raises(ValueError, match="locked"):
        E.load_panel({"panels": {"x": str(panel)}}, "x")
    assert E.load_panel({"panels": {"x": str(panel)}, "unlock_test": True}, "x")[0]["id"] == test_id
