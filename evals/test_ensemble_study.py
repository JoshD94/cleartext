"""Ensemble-study design, selection, storage and resume, without native models."""

import argparse
import json

import pytest

from evals import ensemble_study as S
from evals.preservation import align_sentences, sentence_nli
from evals.sari import corpus_sari


def test_ofat_design_covers_every_single_change_once():
    rows = S.ofat()
    assert rows[0] == S.BASELINE
    assert len(rows) == 1 + sum(len(v) - 1 for v in S.FACTORS.values())
    assert len({S.config_id(r) for r in rows}) == len(rows)
    for row in rows[1:]:
        assert sum(row[k] != S.BASELINE[k] for k in S.FACTORS) == 1


def test_factorial_and_ids_round_trip():
    rows = S.factorial({"words": ["v7", "contextual"], "negation": ["off", "on"]})
    assert len(rows) == 4
    for row in rows:
        assert S.parse_id(S.config_id(row)) == row
    with pytest.raises(ValueError):
        S.normalize({"words": "llm"})
    with pytest.raises(ValueError):
        S.normalize({"passive": "on"})


def test_adding_a_later_factor_keeps_existing_ids(monkeypatch):
    old = S.config_id(S.BASELINE)
    monkeypatch.setattr(S, "FACTORS", {**S.FACTORS, "passive": ("off", "on")})
    monkeypatch.setattr(S, "LEGACY", {"passive": "off"})
    assert S.config_id(S.BASELINE) == old
    assert S.config_id({**S.BASELINE, "passive": "on"}) == old + "|passive=on"
    assert S.parse_id(old)["passive"] == "off"
    assert len(S.ofat()) == 1 + sum(len(v) - 1 for v in S.FACTORS.values())


def summary(fk, nli=0.97, below=0, numbers=True, negations=True, entities=0.99, n=90):
    return {"n": n, "fk_grade": fk, "nli_mean": nli, "nli_below_080": below,
            "sent_nli_mean": nli, "sent_nli_below_080": below, "numbers_all": numbers,
            "negations_all": negations, "entities": entities}


def test_gates_follow_the_frozen_rule():
    assert all(S.gates(summary(13), 0.99).values())
    assert not S.gates(summary(13, nli=0.949), 0.99)["nli_mean_ge_095"]
    assert S.gates(summary(13, below=4), 0.99)["nli_below_080_le_5pct"]
    assert not S.gates(summary(13, below=5), 0.99)["nli_below_080_le_5pct"]
    assert not S.gates(summary(13, negations=False), 0.99)["numbers_negations_all"]
    assert not S.gates(summary(13, entities=0.98), 0.99)["entities_ge_rule_based"]


def test_select_prefers_lowest_grade_and_breaks_ties_by_sari():
    rows = {"original": summary(14), "rule_based": summary(13),
            "a": summary(13.00), "b": summary(13.05), "c": summary(12.0, nli=0.90), "d": summary(13.5)}
    picked = S.select(rows, 0.99)
    assert picked["winner"] == "a" and picked["tied"] == ["a", "b"]
    assert picked["tie_unresolved_missing_sari"]
    assert S.select(rows, 0.99, sari={"a": 30.0, "b": 31.0})["winner"] == "b"
    assert S.select({"x": summary(12, nli=0.5)}, 0.99)["winner"] is None


def test_pareto_frontier():
    rows = {"a": summary(13, 0.99), "b": summary(12, 0.90), "c": summary(13.5, 0.95), "d": summary(12, 0.95)}
    assert S.pareto(rows) == ["d", "a"]


def test_summarize_counts_abstentions_as_low_nli():
    passages = [{"id": str(k), "original": f"Text {k}."} for k in range(3)]
    outputs = {"0": {"output": "Text 0.", "stages": []}, "1": {"output": "Txt 1.", "stages": ["word"]},
               "2": {"output": "Txt 2.", "stages": ["word", "phrase"]}}

    def metric(nli, negation=1.0):
        return {"readability": {"flesch_kincaid_grade": 5.0, "flesch_reading_ease": 60.0,
                                "avg_sentence_length": 2.0},
                "preservation": {"nli_mean": nli, "embedding_similarity": 1.0, "entities_preserved": 1.0,
                                 "numbers_preserved": 1.0, "negations_preserved": negation,
                                 "tenses_preserved": 1.0, "tfidf_cosine_similarity": 1.0},
                "sentence_nli": {"sent_nli_mean": nli, "sent_nli_changed_min": nli}}

    out = S.summarize(passages, outputs, {"0": metric(0.99), "1": metric(None), "2": metric(0.5, 0.0)})
    assert out["changed"] == 2 and out["nli_abstained"] == 1 and out["nli_below_080"] == 2
    assert out["nli_mean"] == pytest.approx(0.745) and out["negations_failed"] == 1
    assert out["stage_passages"] == {"word": 2, "phrase": 1}


def test_store_identity_and_metric_pairs(tmp_path):
    store = S.StudyStore(tmp_path, "g" * 64, "s" * 64)
    passage = {"id": "p", "original": "Alice waited."}
    assert store.output(passage, "original") is None
    store.save_output(passage, "original", {"output": "Alice waited.", "stages": []})
    assert store.output(passage, "original")["output"] == "Alice waited."
    with pytest.raises(ValueError, match="identity"):
        S.StudyStore(tmp_path, "g" * 64, "s" * 64).output({**passage, "original": "Bob."}, "original")
    store.save_metrics("Alice waited.", "Alice waited.", {"x": 1})
    assert S.StudyStore(tmp_path, "h" * 64, "s" * 64).metrics("Alice waited.", "Alice waited.")["x"] == 1
    with pytest.raises(ValueError, match="Metric cache identity"):
        S.StudyStore(tmp_path, "g" * 64, "s" * 16 + "t" * 48).metrics("Alice waited.", "Alice waited.")


def fake_plan(tmp_path, passages):
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps(passages))
    plan = tmp_path / "state" / "plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"panels": {"sourced": str(panel)}, "panel_sha256": {}}))
    return plan


def test_run_resumes_without_regenerating(tmp_path, monkeypatch):
    passages = [{"id": f"p{k}", "original": f"Hard words {k}."} for k in range(3)]
    plan = fake_plan(tmp_path, passages)
    design = tmp_path / "design.json"
    design.write_text(json.dumps({"configs": [S.BASELINE, {**S.BASELINE, "words": "off"}]}))
    calls, scored = [], []

    class Generator:
        def __init__(self, *args):
            pass

        def begin(self, passage):
            pass

        def end(self, passage):
            pass

        def predict(self, name, text):
            calls.append((name, text))
            return {"output": text if name != S.config_id(S.BASELINE) else "Easy " + text, "stages": []}

    class Scorer:
        def __init__(self, *args):
            pass

        def __call__(self, source, output):
            scored.append((source, output))
            return {"ok": True}

    monkeypatch.setattr(S, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(S, "signatures", lambda plan: ("g" * 64, "s" * 64))
    monkeypatch.setattr(S, "StudyGenerator", Generator)
    monkeypatch.setattr(S, "StudyScorer", Scorer)
    args = argparse.Namespace(plan=plan, state=plan.parent, design=design, panel="sourced", shard="0/1",
                              threads=1, max_seconds=1000, score=True, preflight=False)
    S.run(args)
    assert len(calls) == 3 * 4
    # rule_based is a dictionary rewrite; original and words=off share one exact pair.
    assert len(scored) == len(set(scored)) and len(scored) <= 3 * 3
    S.run(args)
    assert len(calls) == 3 * 4 and len(set(scored)) == len(scored)
    receipts = sorted((plan.parent / "batches").iterdir())
    assert json.loads(receipts[-1].read_text())["shard_complete"]


def test_collect_refuses_conflicting_states(tmp_path, monkeypatch):
    passage = {"id": "p", "original": "Alice waited."}
    plan = fake_plan(tmp_path, [passage])
    other = tmp_path / "other"
    other.mkdir()
    (other / "plan.json").write_text(plan.read_text())
    monkeypatch.setattr(S, "signatures", lambda plan: ("g" * 64, "s" * 64))
    S.StudyStore(plan.parent, "g" * 64, "s" * 64).save_output(passage, "original", {"output": "A."})
    S.StudyStore(other, "g" * 64, "s" * 64).save_output(passage, "original", {"output": "B."})
    with pytest.raises(ValueError, match="Conflicting"):
        S.collect([plan.parent, other], json.loads(plan.read_text()), "sourced", ["original"], need_metrics=False)


def test_sentence_alignment_handles_edits_splits_and_drops():
    assert align_sentences("A dog ran. The authenticating server failed.",
                           "A dog ran. The certifying server failed.") == [
        ("A dog ran.", "A dog ran."), ("The authenticating server failed.", "The certifying server failed.")]
    assert align_sentences("It was late, and we left. Then we slept.",
                           "It was late. We left. Then we slept.") == [
        ("It was late, and we left.", "It was late. We left."), ("Then we slept.", "Then we slept.")]
    assert align_sentences("One here. Two there. Three.", "One here. Other words now. Three.")[1] == (
        "Two there.", "Other words now.")
    assert align_sentences("One here. Two there.", "One here.")[1] == ("Two there.", "")


def test_sentence_nli_scores_groups_and_drops():
    entail = lambda p, h: 1.0 if p == h else 0.4
    report = sentence_nli("A dog ran. Cats sleep.", "A dog ran. Cats nap.", entail=entail)
    assert report["sent_nli_mean"] == pytest.approx(0.7) and report["sent_nli_changed_min"] == 0.4
    dropped = sentence_nli("A dog ran. Cats sleep.", "A dog ran.", entail=entail)
    assert dropped["sent_nli_min"] == 0.0 and dropped["sent_nli_groups"] == 2


def test_sari_matches_reference_implementation():
    # Hugging Face `evaluate` sari documentation example.
    value = corpus_sari(["About 95 species are currently accepted."], ["About 95 you now get in."],
                        [["About 95 species are currently known.", "About 95 species are now accepted.",
                          "95 species are now accepted."]])
    assert value == pytest.approx(26.953601953601954)
    assert corpus_sari(["a b c."], ["a b c."], [["a b c."]]) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        corpus_sari(["a"], ["a", "b"], [["a"]])


def test_versus_baseline_flags_active_levels():
    base, other, quiet = S.config_id(S.BASELINE), S.config_id({**S.BASELINE, "words": "off"}), \
        S.config_id({**S.BASELINE, "grammar": "on"})
    passages = [{"id": str(k), "original": "x"} for k in range(4)]
    outs = lambda text: {p["id"]: {"output": text} for p in passages}
    complete = {base: (outs("a"), {}), other: (outs("b"), {}), quiet: (outs("a"), {})}
    summaries = {base: summary(13.0), other: summary(13.01), quiet: summary(13.0)}
    result = S.versus_baseline(passages, complete, summaries)
    assert result[other]["active"] and result[other]["outputs_differ"] == 4
    assert not result[quiet]["active"] and result[quiet]["fk_grade"] == 0


def test_noninitial_entities_ignore_sentence_starts():
    src = "However, Alice met Bob. Similarly the cat sat."
    assert S.noninitial_entities_preserved(src, "But, Alice met Bob. Also the cat sat.") == 1.0
    assert S.noninitial_entities_preserved(src, "However, she met Bob. Similarly the cat sat.") == 0.5
    assert S.noninitial_entities_preserved("No names here.", "None.") == 1.0
