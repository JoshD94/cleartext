"""Ensemble-study design, selection, storage and resume, without native models."""

import argparse
import json

import pytest

from evals import ensemble_study as S
from evals.preservation import align_sentences, sentence_nli
from evals.sari import corpus_sari


def test_ofat_design_covers_every_single_change_once():
    rows = S.ofat()
    assert rows[0] == S.normalize(S.BASELINE)
    assert len(rows) == 1 + sum(len(v) - 1 for v in S.FACTORS.values())
    assert len({S.config_id(r) for r in rows}) == len(rows)
    for row in rows[1:]:
        assert sum(row[k] != S.normalize(S.BASELINE)[k] for k in S.FACTORS) == 1


def test_factorial_and_ids_round_trip():
    rows = S.factorial({"words": ["v7", "contextual"], "negation": ["off", "on"]})
    assert len(rows) == 4
    for row in rows:
        assert S.parse_id(S.config_id(row)) == row
    with pytest.raises(ValueError):
        S.normalize({"words": "llm"})
    with pytest.raises(ValueError):
        S.normalize({"passive": "maybe"})


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


def test_sari_zero_division_convention_penalizes_copying():
    src, refs = ["The feline sat on the mat."], [["The cat sat on the mat.", "A cat sat on the mat."]]
    copy_hf = corpus_sari(src, src, refs, zero_division=1)
    copy_orig = corpus_sari(src, src, refs, zero_division=0)
    edit_orig = corpus_sari(src, ["The cat sat on the mat."], refs, zero_division=0)
    assert copy_orig < copy_hf and edit_orig > copy_orig


def test_recommended_config_settings(tmp_path):
    chosen = S.settings(S.ROOT / "evals/reports/20261109-ensemble-study/recommended-config.json")
    assert chosen["analyze"] == {"structure": True, "phrases": True, "words": False,
                                 "negation": False, "guardrails": True}
    assert chosen["t5"].endswith("seq2seq") and chosen["run"].endswith("ensemble-v7-20260928")
    assert chosen["passive"] == "off" and chosen["phrase_finder"] is None and not chosen["word_veto"]
    v2 = tmp_path / "v2.json"
    v2.write_text(json.dumps({"factors": {"words": "off", "phrases": "authored+crf", "structure": "t5",
                                          "passive": "agentive"},
                              "structure_checkpoint": {"path": "~/t5"}}))
    chosen = S.settings(v2)
    assert chosen["passive"] is None and chosen["phrase_finder"] == ["manual", "verified", "model"]
    assert chosen["analyze"]["phrases"] and chosen["compound_tagger"] is None
    v2.write_text(json.dumps({"factors": {"words": "off", "structure": "t5+compound_tagger"},
                              "structure_checkpoint": {"path": "~/t5"}}))
    with pytest.raises(ValueError, match="compound_tagger.path"):
        S.settings(v2)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"factors": {"structure": "t5"}}))
    with pytest.raises(ValueError, match="structure_checkpoint"):
        S.settings(bad)


def test_passive_and_new_levels_keep_nov9_ids():
    winner = {"words": "off", "phrases": "authored", "structure": "t5"}
    assert S.config_id(winner) == ("words=off|phrases=authored|structure=t5|negation=off|"
                                   "grammar=off|terms=off")
    assert S.config_id({**winner, "passive": "on"}).endswith("|passive=on")
    assert S.parse_id(S.config_id({**winner, "phrases": "authored+crf"}))["phrases"] == "authored+crf"


def test_paired_delta_bootstrap():
    out = S.paired_delta([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0], resamples=2000)
    assert out["mean"] == pytest.approx(1.5) and out["passages_nonzero"] == 3
    assert out["ci95"][0] < 1.5 < out["ci95"][1]
    assert S.paired_delta([1.0, None], [0.5, 0.5])["n"] == 1
    zero = S.paired_delta([2.0, 2.0], [2.0, 2.0])
    assert zero["ci95"] == [0.0, 0.0] and zero["passages_nonzero"] == 0


def test_memo_fallback_reads_old_state_and_writes_new(tmp_path):
    old = S.DiskMemo(tmp_path / "old", "t5", "o" * 64)
    old["A b."] = {"output": "A b.", "raw": "A. B."}
    memo = S.DiskMemo(tmp_path / "new", "t5", "n" * 64, (tmp_path / "old", "o" * 64))
    assert "A b." in memo and memo["A b."]["raw"] == "A. B."
    memo["C d."] = {"output": "C d.", "raw": "C d."}
    assert "C d." not in old and S.DiskMemo(tmp_path / "new", "t5", "n" * 64)["C d."]["raw"] == "C d."


def test_state_signatures_come_from_batch_receipts(tmp_path):
    (tmp_path / "batches").mkdir()
    for k in range(2):
        (tmp_path / "batches" / f"{k}.json").write_text(json.dumps(
            {"generation_signature": "g" * 64, "scoring_signature": "s" * 64}))
    assert S.state_signatures(tmp_path) == ("g" * 64, "s" * 64)
    (tmp_path / "batches" / "2.json").write_text(json.dumps(
        {"generation_signature": "h" * 64, "scoring_signature": "s" * 64}))
    with pytest.raises(ValueError, match="signature pairs"):
        S.state_signatures(tmp_path)


def test_ast_check_ignores_only_analyze():
    base = "class EnsembleClearText:\n    def analyze(self):\n        return 1\n    def decide(self):\n        return 2\n"
    assert S._without_analyze(base.encode()) == S._without_analyze(base.replace("return 1", "return 3").encode())
    assert S._without_analyze(base.encode()) != S._without_analyze(base.replace("return 2", "return 3").encode())
