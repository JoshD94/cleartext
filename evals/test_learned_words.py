"""Contract checks for the learned word stage (no BERT, no downloaded data)."""

import random

import numpy as np

from cleartext.learned_words import splits as SP
from cleartext.learned_words.decide import gate_rows, generated
from cleartext.learned_words.features import window
from evals import learned_words_study as S


def test_window_keeps_same_context_both_sides():
    text = "The router measured the latency, which was high on every link today."
    s = text.index("latency")
    a, b = window(text, s, s + len("latency"), "delay", size=3)
    # Text after the target is split on whitespace, so attached punctuation
    # counts as one of the window's words on both sides alike.
    assert a == "router measured the latency , which was"
    assert b == "router measured the delay , which was"


def test_generated_drops_given_only_rows():
    rows = [{"sources": ["given"]}, {"sources": ["bert", "given"]}, {"sources": ["wordnet"]}]
    assert generated(rows) == rows[1:]


def _row(iid, word, gold, **f):
    base = {"d_zipf": 1.0, "ctx_cos": 0.9, "t_zipf": 3.0, "t_senses": 2, "pos_n": 1.0, "pos_v": 0.0,
            "pos_a": 0.0, "pos_r": 0.0, "t_has_compound": 0.0, "t_compound": 0.0}
    base.update(f)
    return {"id": iid, "word": word, "gold": gold, "group": iid, "f": base, "key": (iid, word)}


def test_gate_rows_take_top_candidate_and_margin():
    rows = [_row("a", "x", 0), _row("a", "y", 2), _row("b", "z", 1)]
    g = gate_rows(rows, np.array([0.2, 0.7, 0.4]), {"a": {"p_complex": 0.9, "p_term": 0.1}})
    by = {r["id"]: r for r in g}
    assert by["a"]["top"]["word"] == "y" and by["a"]["label"] == 1.0
    assert abs(by["a"]["g"]["margin"] - 0.5) < 1e-9
    assert by["b"]["g"]["p_term"] == 0.5  # missing stage-1 scores fall back to 0.5


def test_threshold_rule_prefers_most_swaps_at_target_precision():
    rng = np.random.RandomState(0)
    n = 400
    label = rng.rand(n) < 0.5
    gate = np.clip(label * 0.6 + rng.rand(n) * 0.5, 0, 1)
    check = np.ones(n)
    grows = [{"label": float(x)} for x in label]
    best, grid = S.choose_thresholds(grows, gate, check)
    assert best["precision"] >= S.TARGET_PRECISION and best["swaps"] >= S.MIN_SWAPS
    better = [g for g in grid if g["precision"] is not None and g["precision"] >= 0.9 and g["swaps"] > best["swaps"]
              and g["swaps"] >= S.MIN_SWAPS]
    assert not better


def test_threshold_rule_falls_back_when_unreachable():
    grows = [{"label": float(i % 2)} for i in range(100)]
    best, _ = S.choose_thresholds(grows, np.full(100, 0.5), np.full(100, 0.5))
    assert best["rule"].startswith("fallback")


def test_lsbert_scores_rank_only_bert_candidates():
    rows = [dict(_row("a", w, 0, mask_logp=m, d_pll=0.0, c_zipf=4.0), sources=s)
            for w, m, s in [("p", -1.0, ["bert"]), ("q", -0.5, ["wordnet"]), ("r", -3.0, ["bert"])]]
    s = S.lsbert_scores(rows)
    assert s[1] == -1.0 and s[0] > s[2]


def test_precision_counts_only_swaps():
    inst = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    dec = {"a": {"swap": True, "gold": 2}, "b": {"swap": True, "gold": 0}, "c": {"swap": False, "gold": 5}}
    p = S.precision(dec, inst)
    assert (p["swaps"], p["correct"], p["precision"]) == (2, 1, 0.5)
    assert S.precision(dec, inst, 2)["correct"] == 1


def test_wikisplit_detok_and_corruptions_are_labelled():
    assert SP.detok("It was released in 2011 . <::::> It 's good .") == "It was released in 2011. It's good."
    row = {"source": "The company opened a new plant in Ohio, and it hired two hundred workers there last year.",
           "parts": ["The company opened a new plant in Ohio.", "It hired two hundred workers there last year."]}
    rng = random.Random(1)
    assert SP.corrupt(row, "wrong_referent", rng)[1].startswith("They hired")
    lost = SP.corrupt(row, "lost_subject", rng)
    assert lost and not lost[1].lower().startswith("it ")
    pairs = SP.build_pairs([row], per_source=2)
    assert [p["label"] for p in pairs].count(1) == 1 and all(p["kind"] in SP.KINDS for p in pairs if p["label"] == 0)
