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


def test_oof_scores_extra_rows_with_the_fold_that_held_out_their_group():
    from cleartext.learned_words.decide import oof
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(0)
    X = rng.rand(200, 2)
    y = (X[:, 0] > 0.5).astype(float)
    g = np.repeat(np.arange(20), 10)
    Xe = rng.rand(6, 2)
    ge = np.array([0, 3, 7, 11, 19, 99])  # 99 never occurs: scored by a model on all rows
    pred, pe = oof(lambda: LogisticRegression(), X, y, g, extra=(Xe, ge))
    assert pred.shape == (200,) and pe.shape == (6,) and not np.isnan(pe).any()


def test_simpler_margin_rule_takes_smallest_margin_at_target_precision():
    from cleartext.learned_words import simpler as SM

    margins = np.repeat([0.0, 0.5, 1.0], 200)
    labels = np.concatenate([np.zeros(200), np.r_[np.ones(120), np.zeros(80)], np.ones(200)]).astype(bool)
    # >= 0.5: 320/400 = 0.80 reaches the target; >= 0.0 does not.
    m, table = SM.choose_margin(margins, labels, grid=np.array([0.0, 0.5, 1.0]))
    assert m == 0.5 and table[0]["precision"] < 0.8


def test_simpler_pairs_label_needs_clear_gap():
    from cleartext.learned_words import simpler as SM

    scores = {"a": 0.9, "b": 0.85, "c": 0.1}
    pred = {"a": 2.0, "b": 1.0, "c": 0.0}
    m, lab, gap = SM.pairs(scores, pred, 0.10, max_pairs=2000)
    clear = dict(zip(np.round(gap, 2), lab))
    assert clear[0.8] and not clear[0.05] and not clear[-0.8]


def test_r3_choose_keeps_one_swap_per_sentence_and_respects_filter():
    from evals.learned_words_apply import choose

    sw = [{"sentence": 0, "gate": 0.9, "check": 0.9, "margin": 0.1},
          {"sentence": 0, "gate": 0.8, "check": 0.9, "margin": 1.0},
          {"sentence": 1, "gate": 0.5, "check": 0.9, "margin": 1.0}]
    assert choose(sw) == [sw[0], sw[2]]
    assert choose(sw, lambda s: s["margin"] >= 0.5) == [sw[1], sw[2]]


def test_judge_pair_marks_target_and_replacement_in_same_window():
    from cleartext.learned_words.judge import pair

    text = "We are required to submit the form before noon."
    s = text.index("required")
    a, b = pair(text, s, s + len("required"), "needed", size=2)
    assert a == "We are [required] to submit"
    assert b == "We are [needed] to submit"


def test_r5_arms_stage_and_judge_picks():
    from evals.learned_words_r5 import arms, swaps_for

    pre = {"margin": 0.1, "judge_t": 0.6, "strict_t": None}
    t = {"pid": "p", "sentence": 0, "before": "We are required to go.", "word": "required", "start": 7, "end": 15,
         "pos": "v", "eligible": True, "stage": {"rep": "needed", "swap": True, "gate": 0.9, "check": 0.95},
         "cands": [{"rep": "needed", "judge": 0.3, "margin": 0.5},
                   {"rep": "asked", "judge": 0.8, "margin": 0.05},
                   {"rep": "told", "judge": 0.7, "margin": 0.4}]}
    rec = {"targets": [t]}
    a = arms(pre)
    assert set(a) == {"b", "e", "f"}
    assert [s["rep"] for s in swaps_for(rec, *a["b"], 0.1)] == ["needed"]
    assert swaps_for(rec, *a["e"], 0.1) == []  # judge rejects the stage's word
    f = swaps_for(rec, *a["f"], 0.1)  # "asked" fails the margin; "told" is the judge's pick
    assert [s["rep"] for s in f] == ["told"] and f[0]["after"] == "We are told to go."


def test_pairwise_relations_flag_families_and_edits():
    from cleartext.learned_words.pairwise import _edit, _family

    assert _edit("upward", "upwards") == 1 and _edit("intention", "intent") == 3
    assert _family("intention", "intend") == 1.0
    assert _family("halt", "stop") == 0.0


def test_corpus_tokens_lowercase_and_keep_hyphens():
    from evals.learned_words_corpus import tokens

    assert tokens("The FDA’s well-known Rule, 2024.") == ["the", "fda's", "well-known", "rule"]


def test_rewrite_align_marks_kept_replaced_and_dropped():
    from cleartext.learned_words.rewrite import DROPPED, KEPT, REPLACED, align

    a = "the committee will commence the lengthy review".split()
    b = "the committee will start the review".split()
    got = {a[i]: (st, r) for i, st, r in align(a, b)}
    assert got["committee"] == (KEPT, None)
    assert got["commence"] == (REPLACED, "start")
    assert got["lengthy"] == (DROPPED, None)


def test_rewrite_table_subtracts_own_fold():
    from cleartext.learned_words.rewrite import KEPT, REPLACED, WIKI, Table, _kp, _kw

    w = np.array([1, 1, 1, 2])
    r = np.array([3, 3, 0, 0])
    status = np.array([REPLACED, REPLACED, KEPT, KEPT], np.int8)
    fold = np.array([0, 1, 1, 2], np.int8)
    dom = np.zeros(4, np.int8)
    T = Table.build(w, np.zeros(4, np.int32), np.zeros(4, np.int32), status, r, fold, dom)
    key = _kp(np.array([1]), np.array([3]), WIKI)
    assert T.get("pair", key)[0] == 2
    assert T.get("pair", key, np.array([0]))[0] == 1
    assert T.get("pair", key, np.array([-1]))[0] == 2
    assert T.get("word", _kw(np.array([1]), WIKI, KEPT), np.array([1]))[0] == 0


def test_stacker_span_finds_the_replaced_occurrence():
    from evals.learned_words_stacker import span

    before = "use the form and use the tool"
    after = "use the form and employ the tool"
    assert span(before, after, "use", "employ") == (17, 20)
    assert span(before, before, "absent", "x") is None


def test_stacker_arm_s_filters_before_the_passage_limits():
    from evals.learned_words_apply import choose
    from evals.learned_words_stacker import arm_picks

    swaps = [{"sentence": 0, "gate": 0.9, "check": 0.9, "margin": 0.5, "stack": 0.2, "word": "a"},
             {"sentence": 0, "gate": 0.5, "check": 0.5, "margin": 0.5, "stack": 0.8, "word": "b"},
             {"sentence": 1, "gate": 0.9, "check": 0.9, "margin": 0.05, "stack": 0.9, "word": "c"}]
    b, s = arm_picks(swaps, 0.5, choose)
    assert [x["word"] for x in b] == ["a"]
    assert [x["word"] for x in s] == ["b"]
