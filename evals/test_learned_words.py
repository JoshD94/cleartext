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


def test_r9_majority_takes_two_of_three_else_most_severe():
    from evals.learned_words_r9 import majority

    assert majority(["good", "good", "meaning"]) == "good"
    assert majority(["notsimpler", "meaning", "notsimpler"]) == "notsimpler"
    assert majority(["good", "notsimpler", "meaning"]) == "meaning"


def test_r9_clean_sentence_filter():
    from evals.learned_words_r9 import _ok_sentence

    assert _ok_sentence("The server MUST reject a request that does not carry a valid token.")
    assert not _ok_sentence("Table 3 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15.")
    assert not _ok_sentence("Too short to use.")


def test_r9_arm_w_filters_before_the_passage_limits():
    from evals.learned_words_apply import choose
    from evals.learned_words_r9 import picks

    def sw(i, sent, margin, stack, w):
        return {"id": i, "sentence": sent, "margin": margin, "stack": stack, "w": w, "gate": 1.0, "check": 1.0 - i / 10}

    swaps = [sw(0, 0, 0.5, 0.9, 0.1), sw(1, 0, 0.5, 0.9, 0.9), sw(2, 1, 0.05, 0.9, 0.9)]
    b, s, w = picks(swaps, 0.5, 0.5, choose)
    assert [x["id"] for x in b] == [0]
    assert [x["id"] for x in s] == [0]
    assert [x["id"] for x in w] == [1]


def test_r10_two_heads_multiply_survival_of_both_errors():
    from evals.learned_words_r9 import TwoHeads

    rng = np.random.RandomState(0)
    X = rng.normal(size=(120, 3))
    marks = np.array(["meaning" if x[0] > 0.8 else "notsimpler" if x[1] > 0.8 else "good" for x in X])
    h = TwoHeads(X, marks)
    p = h.predict_proba(X)[:, 1]
    expect = (1 - h.m.predict_proba(X)[:, 1]) * (1 - h.n.predict_proba(X)[:, 1])
    assert np.allclose(p, expect)
    assert p[marks == "good"].mean() > p[marks != "good"].mean()


def test_r12_priors_smooth_and_leave_one_out():
    from evals.learned_words_r12 import Priors

    pri = Priors([("multiple", "several")] * 3 + [("occur", "happen")], [0, 0, 0, 1], a=2.0)
    assert abs(pri.m - 0.25) < 1e-9
    pp, pt, n = pri.rates(("multiple", "several"))
    assert n == 3 and abs(pp - (0 + 0.5) / 5) < 1e-9 and abs(pt - pp) < 1e-9
    pp, _, n = pri.rates(("multiple", "several"), own=0)  # the row's own label is removed
    assert n == 2 and abs(pp - 0.5 / 4) < 1e-9
    pp, _, n = pri.rates(("unseen", "word"))
    assert n == 0 and abs(pp - 0.25) < 1e-9
    assert pri.seen(("occur", "happen")) and not pri.seen(("occur", "start"))


def test_r12_arm_p_learns_from_pair_history():
    from evals.learned_words_r12 import ArmP

    rows = []
    for i in range(60):
        bad = i % 3 == 0
        rows.append({"key": ("multiple", "several") if bad else (f"w{i}", "x"), "stack": 0.7, "cover": 0.0,
                     "good": int(not bad)})
    m = ArmP(rows)
    s = m.score([{"key": ("multiple", "several"), "stack": 0.7, "cover": 0.0, "good": 0},
                 {"key": ("new", "pair"), "stack": 0.7, "cover": 0.0, "good": 0}])
    assert s[0] < s[1]


def test_r12_arms_filter_before_the_passage_limits():
    from evals.learned_words_apply import choose
    from evals.learned_words_r12 import picks

    class M:
        def score(self, rows):
            return np.array([r["stack"] for r in rows])  # stand-in for arm p

    def sw(i, sent, stack, cover=0.0, margin=0.5):
        return {"id": i, "sentence": sent, "margin": margin, "stack": stack, "cover": cover, "gate": 1.0,
                "check": 1.0 - i / 10, "word": "w", "rep": f"r{i}", "pos": "v"}

    rec = {"all": [sw(0, 0, 0.55), sw(1, 0, 0.9), sw(2, 1, 0.9, margin=0.05)], "sense": [sw(3, 0, 0.6, 1.0)]}
    out = picks(rec, M(), {"s": 0.5, "t": 0.8, "p": 0.8, "p65": 0.6}, choose)
    assert [x["id"] for x in out["b"]] == [0]
    assert [x["id"] for x in out["s"]] == [0]
    assert [x["id"] for x in out["t"]] == [1]
    assert [x["id"] for x in out["p"]] == [1]
    assert [x["id"] for x in out["gs"]] == [3]


def test_r13_fallback_uses_runner_up_only_for_targets_without_an_accepted_top():
    from evals.learned_words_apply import choose
    from evals.learned_words_r13 import fallback, picks

    def sw(tid, sent, p, rep, margin=0.5):
        return {"target_id": tid, "sentence": sent, "margin": margin, "stack": p, "cover": 0.0, "gate": 1.0,
                "check": p, "word": "w", "rep": rep, "pos": "v", "p": p}

    top = [sw("t1", 0, 0.9, "a"), sw("t2", 1, 0.3, "b")]
    run = [sw("t1", 0, 0.95, "c"), sw("t2", 1, 0.8, "d"), sw("t3", 2, 0.9, "e")]
    got = fallback(top, run, lambda s: s["p"] >= 0.5)
    assert [s["rep"] for s in got] == ["a", "d", "e"]

    class M:
        def score(self, rows):
            return np.array([r["stack"] for r in rows])

    rec = {"all": top, "next": run, "sense": []}
    out = picks(rec, M(), {"s": 0.5, "t": 0.5, "p": 0.5, "p65": 0.2}, choose)
    assert [s["rep"] for s in out["p"]] == ["a"]
    assert sorted(s["rep"] for s in out["pn"]) == ["a", "d", "e"]
    assert sorted(s["rep"] for s in out["p65n"]) == ["a", "b", "e"]


def test_r14_arm_m_adds_meaning_signals_to_arm_p():
    from evals import learned_words_stacker as S
    from evals.learned_words_r14 import ArmM

    rng = np.random.RandomState(1)
    rows = []
    for i in range(80):
        x = [0.0] * len(S.KEYS)
        good = i % 2
        x[S.KEYS.index("ctx_cos")] = 0.9 if good else 0.5
        rows.append({"key": (f"t{i}", "r"), "stack": 0.7, "cover": 0.0, "x31": x, "good": good})
    m = ArmM(rows)
    hi, lo = [dict(rows[1]), dict(rows[0])]
    hi["key"], lo["key"] = ("new", "a"), ("new", "b")
    s = m.score([hi, lo])
    assert s[0] > s[1]
    assert m.clf[-1].coef_.shape[1] == 6


def test_r14_arms_filter_before_the_passage_limits():
    from evals.learned_words_apply import choose
    from evals.learned_words_r14 import picks

    class M:
        def score(self, rows):
            return np.array([r["stack"] for r in rows])

    def sw(i, sent, stack, margin=0.5):
        return {"sentence": sent, "margin": margin, "stack": stack, "cover": 0.0, "gate": 1.0, "check": 1.0 - i / 10,
                "word": "w", "rep": f"r{i}", "pos": "v", "x31": [0.0] * 31}

    rec = {"all": [sw(0, 0, 0.55), sw(1, 0, 0.9)], "sense": []}
    out = picks(rec, M(), M(), {"s": 0.5, "p": 0.8, "m": 0.8, "mw": 0.5}, choose)
    assert [x["rep"] for x in out["b"]] == ["r0"]
    assert [x["rep"] for x in out["m"]] == ["r1"]
    assert [x["rep"] for x in out["mw"]] == ["r0"]


def test_r15_cutoff_keeps_the_set_share_of_good_technical_rows():
    from evals.learned_words_r15 import GOOD_KEPT, cutoff

    rows = [{"drop": float(i), "tech": True, "good": 1} for i in range(11)]
    rows += [{"drop": 100.0, "tech": False, "good": 1}, {"drop": 50.0, "tech": True, "good": 0}]
    assert abs(cutoff(rows) - 10 * GOOD_KEPT) < 1e-9


def test_r15_veto_applies_after_arm_p_and_before_the_passage_limits():
    from evals.learned_words_apply import choose
    from evals import learned_words_r15 as V

    class M:
        def score(self, rows):
            return np.array([r["stack"] for r in rows])

    def sw(i, sent, stack):
        return {"pid": "x", "sentence": sent, "margin": 0.5, "stack": stack, "cover": 0.0, "gate": 1.0,
                "check": 1.0 - i / 10, "word": "administered", "rep": f"r{i}", "pos": "v",
                "before": "It is administered orally.", "start": 6}

    orig = V.max_drop
    V.max_drop = lambda before, word, rep, start, own=None: {"r0": 5.0, "r1": 1.0}[rep]
    try:
        out = V.picks({"all": [sw(0, 0, 0.9), sw(1, 1, 0.8)], "sense": []}, M(), M(),
                      {"s": 0.5, "p": 0.5, "p40": 0.85, "pv": 0.7, "cutoff": 2.0}, choose, {})
    finally:
        V.max_drop = orig
    assert [x["rep"] for x in out["p40"]] == ["r0"]
    assert [x["rep"] for x in out["pv"]] == ["r1"]


def test_r16_reuses_round_15_veto_and_arms():
    from evals import learned_words_r15 as V15
    from evals import learned_words_r16 as V16

    assert V16.ARMS == V15.ARMS and V16.KEEP == V15.KEEP and V16.GOOD_KEPT == V15.GOOD_KEPT
    assert V16.max_drop is V15.max_drop and V16.TECH == ("r11", "r12", "r13")
    rows = [{"drop": float(i), "tech": True, "good": 1} for i in range(11)]
    assert V16.cutoff(rows) == V15.cutoff(rows)


def test_r17_refitted_arm_p_filters_before_the_passage_limits():
    from evals.learned_words_apply import choose
    from evals.learned_words_r17 import picks

    class M:
        def score(self, rows):
            return np.array([r["stack"] for r in rows])

    def sw(i, sent, stack):
        return {"sentence": sent, "margin": 0.5, "stack": stack, "cover": 0.0, "gate": 1.0, "check": 1.0 - i / 10,
                "word": "w", "rep": f"r{i}", "pos": "v"}

    out = picks({"all": [sw(0, 0, 0.6), sw(1, 0, 0.9)], "sense": []}, M(), M(),
                {"s": 0.5, "p": 0.5, "p40": 0.8, "p50": 0.5}, choose)
    assert [x["rep"] for x in out["p40"]] == ["r1"]
    assert [x["rep"] for x in out["p50"]] == ["r0"]


def test_r18_replicates_round_17_arms_and_excludes_its_panel():
    import inspect

    from evals import learned_words_r17 as V17
    from evals import learned_words_r18 as V18

    assert V18.ARMS == V17.ARMS and V18.SEED != V17.SEED and V18.PER_SOURCE == V17.PER_SOURCE
    assert "P17.PANEL" in inspect.getsource(V18.panel)
    assert inspect.getsource(V18.picks).replace("V18", "") == inspect.getsource(V17.picks).replace("V17", "")
