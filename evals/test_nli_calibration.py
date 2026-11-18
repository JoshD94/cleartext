"""Offline checks for evals/nli_calibration.py (no models, no state)."""

from evals import nli_calibration as N


def test_twin_name_drops_words_and_word_params():
    name = ('words=contextual_veto|phrases=authored|structure=t5+compound_tagger|negation=off'
            '#{"w.veto":0.97,"w.veto_refill":false}')
    assert N.twin_name(name) == "words=off|phrases=authored|structure=t5+compound_tagger|negation=off"
    assert N.twin_name('words=v7|structure=t5#{"t5.beams":4,"w.floor":0.3}') == 'words=off|structure=t5#{"t5.beams":4}'


def test_single_swap_finds_one_replaced_run():
    before = "Monitor patients. Hematologic SPMs have been observed. More text."
    after = "Monitor patients. Hematologic SPMs have been found. More text."
    (a0, a1), (b0, b1) = N.single_swap(before, after)
    assert before[a0:a1] == "observed" and after[b0:b1] == "found"
    assert N.single_swap(before, before) is None
    assert N.single_swap("a b c d", "x b c y") is None  # two edits


def test_sentence_and_highlight():
    text = "First one here. The cat sat down. Last."
    start, end = N._sentence_at(text, text.index("cat"))
    assert text[start:end].strip() == "The cat sat down."
    assert N.highlight("The dog sat.", "The cat sat.") == "The <mark>dog</mark> sat."


def test_band_edges():
    assert N.band(0.95) == "≥0.95" and N.band(1.0) == "≥0.95"
    assert N.band(0.9499) == "0.90–0.95" and N.band(0.8) == "0.80–0.90" and N.band(0.0) == "<0.80"


def test_gate_bounds():
    row = {"nli_mean": 0.93, "numbers": 1.0, "negations": 1.0, "entities": 1.0}
    rows = [dict(row) for _ in range(20)]
    assert not N.gate(rows, 1.0, N.GATES["current (≥0.95; ≤5% <0.80)"])
    assert N.gate(rows, 1.0, N.GATES["mean ≥0.90; ≤5% <0.80"])
    rows[0]["nli_mean"], rows[1]["nli_mean"] = 0.75, 0.75  # 10% below 0.80
    assert not N.gate(rows, 1.0, N.GATES["mean ≥0.90; ≤5% <0.80"])
    assert N.gate(rows, 1.0, N.GATES["mean ≥0.90; ≤10% <0.80"])
    rows[2]["negations"] = 0.0
    assert not N.gate(rows, 1.0, N.GATES["mean ≥0.90; ≤10% <0.80"])


def test_rates_auc_wilson():
    r = N.rates([0.5, 0.96, 0.99, 0.85], thresholds=(0.9,))
    assert r["0.9"]["k"] == 2 and r["0.9"]["rate"] == 0.5
    assert N.auc([0.9, 0.8], [0.1, 0.2]) == 1.0 and N.auc([0.5], [0.5]) == 0.5
    lo, hi = N.wilson(5, 10)
    assert lo < 0.5 < hi


def test_rates_clustered_ci_and_marks_key(tmp_path):
    r = N.rates([0.5, 0.6, 0.99, 0.99], thresholds=(0.9,), groups=["a", "a", "b", "b"], resamples=200)
    assert r["0.9"]["rate"] == 0.5 and r["0.9"]["ci"][0] <= 0.5 <= r["0.9"]["ci"][1]
    N.write_jsonl(tmp_path / "review-key.jsonl", [
        {"n": 1, "swap_key": "a" * 64, "band": "≥0.95", "base_mean": 0.99, "base_min": 0.99, "small_min": 0.99},
        {"n": 2, "swap_key": "b" * 64, "band": "<0.80", "base_mean": 0.5, "base_min": 0.5, "small_min": 0.5}])
    (tmp_path / "review-bands.json").write_text('{"≥0.95": 0.8, "<0.80": 0.2}')
    marks = tmp_path / "marks.csv"
    marks.write_text(f"n,swap_key,mark\n1,{'a' * 16},good\n2,{'c' * 16},bad\n")
    import pytest
    with pytest.raises(SystemExit):
        N.run_marks(type("A", (), {"state": tmp_path, "marks": marks}))
    marks.write_text(f"n,swap_key,mark\n1,{'a' * 16},good\n2,{'b' * 16},bad\n")
    N.run_marks(type("A", (), {"state": tmp_path, "marks": marks}))
