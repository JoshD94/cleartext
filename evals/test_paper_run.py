"""Paper-run alignment, extraction helpers and rating-kit scoring (no models)."""

import csv
import json
import re

import pytest

from evals import paper_extract, paper_page, paper_run, rating_kit, rating_score


def regex_sentences(text):
    spans, start = [], 0
    for m in re.finditer(r"[.!?](?=\s+[A-Z])", text):
        spans.append((start, m.end()))
        start = m.end()
        while start < len(text) and text[start].isspace():
            start += 1
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


ORIGINAL = "The cat sat on the mat, and the dog slept by the door. In order to win, we trained hard. Results were 42% better."


def test_snapshots_follow_structure_phrase_and_word_edits():
    split = ORIGINAL.replace("mat, and the dog", "mat. The dog")
    phrased = split.replace("In order to win", "To win")
    final = phrased.replace("trained", "practiced")
    start = split.index("In order to win")
    entry = {"output": final, "edits": [
        {"stage": "structure", "rule": "learned seq2seq", "before": ORIGINAL, "after": split},
        {"stage": "phrase", "start": start, "end": start + len("In order to win"), "replacement": "To win"},
        {"stage": "word", "source": "trained", "replacement": "practiced"}]}
    assert [s for s, _ in paper_run.snapshots(ORIGINAL, entry)] == ["original", "t5", "phrase", "word"]
    assert paper_run.snapshots(ORIGINAL, entry)[-1][1] == final


def test_snapshots_unrecorded_stage_ends_where_structure_begins():
    passive = ORIGINAL.replace("The cat sat", "The cat was sitting")
    split = passive.replace("mat, and the dog", "mat. The dog")
    entry = {"output": split, "edits": [{"stage": "passive", "sentence": 0},
                                         {"stage": "structure", "rule": "learned seq2seq", "before": passive, "after": split}]}
    assert paper_run.snapshots(ORIGINAL, entry) == [("original", ORIGINAL), ("passive", passive), ("t5", split)]


def test_snapshots_unrecorded_edit_ends_at_output():
    entry = {"output": "changed", "edits": []}
    assert paper_run.snapshots("same", entry)[-1] == ("other", "changed")
    assert paper_run.snapshots("same", {"output": "same", "edits": []}) == [("original", "same")]


def test_segments_keep_a_split_with_its_source_sentence():
    spans = regex_sentences(ORIGINAL)
    split = ORIGINAL.replace("mat, and the dog", "mat. The dog").replace("In order to win", "To win")
    segs = paper_run.segments(ORIGINAL, spans, split, [a for a, _ in regex_sentences(split)])
    assert [s.strip() for s in segs] == ["The cat sat on the mat. The dog slept by the door.",
                                         "To win, we trained hard.", "Results were 42% better."]


def test_segments_replaced_first_word_stays_with_its_sentence():
    original = "We ran. Furthermore, we won the cup."
    output = "We ran. Also, we won the cup."
    spans = regex_sentences(original)
    segs = paper_run.segments(original, spans, output, [a for a, _ in regex_sentences(output)])
    assert [s.strip() for s in segs] == ["We ran.", "Also, we won the cup."]


def test_segments_identity_and_dropped_sentence():
    spans = regex_sentences(ORIGINAL)
    assert paper_run.segments(ORIGINAL, spans, ORIGINAL, []) == [ORIGINAL[a:b] for a, b in spans]
    out = "The cat sat on the mat, and the dog slept by the door. Results were 42% better."
    segs = paper_run.segments(ORIGINAL, spans, out, [a for a, _ in regex_sentences(out)])
    assert segs[1].strip() == "" and segs[2].strip() == "Results were 42% better."


def test_sentence_row_flags_only_changed_sentences(monkeypatch):
    weights = json.loads((paper_run.Path(paper_run.__file__).parent / "trained_weights.json").read_text())
    low = lambda p, h: 0.5
    same = paper_run.sentence_row("It was not 42.", "It was not 42.", [], low, weights)
    assert same["flags"] == [] and not same["changed"]
    row = paper_run.sentence_row("It was not 42.", "It was 41.", ["t5"], low, weights)
    assert row["flags"] == ["nli", "number", "negation"]
    assert row["stages"] == ["t5"] and row["lost"]["numbers"] == ["42"]
    assert set(row["kea_roy"]) == {"S", "S_lex", "S_read", "S_jargon", "P_meaning", "U"}
    dropped = paper_run.sentence_row("It was not 42.", "", ["t5"], low, weights)
    assert dropped["fk_after"] is None and dropped["nli_mean"] == 0.0 and "nli" in dropped["flags"]
    json.dumps(dropped, allow_nan=False)


def test_dehyphenate_prefers_document_spelling_then_wordfreq():
    prefer = paper_extract.vocabulary_for("TurkCorpus TurkCorpus Alva-Manchego multi-reference")
    assert paper_extract.dehyphenate(["data in Turk-", "Corpus and"], prefer) == "data in TurkCorpus and"
    assert paper_extract.dehyphenate(["sim-", "plification"], prefer) == "simplification"
    assert paper_extract.dehyphenate(["Alva-", "Manchego et al."], prefer) == "Alva-Manchego et al."
    assert paper_extract.dehyphenate(["a multi-", "reference corpus"], prefer) == "a multi-reference corpus"


def test_fix_accents_combines_spacing_marks():
    assert paper_extract.fix_accents("Benoˆıt Alu´ısio ˇStajner") == "Benoît Aluísio Štajner"


# --- rating kit -------------------------------------------------------------------

def fake_pool(n_paper=40, n_panel=60):
    pool = []
    for source, n in (("paper", n_paper), ("panel", n_panel)):
        for k in range(n):
            original = f"{source} sentence {k} is long and complicated."
            for condition in rating_kit.CONDITIONS:
                output = original if condition == "control" else f"{source} {k} variant {rating_kit.CONDITIONS.index(condition)} simple."
                pool.append({"source": source, "unit": f"{source}-{k}", "stratum": f"d{k % 3}",
                             "condition": condition, "original": original, "simplified": output,
                             "metrics": {"fk_delta": -k / 10, "kea_roy_S": k / 100, "nli_mean": 1 - k / 200}})
    return pool


def test_kit_sample_is_seeded_stratified_blind_and_unique():
    items = rating_kit.sample(fake_pool(), seed=7)
    again = rating_kit.sample(fake_pool(), seed=7)
    assert [i["item"] for i in items] == [i["item"] for i in again]
    assert len(items) == 50 and len({(i["original"], i["simplified"]) for i in items}) == 50
    for cond in rating_kit.CONDITIONS:
        units = [i["unit"] for i in items if i["condition"] == cond]
        assert len(units) == len(set(units))
    counts = {}
    for i in items:
        counts[(i["source"], i["condition"])] = counts.get((i["source"], i["condition"]), 0) + 1
    assert counts == {(s, c): n for s, plan in rating_kit.QUOTAS.items() for c, n in plan.items()}
    form = rating_kit.form_html(items, "kit-test")
    for word in ("winner", "t5_unchecked", "v7", "control"):
        assert word not in form
    assert "localStorage" in form and "csv" in form.lower()


def write_fake_csvs(tmp_path, items, noise):
    import random
    paths = []
    for r, rater in enumerate(("joshua", "kea-roy", "chhoeng")):
        rng = random.Random(r)
        path = tmp_path / f"{rater}.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(rating_score.COLUMNS)
            for item in items:
                base = 5 if item["condition"] == "control" else 3
                vals = [max(1, min(5, base + rng.choice(noise))) for _ in range(3)]
                w.writerow([rater, item["item"], *vals, ""])
        paths.append(path)
    return paths


def test_rating_score_runs_on_fake_data(tmp_path):
    items = rating_kit.sample(fake_pool(), seed=7)
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"kit": "kit-test", "items": items}))
    out = rating_score.score(key, write_fake_csvs(tmp_path, items, [0, 0, 0, 1, -1]))
    assert out["raters"] == 3 and out["items_rated"] == 50
    for dim in rating_score.DIMENSIONS:
        assert -1 <= out["alpha"][dim]["interval"] <= 1
        assert set(out["correlations"][dim]) == set(rating_score.METRICS)
    assert out["by_condition"]["control"]["simplicity"]["n"] > 0


def test_krippendorff_alpha_matches_published_example():
    # Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability", 4 coders x 12 units.
    n = None
    data = [[1, 2, 3, 3, 2, 1, 4, 1, 2, n, n, n],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, n, 3],
            [n, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, n],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, n]]
    units = [[row[u] for row in data if row[u] is not None] for u in range(12)]
    assert rating_score.alpha(units, "nominal") == pytest.approx(0.743, abs=1e-3)
    assert rating_score.alpha(units, "interval") == pytest.approx(0.849, abs=1e-3)
    assert rating_score.alpha(units, "ordinal") == pytest.approx(0.815, abs=1e-3)
    assert rating_score.alpha([[3, 3, 3], [1, 1]], "interval") == 1.0
    assert rating_score.alpha([[3, 3], [3, 3]], "interval") is None


def test_rating_score_rejects_duplicate_ratings(tmp_path):
    items = rating_kit.sample(fake_pool(), seed=7)
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"kit": "kit-test", "items": items}))
    path = tmp_path / "a.csv"
    path.write_text("rater,item,simplicity,meaning,fluency,comment\nx,R01,3,3,3,\nx,R01,4,4,4,\n")
    with pytest.raises(ValueError, match="twice"):
        rating_score.score(key, [path])
    other = tmp_path / "b.csv"
    other.write_text("rater,item,simplicity,meaning,fluency,comment\nx,R02,3,3,3,\n")
    path.write_text("rater,item,simplicity,meaning,fluency,comment\nx,R01,3,3,3,\n")
    with pytest.raises(ValueError, match="appears in"):
        rating_score.score(key, [path, other])


def test_raw_text_pages_refuse_the_checkout():
    with pytest.raises(SystemExit):
        paper_page.refuse_checkout(paper_run.Path(paper_run.__file__).parent / "page.html")


def test_aggregate_reports_partial_coverage_on_its_own_paragraphs():
    def srow(text, changed):
        return {"source": text, "output": text + (" Done." if changed else ""), "changed": changed,
                "stages": ["word"] if changed else [], "fk_before": 10.0, "fk_after": 9.0 if changed else 10.0,
                "nli_forward": 0.9, "nli_backward": 0.9, "nli_mean": 0.9, "flags": [],
                "lost": {"numbers": [], "negations": [], "entities": []},
                "kea_roy": {"S": 0.5, "U": 0.4, "P_meaning": 0.9}}
    para = {"nli_mean": 0.95, "nli_forward": 0.95, "nli_backward": 0.95, "numbers_preserved": 1.0,
            "negations_preserved": 1.0, "entities_preserved": 1.0}
    rows = []
    for k in range(3):
        text = f"Sentence {k} is here."
        configs = {"winner": {"output": text, "rows": [srow(text, False)], "paragraph": para}}
        if k == 0:
            configs["v7_words"] = {"output": text + " Done.", "rows": [srow(text, True)], "paragraph": para}
        rows.append({"id": f"p{k}", "section": "S1" if k < 2 else "S2", "original": text,
                     "sentences": [text], "configs": configs})
    out = paper_run.aggregate(rows)
    assert out["configs"]["winner"]["paragraphs_scored"] == 3
    v7 = out["configs"]["v7_words"]
    assert (v7["paragraphs_scored"], v7["sentences_scored"], v7["sentences_changed"]) == (1, 1, 1)
    assert v7["percent_changed"] == 100.0
    s1 = out["sections"][0]
    assert s1["fk_v7_words"] is None and s1["changed_v7_words"] == 1 and s1["fk_winner"] is not None
