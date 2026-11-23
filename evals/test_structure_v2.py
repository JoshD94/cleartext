"""Structure v2 harness: vote, interval, tally and pool helpers (no models)."""

from pathlib import Path

from evals import structure_v2 as S


def test_majority_and_most_severe():
    assert S.majority(["good", "good", "meaning"]) == "good"
    assert S.majority(["good", "notsimpler", "meaning"]) == "meaning"
    assert S.majority(["good", "notsimpler", "ungrammatical"]) == "ungrammatical"


def test_wilson_interval():
    lo, hi = S.wilson(37, 40)
    assert 0.79 < lo < 0.81 and 0.97 < hi < 0.98
    assert S.wilson(0, 0) == [None, None]


def test_tally_counts_shared_cards_per_arm():
    items = [{"arm": "a", "card": "c1", "rules": ["r"], "source": "x"},
             {"arm": "a", "card": "c2", "rules": ["r"], "source": "y"},
             {"arm": "b", "card": "c1", "rules": ["s"], "source": "x"}]
    marks = [{"c1": "good", "c2": "meaning"}, {"c1": "good", "c2": "meaning"}, {"c1": "notsimpler", "c2": "good"}]
    out, agree = S.tally(items, marks, ["a", "b"], {"a": {"changed": 2, "sentences": 100}})
    assert out["a"]["n"] == 2 and out["a"]["good"] == 1 and out["a"]["marks"] == {"good": 1, "meaning": 1}
    assert out["b"]["good_rate"] == 1.0
    assert out["a"]["edits_per_100_sentences"] == 2.0
    assert 0 < agree["pairwise"] < 1


def test_pool_helpers():
    assert S.bucket("rfc9000") == S.bucket("rfc9000")
    assert S.clean("The committee approved the new budget for the coming fiscal year.")
    assert not S.clean("Too short.")
    assert not S.clean("DOSAGE AND ADMINISTRATION The dose is one tablet taken twice a day with food.")
    assert S.grams("a b c d e f g h i", 8) == {tuple("abcdefgh"), tuple("bcdefghi")}


def test_pool_c_excludes_every_earlier_pool_key_and_dev_pool(tmp_path, monkeypatch):
    import argparse
    import gzip
    import json
    old, new, dl = tmp_path / "old", tmp_path / "new", tmp_path / "dl"
    for d in (old, new / "dev", dl):
        d.mkdir(parents=True)
    a = "These events were adjudicated by an independent blinded committee of experts."
    b = "The song was written and produced by a team of five writers in two weeks."
    c = "The rule was proposed by the agency after a long period of public comment."
    keep = "The bridge was opened by the mayor on a cold morning in early spring."
    (old / "pool-E.json").write_text(json.dumps([{"doc": "x", "sentence": a}]))  # an old pool (round 8's gap)
    (new / "rating-r9-key.json").write_text(json.dumps({"items": [{"before": b}]}))  # a rating key
    (new / "dev" / "devpool.json").write_text(json.dumps([{"doc": "y", "sentence": c}]))  # the dev pool
    (new / "pool-L.json").write_text(json.dumps([{"doc": "z", "sentence": keep}]))  # its own old build
    with gzip.open(dl / "openfda.jsonl.gz", "wt") as w:
        for i, sent in enumerate(["Note: " + a, b + " Later.", c, keep]):
            w.write(json.dumps({"doc": f"d{i}", "s": sent}) + "\n")
    monkeypatch.setattr(S, "OLD_STATE", old)
    monkeypatch.setattr(S, "used_texts", lambda: (set(), [], []))
    monkeypatch.setattr(S, "outside_checkout", lambda p: Path(p))
    S.run_pool_c(argparse.Namespace(state=str(new), download=str(dl), name="L", per_source=10, per_doc=5))
    wide = json.loads((new / "pool-Lwide.json").read_text())
    assert [r["sentence"] for r in wide] == [keep]
    assert wide[0]["source"] == "openfda12"
    manifest = json.loads((new / "pool-L-manifest.json").read_text())
    assert manifest["sources"]["openfda12"]["sentences_8gram_overlap"] == 3


def test_prefilter_superset():
    assert S.could_fire("The rule was challenged by OSHA.", ["passive_v6"])
    assert not S.could_fire("OSHA challenged the rule.", ["passive_v6"])
    assert S.could_fire("The agency conducted an evaluation of the program.", ["nominal_v3"])
    assert not S.could_fire("The agency evaluated the program.", ["nominal_v3"])
    assert S.could_fire("Anything at all.", ["fronted_split"])
