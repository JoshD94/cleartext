"""Round 25: round 24's arms frozen on a new window, a replication (see ROUND25.md).

  learned-words r25 precommit                fix round 24's models and thresholds (no model is fitted)
  learned-words r25 apply [--max-seconds N]  fresh panel (once), round 11's generator, frame features (checkpointed)
  learned-words r25 page | score             three blind rater files; majority-vote scoring

Raw text and marks stay in the state directory (r25/). Round 24's apply runs unchanged; the panel,
page and score functions are round 24's with round 25's tags, seeds and one more excluded window.
"""

import contextlib
import csv
import hashlib
import json
import random
import time
from collections import Counter

from . import learned_words_r12 as P12
from . import learned_words_r19 as P19
from . import learned_words_r21 as Q
from . import learned_words_r22 as G
from . import learned_words_r24 as F
from . import learned_words_stacker as S
from .learned_words import REPORT, STATE, read_jsonl

DIR = STATE / "r25"
PANEL = DIR / "passages.json"
APPLY = DIR / "apply.jsonl"
RATING = DIR / "rating"
R24_PANEL = F.PANEL
POOL = F.POOL  # unseen pairs: absent from the 1,454 rows arm f was fitted on
# more technical text than round 24 (frame errors live there): the last unused Federal Register rules too
PER_SOURCE = {"news": ("news2", 40), "wiki": ("wiki2", 50), "openfda": ("openfda", 50), "fedreg": ("fedreg", 20)}
GENRES, EXT_NAMES = F.GENRES, F.EXT_NAMES
_docs, load_models, picks = F._docs, F.load_models, F.picks
PRECOMMIT = REPORT / "r25-precommit.json"
RESULT = REPORT / "round25.json"
SEED = 20261225
SOURCES = ("evals/learned_words_r25.py",) + F.SOURCES


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


@contextlib.contextmanager
def window():
    """Point round 24's apply at round 25's window: its directory, seed, pre-commitment and panel rule."""
    names = ("DIR", "PANEL", "APPLY", "RATING", "PRECOMMIT", "RESULT", "SEED", "panel")
    saved = {k: getattr(F, k) for k in names}
    F.DIR, F.PANEL, F.APPLY, F.RATING = DIR, PANEL, APPLY, RATING
    F.PRECOMMIT, F.RESULT, F.SEED, F.panel = PRECOMMIT, RESULT, SEED, panel
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(F, k, v)


def _corpus(name):
    return (F.GEN.GENRE if name in EXT_NAMES.values() else Q.R.CORPUS) / f"{name}.jsonl.gz"


def precommit(argv):
    """Round 25 fits nothing: it fixes round 24's arm f with both thresholds and the frozen p50 and qg,
    after checking them against round 24's pre-commitment and its result file."""
    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    from . import learned_words_r9 as R

    pre24 = json.load(open(F.PRECOMMIT))
    gate = {"r24_cv_go": bool(json.load(open(F.CV_REPORT))["go"]), "r24_scored": F.RESULT.exists(),
            "r24_f_met_its_bars": bool(json.load(open(F.RESULT))["success"]),
            "f_model_unchanged": _sha(F.MODEL_F) == pre24["model_f_sha256"],
            "p50_model_unchanged": _sha(P19.MODEL) == pre24["model_p50_sha256"],
            "qg_model_unchanged": _sha(G.MODEL_QG) == pre24["model_qg_sha256"]}
    print(json.dumps(gate, indent=1))
    if not all(gate.values()):
        raise SystemExit("NO-GO: no fresh text is spent")
    DIR.mkdir(parents=True, exist_ok=True)
    keep = ("thresholds", "arms", "keep_tech", "keep_genre", "f_args", "model_f_sha256", "model_p50_sha256",
            "model_qg_sha256", "pool_sha256", "frame_features_sha256", "frame_keys_sha256", "pair_lexical_sha256",
            "wc_lexicon_sha256", "simpleppdbpp_raw_sha256")
    json.dump({"gate": gate, **{k: pre24[k] for k in keep}, "r24_precommit_sha256": _sha(F.PRECOMMIT),
               "sense_table_sha256": _sha(R.TABLE), "arm_s_model_sha256": _sha(S.MODEL),
               "source_sha256": {p: _sha(p) for p in SOURCES}, "panel_seed": SEED,
               "per_source": {k: v[1] for k, v in PER_SOURCE.items()},
               "corpus_sha256": {v[0]: _sha(_corpus(v[0])) for v in PER_SOURCE.values()},
               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(PRECOMMIT, "w"), indent=1)
    print(open(PRECOMMIT).read())


# --- round 24's window functions, with round 25's tags and exclusions ----------------------

def panel():
    """Round 24's panel rule (seed SEED) on the extended genre corpora and unused openFDA labels.
    Excludes every document of every earlier window (rounds 11-24, both Joshua windows) and any
    passage sharing a word 8-gram with any earlier panel or window, any rated development sentence
    or test-set v3's panel file (exclusion only)."""
    from cleartext.learned_words.rewrite import tokens

    from . import learned_words_r9 as R

    if PANEL.exists():
        raise SystemExit(f"REFUSED: {PANEL} exists")
    prev = []
    for p in (R.PANEL, Q.P12.PANEL, Q.P13.PANEL, Q.P17.PANEL, Q.P18.PANEL, Q.P19.PANEL, Q.P20.PANEL,
              Q.DIR / "passages.json", Q.JDIR / "passages.json", STATE / "joshua-r21b/passages.json", G.PANEL, R24_PANEL):
        prev += json.load(open(p))
    used = [x["original"] for x in prev]
    for p in ("panel-r4", "panel-r5", "panel-r8"):
        used += [x["original"] for x in json.load(open(STATE / p / "passages.json"))]
    used += [s["before"] for s in R.dev_swaps()]
    used += [x["original"] for x in json.load(open(Q.V3_PANEL))]
    grams = P12._grams(used)
    prev_docs = {(x["domain"], x["source_doc"]) for x in prev}
    out = []
    for dom, (name, want) in PER_SOURCE.items():
        docs = _docs(name)
        order = sorted(docs)
        rng = random.Random(f"{SEED}:{dom}")
        rng.shuffle(order)
        n = 0
        for doc in order:
            if n >= want:
                break
            if (dom, doc) in prev_docs:
                continue
            sents = docs[doc]
            starts = [i for i in range(len(sents) - P12.PASSAGE_SENTENCES + 1)
                      if all(R._ok_sentence(x) for x in sents[i:i + P12.PASSAGE_SENTENCES])]
            if not starts:
                continue
            i = rng.choice(starts)
            text = " ".join(sents[i:i + P12.PASSAGE_SENTENCES])
            tt = tokens(text)
            if any(tuple(tt[k:k + 8]) in grams for k in range(len(tt) - 7)):
                continue
            n += 1
            out.append({"id": f"r25-{dom}-{n:02d}", "domain": dom, "kind": "standard", "source_doc": doc,
                        "corpus": name, "first_sentence": i, "original": text,
                        "sha256": hashlib.sha256(text.encode()).hexdigest()})
    DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(PANEL, "w"), indent=1)
    print(json.dumps({"passages": len(out), "by_source": Counter(p["domain"] for p in out)}), flush=True)


def page(argv):
    from cleartext.learned_words.pairlex import PairLex

    from .learned_words_apply import _skey, choose

    models, pre = load_models()
    lexicon = PairLex()
    items = {}
    for r in read_jsonl(APPLY):
        for arm, chosen in picks(r, models, pre["thresholds"], choose, lexicon).items():
            if arm not in pre["arms"]:
                continue
            for s in chosen:
                k = _skey(s)
                it = items.setdefault(k, dict({x: v for x, v in s.items() if x not in ("x31", "fr", "nb")}, key=k, arms=[],
                                              domain=r["domain"], shown=Q.realized(s)))
                if arm not in it["arms"]:
                    it["arms"].append(arm)
    items = sorted(items.values(), key=lambda x: x["key"])
    rng = random.Random(1225)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    base = [dict(x, source="r25") for x in items] + [dict(x, source="old", old_mark=x["mark"], shown=x["after"])
                                                     for x in old[:P12.CONSISTENCY_ITEMS]]
    RATING.mkdir(parents=True, exist_ok=True)
    for r in range(1, P12.RATERS + 1):
        if (RATING / f"marks-{r}.csv").exists():
            raise SystemExit("REFUSED: marks exist; the page is fixed")
    json.dump({"swaps": items}, open(RATING / "key.json", "w"), indent=1)
    ids = {n: f"m{n}" for n in range(1, len(base) + 1)}
    json.dump([{"id": ids[n], "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r25" else None} for n, x in enumerate(base, 1)],
              open(RATING / "blind-map.json", "w"), indent=1)
    brief = (STATE / "r9/rater-brief.txt").read_text()
    for r in range(1, P12.RATERS + 1):
        order = list(enumerate(base, 1))
        random.Random(f"1225:{r}").shuffle(order)
        path = RATING / f"blind-{r}.json"
        json.dump([{"id": ids[n], "before": x["before"], "after": x["shown"], "word": x["word"], "rep": x["rep"]}
                   for n, x in order], open(path, "w"), indent=1)
        n = len(base)
        b = brief.replace(str(STATE / "rating-r5/blind-mixed.json"), str(path)).replace("254 items", f"{n} items")
        b = b.replace("(all 254, ids m1..m254)", f"(all {n}, ids m1..m{n})")
        b = b.replace(str(STATE / "rating-r5/blind-mixed-marks.csv"), str(RATING / f"marks-{r}.csv"))
        b = b.replace("has 255 lines", f"has {n + 1} lines")
        (RATING / f"brief-{r}.txt").write_text(b)
    print(json.dumps({"r25_items": len(items), "items": len(base), "realized": sum(x["shown"] != x["after"] for x in items),
                      "per_arm": Counter(a for x in items for a in x["arms"]),
                      "per_domain": Counter(x["domain"] for x in items)}, indent=1))


def score(argv):
    from . import learned_words_r9 as R
    from .learned_words_study import wilson

    load_models()
    pool_rows = json.load(open(POOL))
    pool_pairs = {tuple(r["key"]) for r in pool_rows}
    c = {}
    for r in pool_rows:
        x = c.setdefault(tuple(r["key"]), [0, 0])
        x[0] += r["good"]
        x[1] += 1
    well = {k for k, (g, n) in c.items() if n >= 3 and g / n >= 0.7}
    mp = json.load(open(RATING / "blind-map.json"))
    ids = {m["id"] for m in mp}
    raters = []
    for r in range(1, P12.RATERS + 1):
        marks = {x["id"]: x["mark"] for x in csv.DictReader(open(RATING / f"marks-{r}.csv"))}
        if set(marks) != ids or set(marks.values()) - set(R.SEVERITY):
            raise SystemExit(f"REFUSED: rater {r} marks are incomplete or use an unknown label")
        raters.append(marks)
    maj = {i: R.majority([m[i] for m in raters]) for i in ids}
    key = {s["key"]: s for s in json.load(open(RATING / "key.json"))["swaps"]}
    recs = read_jsonl(APPLY)
    sentences = sum(r["sentences"] for r in recs)
    by_dom = Counter()
    for r in recs:
        by_dom[r["domain"]] += r["sentences"]

    def pk(m):
        s = key[m["key"]]
        return tuple(P12.pair_key(s["word"], s["rep"], s["pos"]))

    arms = json.load(open(PRECOMMIT))["arms"]

    def arm_stats(marks):
        res = {}
        for arm in arms:
            got = [m for m in mp if m["source"] == "r25" and arm in m["arms"]]
            ms = [marks[m["id"]] for m in got]
            cc = Counter(ms)
            un = [marks[m["id"]] for m in got if pk(m) not in pool_pairs]
            fw = [marks[m["id"]] for m in got if pk(m) in well]
            res["arm_" + arm] = {"swaps": len(ms), **cc, "good_rate": cc["good"] / max(1, len(ms)),
                                 "good_ci95": wilson(cc["good"], len(ms)), "meaning_rate": cc["meaning"] / max(1, len(ms)),
                                 "per_100_sentences": 100 * len(ms) / max(1, sentences),
                                 "distinct_pairs": len({pk(m) for m in got}),
                                 "unseen_pairs": {"swaps": len(un), "good": un.count("good"), "meaning": un.count("meaning")},
                                 "well_rated_pairs": {"swaps": len(fw), "meaning": fw.count("meaning")},
                                 "by_domain": {d: {"swaps": sum(key[m["key"]]["domain"] == d for m in got),
                                                   "good": sum(key[m["key"]]["domain"] == d and marks[m["id"]] == "good"
                                                               for m in got)} for d in PER_SOURCE}}
        return res

    out = {"items": len(ids), "raters": P12.RATERS, "passages": len(recs), "sentences": sentences,
           "sentences_by_domain": dict(by_dom), "majority": arm_stats(maj), "per_rater": [arm_stats(m) for m in raters]}
    M = out["majority"]
    out["yield_vs_b"] = {a: M["arm_" + a]["swaps"] / max(1, M["arm_b"]["swaps"]) for a in arms}
    v, p = M["arm_f"], M["arm_p50"]
    out["success_f"] = {"good_ge_0.90": v["good_rate"] >= 0.90, "meaning_le_0.08": v["meaning_rate"] <= 0.08,
                        "yield_ge_0.40": out["yield_vs_b"]["f"] >= 0.40}
    out["success"] = bool(all(out["success_f"].values()))
    out["secondary_f"] = {
        "frame_errors_le_p50": v["well_rated_pairs"]["meaning"] <= p["well_rated_pairs"]["meaning"],
        "unseen_swaps_ge_1.25x_p50": v["unseen_pairs"]["swaps"] >= 1.25 * p["unseen_pairs"]["swaps"],
        "unseen_good_ge_0.90": v["unseen_pairs"]["good"] >= 0.90 * max(1, v["unseen_pairs"]["swaps"])}
    if "fg" in arms:
        w = M["arm_fg"]
        out["success_fg"] = {"good_ge_0.90": w["good_rate"] >= 0.90, "meaning_le_0.08": w["meaning_rate"] <= 0.08,
                             "yield_ge_0.40": out["yield_vs_b"]["fg"] >= 0.40}
    r25 = [x["id"] for x in mp if x["source"] == "r25"]
    out["unanimous_share"] = sum(len({rt[i] for rt in raters}) == 1 for i in r25) / max(1, len(r25))
    out["pairwise_agreement"] = {f"{a + 1}-{c + 1}": sum(raters[a][i] == raters[c][i] for i in r25) / max(1, len(r25))
                                 for a in range(P12.RATERS) for c in range(a + 1, P12.RATERS)}
    old = [x for x in mp if x["source"] == "old"]
    out["consistency_with_old"] = [sum(x["old_mark"] == rt[x["id"]] for x in old) for rt in raters] + [len(old)]
    RESULT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_rater"}, indent=1))


def main(argv):
    if argv[:1] == ["precommit"]:
        return precommit(argv[1:])
    if argv[:1] == ["apply"]:
        with window():
            return F.apply(argv[1:])
    cmd = {"page": page, "score": score}
    if argv[:1] and argv[0] in cmd:
        with window():
            return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words r25 precommit|apply|page|score")
