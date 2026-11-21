"""Round 9, arm w: sense match and parse stability on top of arm s (see ROUND9.md).

  learned-words r9 senses [--max-seconds N]     SemCor sense table (checkpointed)
  learned-words r9 features [--max-seconds N]   31 arm-s + 13 new features for the 371 dev rows
  learned-words r9 train                        nested CV go/no-go; final fit and pre-commit if go
  learned-words r9 panel                        fresh passages from the round-6 technical corpus
  learned-words r9 apply [--max-seconds N]      stage + arm s + arm w on the panel (checkpointed)
  learned-words r9 page | score                 three blind rater files; majority-vote scoring

Raw text, features and marks stay in the state directory (r9/).
"""

import csv
import gzip
import hashlib
import json
import pickle
import random
import time
from collections import Counter

import numpy as np

from . import learned_words_stacker as S
from .learned_words import REPORT, STATE, read_jsonl

DIR = STATE / "r9"
TABLE = DIR / "senses.pkl"
FEATURES = DIR / "features.jsonl"
MODEL = DIR / "model.pkl"
PANEL = DIR / "passages.json"
APPLY = DIR / "apply.jsonl"
RATING = DIR / "rating"
PRECOMMIT = REPORT / "r9-precommit.json"
CV_REPORT = REPORT / "r9-cv.json"
CORPUS = STATE / "corpus"
KEEP = 0.55
REPEATS = 20
SEED = 20261126
RATERS = 3
CONSISTENCY_ITEMS = 20
PER_SOURCE = 45
PASSAGE_SENTENCES = 4


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _threads(argv, default=4):
    import torch

    torch.set_num_threads(int(argv[argv.index("--threads") + 1]) if "--threads" in argv else default)


def _seconds(argv, default):
    return float(argv[argv.index("--max-seconds") + 1]) if "--max-seconds" in argv else default


def senses(argv):
    from cleartext.learned_words import senses as SN

    _threads(argv)
    DIR.mkdir(exist_ok=True)
    SN.build(TABLE, max_seconds=_seconds(argv, 1500))


def dev_swaps():
    """The 371 development rows: rounds 3-5 (replay items) and the round-8 fresh window.
    Never Joshua's 60, round 2, or the locked test split."""
    items = [dict(s, src=f"replay:{n}") for n, s in enumerate(json.load(open(STATE / "replay/items.json")))]
    mp = json.load(open(STATE / "rating-r8/blind-mixed-map.json"))
    marks = {r["id"]: r["mark"] for r in csv.DictReader(open(STATE / "rating-r8/blind-mixed-marks.csv"))}
    key = {s["key"]: s for s in json.load(open(STATE / "rating-r8/key.json"))["swaps"]}
    r8 = [dict(key[m["key"]], mark=marks[m["id"]], round="r8", src=f"r8:{m['key']}") for m in mp if m["source"] == "r8"]
    return items + r8


def _new_features(sn, s, start):
    from cleartext.learned_words import senses as SN

    return SN.swap_features(sn, s["before"], s["after"], start, s["word"], s["rep"], s["pos"])


def features(argv):
    from cleartext.learned_words import senses as SN

    from .learned_words_simpler import SIMPLER

    _threads(argv)
    deadline = time.monotonic() + _seconds(argv, 1440)
    sn = SN.load(TABLE)
    old = {f"replay:{r['n']}": r["row"] for r in read_jsonl(S.FEATURES)}
    model = simpler = None
    done = {r["src"] for r in read_jsonl(FEATURES)}
    with open(FEATURES, "a") as out:
        for s in dev_swaps():
            if s["src"] in done:
                continue
            if time.monotonic() > deadline:
                break
            row = old.get(s["src"])
            if s["src"].startswith("r8:"):
                if model is None:
                    model = pickle.load(open(STATE / "model.pkl", "rb"))
                    simpler = pickle.load(open(SIMPLER, "rb"))
                row = S.swap_row(model, simpler, s)
            if row is not None:
                row = dict(row, new=_new_features(sn, s, row["start"]))
            out.write(json.dumps({"src": s["src"], "row": row}) + "\n")
            out.flush()
    rows = read_jsonl(FEATURES)
    left = len(dev_swaps()) - len({r["src"] for r in rows})
    print(f"{len(rows)} rows, {sum(r['row'] is None for r in rows)} unresolved, {left} left", flush=True)
    print("COMPLETE" if not left else "PARTIAL", flush=True)


def matrices(T, ids, swaps, rows):
    """(31 arm-s features, 44 arm-w features) for resolved swaps."""
    from cleartext.learned_words import senses as SN

    A = np.array([[r["f"][k] for k in S.STAGE_KEYS] for r in rows], float)
    spans = [dict(s, start=r["start"]) for s, r in zip(swaps, rows)]
    Xs = np.nan_to_num(np.hstack([A, S.rewrite_features(T, ids, spans)]))
    N = np.nan_to_num(np.array([[r["new"][k] for k in SN.KEYS] for r in rows], float))
    return Xs, np.hstack([Xs, N])


def dataset():
    from .learned_words_rewrite import load

    _, T, ids, _ = load()
    by_src = {s["src"]: s for s in dev_swaps()}
    rows = [r for r in read_jsonl(FEATURES) if r["row"] is not None]
    swaps = [by_src[r["src"]] for r in rows]
    Xs, Xw = matrices(T, ids, swaps, [r["row"] for r in rows])
    marks = np.array([s["mark"] for s in swaps])
    b = np.array(["b" in s["arms"] for s in swaps])
    groups = np.array([r["row"]["lemma"] for r in rows])
    return Xs, Xw, marks, b, groups, swaps


def nested(X, y, b, groups, seed):
    from sklearn.model_selection import GroupKFold

    score, keep = np.zeros(len(y)), np.zeros(len(y), bool)
    for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        inner = S._oof(X[tr], y[tr], groups[tr], seed + 1)
        t = np.quantile(inner[b[tr]], 1 - KEEP)
        score[te] = S._fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        keep[te] = score[te] >= t
    return score, keep


def train(argv):
    from sklearn.metrics import roc_auc_score

    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    Xs, Xw, marks, b, groups, swaps = dataset()
    if len(marks) != len(dev_swaps()) or b.sum() != 189:
        raise SystemExit(f"REFUSED: {len(marks)} of {len(dev_swaps())} rows, {b.sum()} of 189 arm-b rows resolved")
    y = (marks == "good").astype(int)
    base = float(y[b].mean())
    meaning_b = int((marks[b] == "meaning").sum())
    res = {"rows": len(y), "arm_b_rows": int(b.sum()), "good_rate_base": base, "meaning_base": meaning_b,
           "keep": KEEP, "repeats": REPEATS}
    for arm, X in (("s", Xs), ("w", Xw)):
        runs = []
        for k in range(REPEATS):
            score, keep = nested(X, y, b, groups, SEED + 10 * k)
            kb = keep & b
            runs.append({"good_rate": float(y[kb].mean()) if kb.any() else 0.0,
                         "keep_share": float(kb.sum() / b.sum()),
                         "meaning_removed": float(1 - (marks[kb] == "meaning").sum() / meaning_b),
                         "errors_removed": float(1 - (marks[kb] != "good").sum() / (marks[b] != "good").sum()),
                         "auc_good_b": float(roc_auc_score(y[b], score[b])),
                         "auc_meaning_b": float(roc_auc_score(marks[b] == "meaning", -score[b]))})
        res[arm] = {"mean": {k: float(np.mean([r[k] for r in runs])) for k in runs[0]},
                    "min_max": {k: [float(np.min([r[k] for r in runs])), float(np.max([r[k] for r in runs]))]
                                for k in runs[0]},
                    "runs_good_rate": [r["good_rate"] for r in runs]}
    w, s = res["w"]["mean"], res["s"]["mean"]
    paired = float(np.mean(np.array(res["w"]["runs_good_rate"]) - np.array(res["s"]["runs_good_rate"])))
    res["paired_good_gain"] = paired
    res["go"] = bool(paired >= 0.03 and w["meaning_removed"] >= 0.5 and w["good_rate"] >= base + 0.10)
    res["features_sha256"] = _sha(FEATURES)
    res["table_sha256"] = _sha(TABLE)
    res["source_sha256"] = {p: _sha(p) for p in ("evals/learned_words_r9.py", "models/src/cleartext/learned_words/senses.py")}
    json.dump(res, open(CV_REPORT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("s", "w")} | {"s": s, "w": w}, indent=1))
    if res["go"]:
        oof = S._oof(Xw, y, groups, SEED)
        t = float(np.quantile(oof[b], 1 - KEEP))
        pickle.dump(S._fit(Xw, y), open(MODEL, "wb"))
        json.dump({"threshold": t, "keep": KEEP, "model_sha256": _sha(MODEL), "table_sha256": _sha(TABLE),
                   "features_sha256": _sha(FEATURES), "arm_s_precommit_sha256": _sha(S.PRECOMMIT),
                   "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(PRECOMMIT, "w"), indent=1)
        print(f"pre-committed threshold {t:.4f}")
    print("COMPLETE")


# --- fresh window ------------------------------------------------------------------

def _ok_sentence(s):
    words = s.split()
    letters = sum(c.isalpha() for c in s)
    return 8 <= len(words) <= 50 and letters >= 0.7 * len(s.replace(" ", "")) and s[-1:] in ".;:)"


def panel(argv):
    """PER_SOURCE passages of PASSAGE_SENTENCES consecutive clean sentences, one per document,
    from a seeded shuffle of each corpus source's documents."""
    from cleartext.learned_words.rewrite import tokens

    if PANEL.exists():
        raise SystemExit(f"REFUSED: {PANEL} exists")
    used = []
    for p in ("panel-r4", "panel-r5", "panel-r8"):
        used += [x["original"] for x in json.load(open(STATE / p / "passages.json"))]
    used += [s["before"] for s in dev_swaps()]
    grams = set()
    for t in used:
        tt = tokens(t)
        grams |= {tuple(tt[i:i + 8]) for i in range(len(tt) - 7)}
    out = []
    for src in ("rfc", "fedreg", "openfda"):
        docs = {}
        for line in gzip.open(CORPUS / f"{src}.jsonl.gz", "rt"):
            r = json.loads(line)
            docs.setdefault(r["doc"], []).append(r["s"])
        order = sorted(docs)
        rng = random.Random(f"{SEED}:{src}")
        rng.shuffle(order)
        n = 0
        for d in order:
            if n >= PER_SOURCE:
                break
            sents = docs[d]
            starts = [i for i in range(len(sents) - PASSAGE_SENTENCES + 1)
                      if all(_ok_sentence(x) for x in sents[i:i + PASSAGE_SENTENCES])]
            if not starts:
                continue
            i = rng.choice(starts)
            text = " ".join(sents[i:i + PASSAGE_SENTENCES])
            tt = tokens(text)
            if any(tuple(tt[k:k + 8]) in grams for k in range(len(tt) - 7)):
                continue
            n += 1
            out.append({"id": f"r9-{src}-{n:02d}", "domain": src, "kind": "standard", "source_doc": d,
                        "first_sentence": i, "original": text, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    json.dump(out, open(PANEL, "w"), indent=1)
    print(json.dumps({"passages": len(out), "by_source": Counter(p["domain"] for p in out)}, indent=1))
    print("COMPLETE")


def load_w():
    pre = json.load(open(PRECOMMIT))
    if _sha(MODEL) != pre["model_sha256"] or _sha(TABLE) != pre["table_sha256"]:
        raise SystemExit("REFUSED: arm w model or sense table differ from the pre-commitment")
    return pickle.load(open(MODEL, "rb")), pre["threshold"]


def apply(argv):
    from cleartext.learned_words import senses as SN

    from .learned_words_apply import word_swaps
    from .learned_words_rewrite import load
    from .learned_words_simpler import SIMPLER

    _threads(argv)
    deadline = time.monotonic() + _seconds(argv, 1380)
    clf_s, _ = S.load_filter()
    clf_w, _ = load_w()
    sn = SN.load(TABLE)
    _, T, ids, _ = load()
    model = pickle.load(open(STATE / "model.pkl", "rb"))
    simpler = pickle.load(open(SIMPLER, "rb"))
    pan = json.load(open(PANEL))
    skip = _rewrite_overlap(pan)
    done = {r["pid"] for r in read_jsonl(APPLY)}
    todo = [p for p in pan if p["id"] not in done and p["id"] not in skip]
    print(f"{len(done)} done, {len(todo)} to do, {len(skip)} skipped for rewrite-corpus overlap", flush=True)
    with open(APPLY, "a") as f:
        for p in todo:
            if time.monotonic() > deadline:
                break
            swaps, seen = word_swaps(p["original"], model, p["id"])
            got, rows = [], []
            for s in swaps:
                r = S.swap_row(model, simpler, s)
                if r is not None:
                    got.append(s)
                    rows.append(dict(r, new=_new_features(sn, s, r["start"])))
            if got:
                Xs, Xw = matrices(T, ids, got, rows)
                for s, r, a, w in zip(got, rows, clf_s.predict_proba(Xs)[:, 1], clf_w.predict_proba(Xw)[:, 1]):
                    s.update(margin=r["f"]["margin"], stack=float(a), w=float(w), new=r["new"])
            out = [s for s in swaps if "w" in s]
            f.write(json.dumps({"pid": p["id"], "domain": p["domain"], "kind": p["kind"], "content_words": seen,
                                "unresolved": len(swaps) - len(out), "swaps": out}) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {p['id']} swaps={len(out)}", flush=True)
    left = len({p["id"] for p in pan} - skip - {r["pid"] for r in read_jsonl(APPLY)})
    print("COMPLETE" if not left else f"PARTIAL {left} left", flush=True)


def _rewrite_overlap(pan):
    """Passage ids sharing a word 8-gram with the rewrite corpora (cached)."""
    from cleartext.learned_words.rewrite import tokens

    from .learned_words_rewrite import pairs

    path = DIR / "panel-overlap.json"
    if path.exists():
        return set(json.load(open(path)))
    index = {}
    for p in pan:
        t = tokens(p["original"])
        for i in range(len(t) - 7):
            index.setdefault(tuple(t[i:i + 8]), set()).add(p["id"])
    hit = set()
    for _, _, a, b in pairs():
        for x in (a, b):
            t = tokens(x)
            for i in range(len(t) - 7):
                hit |= index.get(tuple(t[i:i + 8]), set())
    json.dump(sorted(hit), open(path, "w"))
    return hit


def picks(swaps, ts, tw, choose):
    """(arm b, arm s, arm w) swaps of one passage. Filters apply first, then the per-passage limits."""
    ok = [s for s in swaps if s["margin"] >= S.MIN_MARGIN]
    return choose(ok), choose([s for s in ok if s["stack"] >= ts]), choose([s for s in ok if s["w"] >= tw])


def page(argv):
    from .learned_words_apply import _skey, choose

    _, ts = S.load_filter()
    _, tw = load_w()
    items = {}
    for r in read_jsonl(APPLY):
        for arm, chosen in zip("bsw", picks(r["swaps"], ts, tw, choose)):
            for s in chosen:
                k = _skey(s)
                items.setdefault(k, dict(s, key=k, arms=[], domain=r["domain"]))["arms"].append(arm)
    items = sorted(items.values(), key=lambda x: x["key"])
    rng = random.Random(1126)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    base = [dict(x, source="r9") for x in items] + [dict(x, source="old", old_mark=x["mark"]) for x in old[:CONSISTENCY_ITEMS]]
    RATING.mkdir(exist_ok=True)
    json.dump({"swaps": items}, open(RATING / "key.json", "w"), indent=1)
    ids = {n: f"m{n}" for n in range(1, len(base) + 1)}
    json.dump([{"id": ids[n], "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r9" else None} for n, x in enumerate(base, 1)],
              open(RATING / "blind-map.json", "w"), indent=1)
    for r in range(1, RATERS + 1):
        order = list(enumerate(base, 1))
        random.Random(f"1126:{r}").shuffle(order)
        json.dump([{"id": ids[n], "before": x["before"], "after": x["after"], "word": x["word"], "rep": x["rep"]}
                   for n, x in order], open(RATING / f"blind-{r}.json", "w"), indent=1)
    print(json.dumps({"r9_items": len(items), "items": len(base),
                      "per_arm": Counter(a for x in items for a in x["arms"])}, indent=1))


SEVERITY = ("meaning", "notsimpler", "good")


def majority(ms):
    c = Counter(ms)
    top, n = c.most_common(1)[0]
    if n >= 2:
        return top
    return min(ms, key=SEVERITY.index)


def score(argv):
    from .learned_words_study import wilson

    mp = json.load(open(RATING / "blind-map.json"))
    ids = {m["id"] for m in mp}
    raters = []
    for r in range(1, RATERS + 1):
        marks = {x["id"]: x["mark"] for x in csv.DictReader(open(RATING / f"marks-{r}.csv"))}
        if set(marks) != ids or set(marks.values()) - set(SEVERITY):
            raise SystemExit(f"REFUSED: rater {r} marks are incomplete or use an unknown label")
        raters.append(marks)
    maj = {i: majority([m[i] for m in raters]) for i in ids}
    out = {"items": len(ids), "raters": RATERS}

    def arm_stats(marks):
        res = {}
        for arm in "bsw":
            ms = [marks[m["id"]] for m in mp if m["source"] == "r9" and arm in m["arms"]]
            c = Counter(ms)
            res["arm_" + arm] = {"swaps": len(ms), **c, "good_rate": c["good"] / max(1, len(ms)),
                                 "good_ci95": wilson(c["good"], len(ms)), "meaning_rate": c["meaning"] / max(1, len(ms))}
        return res

    out["majority"] = arm_stats(maj)
    out["per_rater"] = [arm_stats(m) for m in raters]
    b, w = out["majority"]["arm_b"], out["majority"]["arm_w"]
    out["keep_share_w"] = w["swaps"] / max(1, b["swaps"])
    out["keep_share_s"] = out["majority"]["arm_s"]["swaps"] / max(1, b["swaps"])
    out["success"] = bool(w["good_rate"] >= 0.78 and w["meaning_rate"] <= 0.15 and out["keep_share_w"] >= 0.45)
    r9 = [m["id"] for m in mp if m["source"] == "r9"]
    out["unanimous_share"] = sum(len({m[i] for m in raters}) == 1 for i in r9) / max(1, len(r9))
    pairs = [(a, c) for a in range(RATERS) for c in range(a + 1, RATERS)]
    out["pairwise_agreement"] = {f"{a + 1}-{c + 1}": sum(raters[a][i] == raters[c][i] for i in r9) / max(1, len(r9))
                                 for a, c in pairs}
    old = [m for m in mp if m["source"] == "old"]
    out["consistency_with_old"] = [sum(m["old_mark"] == rt[m["id"]] for m in old) for rt in raters] + [len(old)]
    (REPORT / "round9.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv):
    cmd = {"senses": senses, "features": features, "train": train, "panel": panel, "apply": apply,
           "page": page, "score": score}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words r9 senses|features|train|panel|apply|page|score")
