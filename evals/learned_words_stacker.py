"""Round 8, arm s: a small filter trained on the blind-rated swaps (see ROUND8.md).

  learned-words stacker features [--max-seconds N]   one feature row per rated swap (checkpointed)
  learned-words stacker train                        nested CV go/no-go; final fit and pre-commit if go

Raw text and features stay in the state directory (stacker/).
"""

import hashlib
import json
import pickle
import time
from collections import Counter

import numpy as np

from .learned_words import REPORT, STATE, read_jsonl

DIR = STATE / "stacker"
FEATURES = DIR / "features.jsonl"
MODEL = DIR / "model.pkl"
PRECOMMIT = REPORT / "stacker-precommit.json"
CV_REPORT = REPORT / "stacker-cv.json"
KEEP = 0.65
REPEATS = 20
SEED = 20261124
STAGE_KEYS = ("check", "nli_fw", "nli_bw", "nli_min", "ctx_cos", "sent_cos", "d_pll", "mask_logp", "d_mask_logp",
              "mask_rank", "d_zipf", "t_zipf", "c_zipf", "d_len", "d_syll", "t_senses", "c_senses", "margin",
              "p_complex", "p_term", "pos_n", "pos_v", "pos_a", "pos_r")
REWRITE_KEYS = ("all_keep", "tech_keep", "all_wr", "all_direction", "tech_n", "ctx_keep_min", "ctx_lift")
KEYS = STAGE_KEYS + REWRITE_KEYS


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def span(before, after, word, rep):
    """(start, end) of the occurrence of `word` that `after` replaced with `rep`."""
    i = -1
    while (i := before.find(word, i + 1)) >= 0:
        if before[:i] == after[:i] and before[i + len(word):] == after[i + len(rep):]:
            return i, i + len(word)
    return None


def swap_row(model, simpler, s):
    """Stage features of one (sentence, word, replacement), or None if it does not resolve."""
    from cleartext.features import nlp
    from cleartext.learned_words import eligibility as E
    from cleartext.learned_words import features as F

    found = span(s["before"], s["after"], s["word"], s["rep"])
    if found is None:
        return None
    start, end = found
    text = s["before"]
    doc = nlp()(text)
    token = next((t for t in doc if t.idx == start), None)
    if token is None:
        return None
    inst = {"id": "stack", "dataset": "stack", "group": token.lemma_.lower(), "text": text, "start": start,
            "end": end, "target": s["word"], "gold": {}, "rated": []}
    # The same candidate generation the stage ran, so rank features match application.
    rows = [r for r in F.featurize(inst, given=False) if r["word"].lower() == s["rep"].lower()]
    if not rows:
        return None
    row = rows[0]
    row["f"].update({"p_complex": float(model["complex"].predict([E.context_features(doc, token)])[0]),
                     "p_term": float(model["term"].predict([token.lemma_.lower()])[0])})
    a, b = F.window(text, start, end, row["word"])
    fw, bw = F.entail([(a, b), (b, a)])
    row["f"].update({"nli_fw": fw, "nli_bw": bw, "nli_min": min(fw, bw)})
    f = row["f"]
    out = {k: float(f[k]) for k in STAGE_KEYS if k not in ("check", "margin")}
    out["check"] = float(model["stage"].check([row])[0])
    out["margin"] = float(simpler.margin([s["word"]], [s["rep"]])[0])
    return {"lemma": row["lemma"], "start": start, "f": out}


def rewrite_features(T, ids, swaps):
    """The seven rewrite-table statistics, using all counts (fold -1)."""
    from cleartext.learned_words.rewrite import KEYS as RK
    from cleartext.learned_words.rewrite import matrix, tokens

    W, R, P, N, ws, rs = [], [], [], [], [], []
    for s in swaps:
        toks = tokens(s["before"])
        w, r = tokens(s["word"])[0], tokens(s["rep"])[0]
        k = len(tokens(s["before"][:s["start"]]))
        if k >= len(toks) or toks[k] != w:
            k = toks.index(w) if w in toks else 0
        W.append(ids.get(w, 0))
        R.append(ids.get(r, 0))
        P.append(ids.get(toks[k - 1], 0) if k else 0)
        N.append(ids.get(toks[k + 1], 0) if k + 1 < len(toks) else 0)
        ws.append(w)
        rs.append(r)
    X = matrix(T, W, R, P, N, np.full(len(W), -1), ws, rs)
    return X[:, [RK.index(k) for k in REWRITE_KEYS]]


def features(argv):
    import torch

    from .learned_words_simpler import SIMPLER

    torch.set_num_threads(int(argv[argv.index("--threads") + 1]) if "--threads" in argv else 4)
    max_seconds = float(argv[argv.index("--max-seconds") + 1]) if "--max-seconds" in argv else 1440
    DIR.mkdir(exist_ok=True)
    model = pickle.load(open(STATE / "model.pkl", "rb"))
    simpler = pickle.load(open(SIMPLER, "rb"))
    items = json.load(open(STATE / "replay/items.json"))
    done = {r["n"] for r in read_jsonl(FEATURES)}
    deadline = time.monotonic() + max_seconds
    with open(FEATURES, "a") as out:
        for n, s in enumerate(items):
            if n in done:
                continue
            if time.monotonic() > deadline:
                break
            row = swap_row(model, simpler, s)
            out.write(json.dumps({"n": n, "row": row}) + "\n")
            out.flush()
    rows = read_jsonl(FEATURES)
    left = len(items) - len({r["n"] for r in rows})
    print(f"{len(rows)} rows, {sum(r['row'] is None for r in rows)} unresolved, {left} left", flush=True)
    print("COMPLETE" if not left else "PARTIAL", flush=True)


def dataset():
    from .learned_words_rewrite import load

    _, T, ids, _ = load()
    items = json.load(open(STATE / "replay/items.json"))
    rows = [r for r in read_jsonl(FEATURES) if r["row"] is not None]
    swaps = [dict(items[r["n"]], start=r["row"]["start"]) for r in rows]
    A = np.array([[r["row"]["f"][k] for k in STAGE_KEYS] for r in rows], float)
    X = np.nan_to_num(np.hstack([A, rewrite_features(T, ids, swaps)]))
    y = np.array([s["mark"] == "good" for s in swaps], int)
    b = np.array(["b" in s["arms"] for s in swaps])
    groups = np.array([r["row"]["lemma"] for r in rows])
    return X, y, b, groups, swaps


def _fit(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X, y)


def _oof(X, y, groups, seed):
    from sklearn.model_selection import GroupKFold

    p = np.zeros(len(y))
    for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        p[te] = _fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def nested(X, y, b, groups, seed):
    """Outer out-of-fold scores and keep decisions; each threshold comes from inner folds."""
    from sklearn.model_selection import GroupKFold

    score, keep = np.zeros(len(y)), np.zeros(len(y), bool)
    for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        inner = _oof(X[tr], y[tr], groups[tr], seed + 1)
        t = np.quantile(inner[b[tr]], 1 - KEEP)
        score[te] = _fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        keep[te] = score[te] >= t
    return score, keep


def train(argv):
    from sklearn.metrics import roc_auc_score

    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    X, y, b, groups, swaps = dataset()
    if b.sum() != 153:
        raise SystemExit(f"REFUSED: {b.sum()} of 153 arm-b swaps resolved")
    marks = np.array([s["mark"] for s in swaps])
    base_good = y[b].mean()
    errors = (marks[b] != "good").sum()
    runs = []
    for k in range(REPEATS):
        score, keep = nested(X, y, b, groups, SEED + 10 * k)
        kb = keep & b
        runs.append({"good_rate": float(y[kb].mean()) if kb.any() else 0.0, "keep_share": float(kb.sum() / b.sum()),
                     "errors_removed": float(1 - (marks[kb] != "good").sum() / errors),
                     "auc_all": float(roc_auc_score(y, score)), "auc_b": float(roc_auc_score(y[b], score[b]))})
    mean = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    spread = {k: [float(np.min([r[k] for r in runs])), float(np.max([r[k] for r in runs]))] for k in runs[0]}
    go = bool(mean["good_rate"] >= base_good + 0.05 and mean["keep_share"] >= 0.60 and mean["errors_removed"] >= 1 / 3)
    res = {"rows": len(y), "stage_b_rows": int(b.sum()), "good_rate_base": float(base_good), "repeats": REPEATS,
           "mean": mean, "min_max": spread, "go": go, "features": list(KEYS),
           "features_sha256": _sha(FEATURES), "source_sha256": _sha("evals/learned_words_stacker.py")}
    json.dump(res, open(CV_REPORT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=1))
    if go:
        oof = _oof(X, y, groups, SEED)
        t = float(np.quantile(oof[b], 1 - KEEP))
        pickle.dump(_fit(X, y), open(MODEL, "wb"))
        json.dump({"threshold": t, "keys": list(KEYS), "model_sha256": _sha(MODEL), "features_sha256": _sha(FEATURES),
                   "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(PRECOMMIT, "w"), indent=1)
        print(f"pre-committed threshold {t:.4f}")
    print("COMPLETE")


# --- round 8 fresh window -----------------------------------------------------------

APPLY_R8 = STATE / "apply-r8"
RATING_R8 = STATE / "rating-r8"
OVERLAP_R8 = DIR / "panel-r8-overlap.json"
MIN_MARGIN = 0.1  # the current stage's pre-committed simplicity margin (arm b)
CONSISTENCY_ITEMS = 20


def load_filter():
    pre = json.load(open(PRECOMMIT))
    if _sha(MODEL) != pre["model_sha256"] or tuple(pre["keys"]) != KEYS:
        raise SystemExit("REFUSED: stacker model differs from the pre-commitment")
    return pickle.load(open(MODEL, "rb")), pre["threshold"]


def panel_overlap(panel):
    """Passage ids sharing a word 8-gram with the rewrite corpora (never rated or applied)."""
    from cleartext.learned_words.rewrite import tokens

    from .learned_words_rewrite import pairs

    if OVERLAP_R8.exists():
        return set(json.load(open(OVERLAP_R8)))
    index = {}
    for p in panel:
        t = tokens(p["original"])
        for i in range(len(t) - 7):
            index.setdefault(tuple(t[i:i + 8]), set()).add(p["id"])
    hit = set()
    for _, _, a, b in pairs():
        for x in (a, b):
            t = tokens(x)
            for i in range(len(t) - 7):
                hit |= index.get(tuple(t[i:i + 8]), set())
    json.dump(sorted(hit), open(OVERLAP_R8, "w"))
    return hit


def score_swaps(clf, T, ids, swaps, rows):
    A = np.array([[r["f"][k] for k in STAGE_KEYS] for r in rows], float)
    spans = [dict(s, start=r["start"]) for s, r in zip(swaps, rows)]
    return clf.predict_proba(np.nan_to_num(np.hstack([A, rewrite_features(T, ids, spans)])))[:, 1]


def apply(argv):
    import torch

    from .learned_words_apply import word_swaps
    from .learned_words_panel import PANEL_R8
    from .learned_words_rewrite import load
    from .learned_words_simpler import SIMPLER

    torch.set_num_threads(int(argv[argv.index("--threads") + 1]) if "--threads" in argv else 4)
    max_seconds = float(argv[argv.index("--max-seconds") + 1]) if "--max-seconds" in argv else 1380
    clf, _ = load_filter()
    _, T, ids, _ = load()
    model = pickle.load(open(STATE / "model.pkl", "rb"))
    simpler = pickle.load(open(SIMPLER, "rb"))
    panel = json.load(open(PANEL_R8))
    skip = panel_overlap(panel)
    APPLY_R8.mkdir(exist_ok=True)
    path = APPLY_R8 / "passages.jsonl"
    done = {r["pid"] for r in read_jsonl(path)}
    todo = [p for p in panel if p["id"] not in done and p["id"] not in skip]
    print(f"{len(done)} done, {len(todo)} to do, {len(skip)} skipped for rewrite-corpus overlap", flush=True)
    deadline = time.monotonic() + max_seconds
    with open(path, "a") as f:
        for p in todo:
            if time.monotonic() > deadline:
                break
            swaps, seen = word_swaps(p["original"], model, p["id"])
            pairs = [(s, r) for s in swaps if (r := swap_row(model, simpler, s)) is not None]
            if pairs:
                got, rows = zip(*pairs)
                for s, r, x in zip(got, rows, score_swaps(clf, T, ids, got, rows)):
                    s.update(margin=r["f"]["margin"], stack=float(x))
            out = [s for s in swaps if "stack" in s]
            f.write(json.dumps({"pid": p["id"], "domain": p["domain"], "kind": p["kind"], "content_words": seen,
                                "unresolved": len(swaps) - len(out), "swaps": out}) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {p['id']} swaps={len(out)}", flush=True)
    left = len({p["id"] for p in panel} - skip - {r["pid"] for r in read_jsonl(path)})
    print("COMPLETE" if not left else f"PARTIAL {left} left", flush=True)


def arm_picks(swaps, t, choose):
    """(arm b, arm s) swaps of one passage. Arm s filters first, then applies the per-passage limits."""
    ok = [s for s in swaps if s["margin"] >= MIN_MARGIN]
    return choose(ok), choose([s for s in ok if s["stack"] >= t])


def page(argv):
    import random

    from .learned_words_apply import _skey, choose

    _, t = load_filter()
    items = {}
    for r in read_jsonl(APPLY_R8 / "passages.jsonl"):
        for arm, chosen in zip("bs", arm_picks(r["swaps"], t, choose)):
            for s in chosen:
                k = _skey(s)
                items.setdefault(k, dict(s, key=k, arms=[], domain=r["domain"], kind=r["kind"]))["arms"].append(arm)
    items = sorted(items.values(), key=lambda x: x["key"])
    rng = random.Random(1125)
    old = [i for i in json.load(open(STATE / "replay/items.json")) if i["round"] == "r5" and "b" in i["arms"]]
    rng.shuffle(old)
    mixed = [dict(x, source="r8") for x in items] + [dict(x, source="old", old_mark=x["mark"]) for x in old[:CONSISTENCY_ITEMS]]
    rng.shuffle(mixed)
    RATING_R8.mkdir(exist_ok=True)
    json.dump({"swaps": items}, open(RATING_R8 / "key.json", "w"), indent=1)
    json.dump([{"id": f"m{n}", "source": x["source"], "key": x.get("key"), "old_mark": x.get("old_mark"),
                "arms": x.get("arms") if x["source"] == "r8" else None} for n, x in enumerate(mixed, 1)],
              open(RATING_R8 / "blind-mixed-map.json", "w"), indent=1)
    json.dump([{"id": f"m{n}", "before": x["before"], "after": x["after"], "word": x["word"], "rep": x["rep"]}
               for n, x in enumerate(mixed, 1)], open(RATING_R8 / "blind-mixed.json", "w"), indent=1)
    print(json.dumps({"r8_items": len(items), "mixed": len(mixed), "per_arm": Counter(a for x in items for a in x["arms"])}, indent=1))


def score(argv):
    import csv

    from .learned_words_study import wilson

    mp = json.load(open(RATING_R8 / "blind-mixed-map.json"))
    marks = {r["id"]: r["mark"] for r in csv.DictReader(open(RATING_R8 / "blind-mixed-marks.csv"))}
    if set(marks) != {m["id"] for m in mp} or set(marks.values()) - {"good", "notsimpler", "meaning"}:
        raise SystemExit("REFUSED: marks are incomplete or use an unknown label")
    out = {"rated": len(marks)}
    for arm in "bs":
        ms = [marks[m["id"]] for m in mp if m["source"] == "r8" and arm in m["arms"]]
        c = Counter(ms)
        out["arm_" + arm] = {"swaps": len(ms), **c, "good_rate": c["good"] / max(1, len(ms)),
                             "good_ci95": wilson(c["good"], len(ms)), "meaning_rate": c["meaning"] / max(1, len(ms))}
    b, s = out["arm_b"], out["arm_s"]
    out["keep_share"] = s["swaps"] / max(1, b["swaps"])
    out["success"] = bool(s["good_rate"] >= 0.78 and out["keep_share"] >= 0.60 and s["meaning_rate"] <= 0.15)
    agree = [(m["old_mark"], marks[m["id"]]) for m in mp if m["source"] == "old"]
    out["consistency"] = {"items": len(agree), "agree": sum(a == x for a, x in agree)}
    (REPORT / "round8.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv):
    cmd = {"features": features, "train": train, "apply": apply, "page": page, "score": score}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words stacker features|train|apply|page|score")
