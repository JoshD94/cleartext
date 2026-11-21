"""Round 6, arm p: pairwise simplicity model (see ROUND6.md).

  learned-words pairwise train    fit, out-of-fold threshold, write pairwise-precommit.json
  learned-words pairwise check    public tests (TSAR test gold, SWORDS test synonyms), report only
  learned-words pairwise replay   go/no-go on the round 3-5 replay set (after the pre-commit)
"""

import hashlib
import json
import pickle
import re
import time
import zipfile
from collections import Counter

import numpy as np

from .learned_words import REPORT, STATE

MODEL = STATE / "pairwise.pkl"
PRECOMMIT = REPORT / "pairwise-precommit.json"
SOURCES = ["models/src/cleartext/learned_words/pairwise.py", "evals/learned_words_pairwise.py"]


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _ok(w):
    return re.fullmatch(r"[a-z]{2,}", w) is not None


def training_pairs():
    from cleartext.learned_words import data as D

    nns = {(D.norm(s), t.lower()) for s, t, _, _ in D.nnseval_lines()}
    text = zipfile.ZipFile(D.V7_RAW / "BenchLS.zip").read("BenchLS/BenchLS.txt").decode("utf8")
    pairs, y, groups = [], [], []
    seen = set()

    def add(a, b, label, g):
        if a != b and _ok(a) and _ok(b) and (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
            y.append(label)
            groups.append(g)

    for line in text.splitlines():
        f = line.split("\t")
        if len(f) < 4 or (D.norm(f[0]), f[1].lower()) in nns:
            continue
        target = f[1].strip().lower()
        ranked = [(int(x.split(":", 1)[0]), x.split(":", 1)[1].strip().lower()) for x in f[3:] if ":" in x]
        ranked = [(r, w) for r, w in ranked if _ok(w)]
        if not ranked or not _ok(target):
            continue
        g = target
        top = [w for r, w in ranked if r == min(r for r, _ in ranked)]
        if len(top) == 1:
            add(target, top[0], 1, g)
            add(top[0], target, 0, g)
        for ra, a in ranked:
            for rb, b in ranked:
                if a != b:
                    add(a, b, int(rb < ra), g)
    for hard, says in D.plain_pairs().items():
        for s in says:
            if _ok(s):
                add(hard, s, 1, hard)
                add(s, hard, 0, hard)
    return pairs, np.array(y), groups


def train(argv):
    from sklearn.model_selection import GroupKFold

    from cleartext.learned_words.pairwise import PairSimpler

    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    difficulty = pickle.load(open(STATE / "simpler.pkl", "rb"))
    pairs, y, groups = training_pairs()
    print(f"{len(pairs)} pairs, {int(y.sum())} positive", flush=True)
    oof = np.zeros(len(pairs))
    gidx = {g: i for i, g in enumerate(sorted(set(groups)))}
    gnum = np.array([gidx[g] for g in groups])
    for k, (tr, te) in enumerate(GroupKFold(5).split(pairs, y, gnum)):
        m = PairSimpler().fit([pairs[i] for i in tr], y[tr], difficulty)
        oof[te] = m.prob([pairs[i] for i in te])
        print(f"fold {k} done", flush=True)
    from sklearn.metrics import roc_auc_score

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    table = [{"t": float(t), "pos_pass": float((oof[y == 1] >= t).mean()), "neg_pass": float((oof[y == 0] >= t).mean())}
             for t in grid]
    for r in table:
        r["balanced_acc"] = (r["pos_pass"] + 1 - r["neg_pass"]) / 2
    best = max(table, key=lambda r: (r["balanced_acc"], -r["t"]))
    model = PairSimpler().fit(pairs, y, difficulty)
    pickle.dump(model, open(MODEL, "wb"))
    pre = {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "rule": "max out-of-fold balanced accuracy (ROUND6.md)",
           "threshold": best["t"], "oof": best, "oof_auc": float(roc_auc_score(y, oof)),
           "pairs": len(pairs), "positives": int(y.sum()),
           "model_sha256": _sha(MODEL), "source_sha256": {p: _sha(p) for p in SOURCES},
           "simpler_pkl_sha256": _sha(STATE / "simpler.pkl")}
    json.dump(pre, open(PRECOMMIT, "w"), indent=1)
    print(json.dumps(pre, indent=1))
    print("COMPLETE")


def load():
    pre = json.load(open(PRECOMMIT))
    if _sha(MODEL) != pre["model_sha256"]:
        raise SystemExit("REFUSED: pairwise.pkl differs from its pre-commitment")
    return pickle.load(open(MODEL, "rb")), pre["threshold"]


def check(argv):
    from cleartext.learned_words import data as D

    model, t = load()
    out = {}
    tsar = [r for r in D.instances() if r["dataset"] == "tsar_test"]
    gold = [(r["target"].lower(), max(r["gold"], key=r["gold"].get)) for r in tsar]
    gold = [(a, b) for a, b in gold if _ok(a) and _ok(b)]
    p = model.prob(gold)
    out["tsar_test_top_gold_pass"] = {"n": len(gold), "pass": float((p >= t).mean())}
    sw = [r for r in D.instances() if r["dataset"] == "swords_test"]
    syn = []
    for r in sw:
        for w, share, n in r["rated"]:
            if share >= 0.8 and n >= 3 and _ok(w.lower()):
                syn.append((r["target"].lower(), w.lower()))
    syn = sorted(set(syn))
    p = model.prob(syn)
    out["swords_test_accepted_synonyms_pass"] = {"n": len(syn), "pass": float((p >= t).mean())}
    json.dump(out, open(REPORT / "pairwise-check.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("COMPLETE")


def replay(argv):
    from sklearn.metrics import roc_auc_score

    model, t = load()
    items = [i for i in json.load(open(STATE / "replay/items.json")) if "b" in i["arms"]]
    p = model.prob([(i["word"], i["rep"]) for i in items])
    kept = [i for i, x in zip(items, p) if x >= t]
    base = Counter(i["mark"] for i in items)
    k = Counter(i["mark"] for i in kept)
    gn = [(i["mark"] == "good", x) for i, x in zip(items, p) if i["mark"] in ("good", "notsimpler")]
    res = {"threshold": t, "baseline": dict(base), "kept": dict(k), "n_base": len(items), "n_kept": len(kept),
           "good_rate_base": base["good"] / len(items), "good_rate_kept": k["good"] / max(1, len(kept)),
           "keep_share": len(kept) / len(items),
           "notsimpler_removed_share": 1 - k["notsimpler"] / max(1, base["notsimpler"]),
           "auc_good_vs_notsimpler": float(roc_auc_score([a for a, _ in gn], [b for _, b in gn]))}
    res["go"] = (res["good_rate_kept"] >= res["good_rate_base"] + 0.05 and res["keep_share"] >= 0.60
                 and res["notsimpler_removed_share"] >= 0.40)
    json.dump(res, open(REPORT / "pairwise-replay.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    for i, x in sorted(zip(items, p), key=lambda z: z[1])[:25]:
        print(f"{x:.2f} {i['mark']:10s} {i['word']} -> {i['rep']}")
    print("COMPLETE")


def main(argv):
    cmd = {"train": train, "check": check, "replay": replay}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words pairwise train|check|replay")
