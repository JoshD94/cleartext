"""Round 6, arm k: learned context/domain-fit veto (see ROUND6.md, addendum k).

  learned-words contextfit train    rows, out-of-fold threshold, write contextfit-precommit.json
  learned-words contextfit replay   go/no-go on the round 3-5 replay set (after the pre-commit)

Training rows (public data only; nothing from any rated round):
  A  SWORDS dev (human): substitutes accepted by >= 50% of >= 3 annotators
     are fits (1); those accepted by <= 10% are misfits (0).
  B  technical corpus (distant): for a word w that occurs both inside a
     multiword glossary term (NIST, US Courts, MeSH; 2-4 words) and outside
     any, with a WordNet or plainlanguage.gov candidate r that is more
     frequent than w, an in-term occurrence swapped to r is a misfit (0) and
     an out-of-term occurrence is a fit (1), matched per word. The
     sentence's own n-grams are removed from the counts.
The features are corpus statistics only (context_fit.py). No glossary flag
is a feature, so the model cannot copy the B label rule.
"""

import hashlib
import gzip
import html
import json
import pickle
import random
import re
import time
from collections import Counter, defaultdict

import numpy as np

from .learned_words import REPORT, STATE
from .learned_words_corpus import CORPUS, tokens

MODEL = STATE / "contextfit.pkl"
PRECOMMIT = REPORT / "contextfit-precommit.json"
COUNTS = CORPUS / "counts.pkl"
SOURCES = ["models/src/cleartext/learned_words/context_fit.py", "evals/learned_words_contextfit.py",
           "evals/learned_words_corpus.py"]
SEED = 20261122
KEEP_FITS = 0.90
PER_WORD = 20
SAMPLE_SENTENCES = 250000


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def multiword_terms():
    """{glossary: set of token tuples (2-4 words)}."""
    import xml.etree.ElementTree as ET

    from cleartext.learned_words import data as D

    def mw(s):
        t = tuple(tokens(html.unescape(re.sub(r"<[^>]+>|\(.*?\)", " ", s))))
        return t if 2 <= len(t) <= 4 else None

    out = {}
    nist = json.loads((D.RAW / "nist-glossary/glossary-export.json").read_text(encoding="utf-8-sig"))
    entries = nist.get("parentTerms", nist) if isinstance(nist, dict) else nist
    out["nist"] = {t for e in entries if (t := mw(str(e.get("term", ""))))}
    page = (D.RAW / "uscourts-glossary/glossary.html").read_text(encoding="utf8")
    out["uscourts"] = {t for x in re.findall(r"<dt>(.*?)</dt>", page) if (t := mw(x))}
    mesh = set()
    with gzip.open(D.RAW / "mesh/desc2026.gz") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag == "Term":
                if t := mw(el.findtext("String") or ""):
                    mesh.add(t)
                el.clear()
            elif el.tag == "DescriptorRecord":
                el.clear()
    out["mesh"] = mesh
    return out


def candidates(w, plain):
    from nltk.corpus import wordnet as wn
    from wordfreq import zipf_frequency as z

    out = set(x for x in plain.get(w, []) if " " not in x)
    for s in wn.synsets(w)[:2]:
        for lem in s.lemmas():
            n = lem.name().lower()
            if n != w and n.isalpha():
                out.add(n)
    return sorted(x for x in out if z(x, "en") > z(w, "en"))


def rows_a():
    from cleartext.learned_words import data as D
    from cleartext.learned_words.context_fit import Counts, load_counts, vector

    C = Counts(load_counts(str(COUNTS)))
    X, y, groups, src = [], [], [], []
    for r in D.swords("dev"):
        before = tokens(r["text"][:r["start"]])
        toks = tokens(r["text"])
        i = len(before)
        if i >= len(toks) or toks[i] != r["target"].lower():
            continue
        for w, share, n in r["rated"]:
            w = w.lower()
            if n < 3 or not w.isalpha() or (0.1 < share < 0.5):
                continue
            X.append(vector(C, toks, i, w))
            y.append(int(share >= 0.5))
            groups.append(f"A:{r['target'].lower()}")
            src.append("swords")
    return X, y, groups, src


def rows_b():
    from cleartext.learned_words import data as D
    from cleartext.learned_words.context_fit import Counts, load_counts, own_ngrams, vector

    terms = multiword_terms()
    index = {}
    for g in ("nist", "uscourts", "mesh"):
        for t in terms[g]:
            index.setdefault(t, g)
    plain = D.plain_pairs()
    counts = load_counts(str(COUNTS))
    rng = random.Random(SEED)
    sents = []
    for source in ("rfc", "fedreg", "openfda"):
        with gzip.open(CORPUS / f"{source}.jsonl.gz", "rt") as r:
            sents += [json.loads(line)["s"] for line in r]
    rng.shuffle(sents)
    inside, outside = defaultdict(list), defaultdict(list)
    for s in sents[:SAMPLE_SENTENCES]:
        toks = tokens(s)
        covered = {}
        for n in (2, 3, 4):
            for a in range(len(toks) - n + 1):
                g = index.get(tuple(toks[a:a + n]))
                if g:
                    for k in range(a, a + n):
                        covered.setdefault(k, g)
        for k, w in enumerate(toks):
            if len(w) < 4 or not w.isalpha():
                continue
            (inside[w] if k in covered else outside[w]).append((s, k, covered.get(k)))
    cand_cache = {}
    X, y, groups, src = [], [], [], []
    for w in sorted(set(inside) & set(outside)):
        if w not in cand_cache:
            cand_cache[w] = candidates(w, plain)
        if not cand_cache[w]:
            continue
        m = min(len(inside[w]), len(outside[w]), PER_WORD)
        for label, pool in ((0, inside[w]), (1, outside[w])):
            for s, k, g in rng.sample(pool, m):
                toks = tokens(s)
                C = Counts(counts, own_ngrams(toks))
                r = rng.choice(cand_cache[w])
                X.append(vector(C, toks, k, r))
                y.append(label)
                groups.append(f"B:{w}")
                src.append(g or "outside")
    return X, y, groups, src


def _model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, min_samples_leaf=20, random_state=4701)


def train(argv):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from cleartext.learned_words.context_fit import KEYS

    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    xa, ya, ga, sa = rows_a()
    print(f"A rows {len(ya)} ({sum(ya)} fits)", flush=True)
    xb, yb, gb, sb = rows_b()
    print(f"B rows {len(yb)} ({sum(yb)} fits) by source {Counter(sb)}", flush=True)
    X = np.array(xa + xb, float)
    y = np.array(ya + yb)
    groups = ga + gb
    src = np.array(sa + sb)
    gid = {g: i for i, g in enumerate(sorted(set(groups)))}
    gnum = np.array([gid[g] for g in groups])
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, gnum):
        oof[te] = _model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    # Leave-one-glossary-out on B: misfits from one glossary held out with
    # the fits of the same words.
    logo = {}
    isb = np.array([g.startswith("B:") for g in groups])
    for g in ("nist", "uscourts", "mesh"):
        words = {groups[k] for k in range(len(y)) if src[k] == g}
        te = np.array([isb[k] and groups[k] in words for k in range(len(y))])
        if te.sum() < 50 or len(set(y[te])) < 2:
            continue
        m = _model().fit(X[~te], y[~te])
        logo[g] = {"rows": int(te.sum()), "auc": float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))}
    # Pre-registered rule: the veto threshold keeps >= 90% of out-of-fold fits.
    t = float(np.quantile(oof[y == 1], 1 - KEEP_FITS))
    res = {"rule": f"keep >= {KEEP_FITS:.0%} of out-of-fold fits (A and B)", "threshold": t,
           "rows": {"A": len(ya), "B": len(yb)}, "fits": {"A": int(sum(ya)), "B": int(sum(yb))},
           "oof_auc": {"all": float(roc_auc_score(y, oof)),
                       "A": float(roc_auc_score(y[~isb], oof[~isb])),
                       "B": float(roc_auc_score(y[isb], oof[isb]))},
           "misfits_vetoed": {"A": float((oof[~isb & (y == 0)] < t).mean()),
                              "B": float((oof[isb & (y == 0)] < t).mean())},
           "leave_one_glossary_out": logo, "features": KEYS}
    model = _model().fit(X, y)
    pickle.dump(model, open(MODEL, "wb"))
    res |= {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "model_sha256": _sha(MODEL),
            "counts_sha256": _sha(COUNTS), "source_sha256": {p: _sha(p) for p in SOURCES}}
    json.dump(res, open(PRECOMMIT, "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("COMPLETE")


def load():
    pre = json.load(open(PRECOMMIT))
    if _sha(MODEL) != pre["model_sha256"] or _sha(COUNTS) != pre["counts_sha256"]:
        raise SystemExit("REFUSED: contextfit model or counts differ from the pre-commitment")
    return pickle.load(open(MODEL, "rb")), pre["threshold"]


def score(model, swaps):
    """swaps: dicts with before, word, rep -> P(fit)."""
    from cleartext.learned_words.context_fit import Counts, load_counts, vector

    C = Counts(load_counts(str(COUNTS)))
    X = []
    for s in swaps:
        toks = tokens(s["before"])
        w = s["word"].lower()
        idx = [k for k, x in enumerate(toks) if x == w]
        k = idx[0] if idx else 0
        if idx and s.get("after"):  # pick the occurrence the swap changed
            after = tokens(s["after"])
            for j in idx:
                if j < len(after) and after[j] == s["rep"].lower().split()[0]:
                    k = j
                    break
        X.append(vector(C, toks, k, s["rep"].lower().split()[0]))
    return model.predict_proba(np.array(X, float))[:, 1] if X else np.array([])


def replay(argv):
    from sklearn.metrics import roc_auc_score

    model, t = load()
    items = [i for i in json.load(open(STATE / "replay/items.json")) if "b" in i["arms"]]
    p = score(model, items)
    kept = [i for i, x in zip(items, p) if x >= t]
    base, k = Counter(i["mark"] for i in items), Counter(i["mark"] for i in kept)
    gm = [(i["mark"] == "good", x) for i, x in zip(items, p) if i["mark"] in ("good", "meaning")]
    res = {"threshold": t, "baseline": dict(base), "kept": dict(k), "n_base": len(items), "n_kept": len(kept),
           "good_rate_base": base["good"] / len(items), "good_rate_kept": k["good"] / max(1, len(kept)),
           "keep_share": len(kept) / len(items),
           "meaning_removed_share": 1 - k["meaning"] / max(1, base["meaning"]),
           "auc_good_vs_meaning": float(roc_auc_score([a for a, _ in gm], [b for _, b in gm]))}
    res["go"] = (res["good_rate_kept"] >= res["good_rate_base"] + 0.05 and res["keep_share"] >= 0.60
                 and res["meaning_removed_share"] >= 1 / 3)
    json.dump(res, open(REPORT / "contextfit-replay.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    for i, x in sorted(zip(items, p), key=lambda z: z[1])[:25]:
        print(f"{x:.2f} {i['mark']:10s} {i['word']} -> {i['rep']}")
    print("COMPLETE")


def main(argv):
    cmd = {"train": train, "replay": replay}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words contextfit train|replay")
