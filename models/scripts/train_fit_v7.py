"""Fit v7: the SWORDS fit stacker with BERT. The main sense model mixes in BERT (train_bert_senses.py), so the
sense_fit and candidate_sense_rank columns are recomputed; the BERT slot columns come from the cached SWORDS scores
(scripts/experiments/bert_benchmark.py, identical to cleartext.contextual.slot_scores). Grouped 5-fold CV as in v5.
Writes runs/ensemble-v7-20260928/{stacker.pkl,config.json,fit_training.json}.
"""

import sys, json, gzip, pickle, shutil, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from cleartext.data import ROOT, RAW
from cleartext.features import parse, target_token
from cleartext.generation import resolve_target, relation_map
from cleartext import ensemble as E
from cleartext.ensemble import checker_model
from cleartext.ensemble_pipeline import build_senses

V5 = ROOT / "runs/ensemble-v5-20260927"
V7 = ROOT / "runs/ensemble-v7-20260928"
started = time.time()
bs = json.loads((V7 / "bert_sense.json").read_text())
config = json.loads((V5 / "config.json").read_text())
config.update(
    bert_sense={
        "vectors": str((V7 / "bert_senses.npz").relative_to(ROOT)),
        "weight": bs["weight"],
        "temperature": bs["temperature"],
    },
    bert_fit=True,
    checker=str((V7 / "checker.pkl").relative_to(ROOT)),
)
sense, _ = build_senses(pickle.load(open(ROOT / config["sense_model"], "rb")), config)
names = json.loads((ROOT / "runs/ensemble-v2-20260927/fit_selection.json").read_text())[
    "members"
]
recompute = {
    "sense_fit": E.SenseFit(sense),
    "candidate_sense_rank": E.CandidateSenseRank(sense),
}
cols = [names.index(n) for n in recompute]


def sense_columns(split):
    path = ROOT / f"data/cache/ensemble-v7-swords-{split}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    c = pickle.load(open(ROOT / f"data/cache/ensemble-v2-swords-{split}.pkl", "rb"))
    _, _, _, records, _ = pickle.load(
        open(ROOT / f"data/cache/swords-features-{split}.pkl", "rb")
    )
    d = json.load(gzip.open(RAW / f"swords_{split}.json.gz", "rt"))
    ids = list(d["contexts"])
    docs = parse([d["contexts"][i]["context"] for i in ids], f"swords-{split}")
    byid = dict(zip(ids, docs))
    bytarget = {}
    for i in np.flatnonzero(c["keep"]):
        bytarget.setdefault(d["substitutes"][records[i]["id"]]["target_id"], []).append(
            i
        )
    X = c["X"].copy()
    for k, (tid, idx) in enumerate(bytarget.items()):
        t = d["targets"][tid]
        doc = byid[t["context_id"]]
        tok = target_token(doc, t["target"], t["offset"])
        tg = resolve_target(tok)
        rel = (
            {a.lower(): b for a, b in relation_map(tg[0], tg[1]).items()} if tg else {}
        )
        cands = []
        for i in idx:
            sub = d["substitutes"][records[i]["id"]]["substitute"]
            key = sub.lower().replace(" ", "_")
            lemma = (wn.morphy(key, tg[1]) if tg else None) or key
            hit = rel.get(key) or rel.get(lemma.lower())
            cands.append(
                {
                    "word": sub,
                    "lemma": lemma.replace("_", " "),
                    "senses": list(hit[1]) if hit else [],
                    "source": hit[0] if hit else None,
                }
            )
        slot = E.Slot(doc, tok, target=tg)
        for j, m in zip(cols, recompute.values()):
            X[idx, j] = m.score(slot, cands)
        if k % 200 == 0:
            print(
                "SWORDS",
                split,
                k,
                "of",
                len(bytarget),
                round(time.time() - started),
                "s",
                flush=True,
            )
    out = {**c, "X": X}
    pickle.dump(out, open(path, "wb"))
    return out


d = sense_columns("dev")
t = sense_columns("test")
k = t["keep"]
s = pickle.load(open(ROOT / "data/cache/bert-swords-scores-v1.pkl", "rb"))
bslot = np.r_[s["dev"]["score"], s["test"]["score"][k]]
borig = np.r_[s["dev"]["original"], s["test"]["original"][k]]
X = np.r_[d["X"], t["X"][k]]
X = np.c_[X, bslot, bslot - borig]
y = np.r_[d["y"], t["y"][k]]
g = np.r_[d["groups"], t["groups"][k]]
gen = np.r_[d["generated"], t["generated"][k]]
rows = list(d["rows"]) + [t["rows"][i] for i in np.flatnonzero(k)]
CHECK = names.index("swords_checker")
C = json.loads((ROOT / "runs/ensemble-v2-20260927/config.json").read_text())[
    "stacker_C"
]
folds = list(GroupKFold(n_splits=5).split(X, y, g))
for a, b in folds:
    X[b, CHECK] = (
        checker_model()
        .fit([rows[i] for i in a], y[a])
        .predict_proba([rows[i] for i in b])[:, 1]
    )
res = {}
# v5 reference (same pool and folds): AUC 0.7789. Compare BERT meaning alone, then meaning plus slot columns.
for label, M in [("BERT meaning only", X[:, :-2]), ("BERT meaning and slot", X)]:
    p = np.zeros(len(y))
    for a, b in folds:
        p[b] = E.StackedFit(None, C).fit(M[a], y[a]).proba(M[b])
    res[label] = {
        "auc_generated": float(roc_auc_score(y[gen], p[gen])),
        "log_loss_generated": float(log_loss(y[gen], np.clip(p[gen], 1e-6, 1 - 1e-6))),
    }
    print("FIT", label, json.dumps(res[label]), flush=True)
shutil.copy(V5 / "checker.pkl", V7 / "checker.pkl")
stacker = E.StackedFit(None, C).fit(X, y)
pickle.dump(stacker, open(V7 / "stacker.pkl", "wb"))
(V7 / "config.json").write_text(json.dumps(config, indent=2))
json.dump(
    {
        "comparison": res,
        "reference_v5_auc": 0.7789,
        "weights": dict(
            zip(
                names + ["bert_slot", "bert_slot_relative"],
                map(float, stacker.model.coef_[0]),
            )
        ),
    },
    open(V7 / "fit_training.json", "w"),
    indent=2,
)
print("COMPLETE", round(time.time() - started), "s", flush=True)
