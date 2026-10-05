"""Compare the BERT side benchmark (bert_benchmark.py) with our current models. Main environment.

Fit: AUC of BERT's slot score alone, and our stacked fit with and without BERT columns, using the same pool and
grouped 5-fold CV as train_fit_v5.py (cross-fitted checker column). Meaning: accuracy on the same 1,500 held-out
SemCor annotations for our sense ensemble and for BERT's nearest sense vector.
"""

import sys, json, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from cleartext.data import ROOT
from cleartext import ensemble as E
from cleartext.ensemble import checker_model
from cleartext.ensemble_pipeline import build_sense, LATEST

CACHE = ROOT / "data/cache"
out = {}

# ---- Meaning ----
b = pickle.load(open(CACHE / "bert-semcor-v1.pkl", "rb"))
config = json.loads((LATEST / "config.json").read_text())
sense = build_sense(pickle.load(open(ROOT / config["sense_model"], "rb")), config)
data, _ = pickle.load(open(CACHE / "semcor-records-v1.pkl", "rb"))
byid = {(r["document"], r["sentence"], r["index"]): r for r in data["test"]}
ours = bert = agree = 0
rows = []
for p in b["preds"]:
    r = byid[p["id"]]
    dist = sense.distribution(r["words"], r["index"], r["lemma"], r["pos"])
    o = max(dist, key=dist.get) if dist else None
    bb = max(p["sims"], key=p["sims"].get)
    ours += o == p["gold"]
    bert += bb == p["gold"]
    agree += o == bb
    rows.append(
        {
            "ours": o == p["gold"],
            "bert": bb == p["gold"],
            "gold_seen": p["counts"].get(p["gold"], 0) > 0,
        }
    )
n = len(rows)
seen = [x for x in rows if x["gold_seen"]]
unseen = [x for x in rows if not x["gold_seen"]]
out["meaning"] = {
    "n": n,
    "ours": ours / n,
    "bert": bert / n,
    "agree": agree / n,
    "gold_sense_seen_in_training": {
        "n": len(seen),
        "ours": float(np.mean([x["ours"] for x in seen])),
        "bert": float(np.mean([x["bert"] for x in seen])),
    },
    "gold_sense_never_seen": {
        "n": len(unseen),
        "ours": float(np.mean([x["ours"] for x in unseen])) if unseen else None,
        "bert": float(np.mean([x["bert"] for x in unseen])) if unseen else None,
    },
}
print("MEANING", json.dumps(out["meaning"]), flush=True)

# ---- Fit ----
s = pickle.load(open(CACHE / "bert-swords-scores-v1.pkl", "rb"))
d = pickle.load(open(CACHE / "ensemble-v2-swords-dev.pkl", "rb"))
t = pickle.load(open(CACHE / "ensemble-v2-swords-test.pkl", "rb"))
k = t["keep"]
X = np.r_[d["X"], t["X"][k]]
y = np.r_[d["y"], t["y"][k]]
g = np.r_[d["groups"], t["groups"][k]]
gen = np.r_[d["generated"], t["generated"][k]]
rows_ = list(d["rows"]) + [t["rows"][i] for i in np.flatnonzero(k)]
bs = np.r_[s["dev"]["score"], s["test"]["score"][k]]
bo = np.r_[s["dev"]["original"], s["test"]["original"][k]]
ok = ~np.isnan(bs)
assert ok.all(), int((~ok).sum())
out["fit_bert_alone"] = {
    "auc_generated_score": float(roc_auc_score(y[gen], bs[gen])),
    "auc_generated_relative": float(roc_auc_score(y[gen], (bs - bo)[gen])),
}
print("BERT ALONE", json.dumps(out["fit_bert_alone"]), flush=True)
names = json.loads((ROOT / "runs/ensemble-v2-20260927/fit_selection.json").read_text())[
    "members"
]
CHECK = names.index("swords_checker")
C = json.loads((ROOT / "runs/ensemble-v2-20260927/config.json").read_text())[
    "stacker_C"
]
folds = list(GroupKFold(n_splits=5).split(X, y, g))
Xf = X.copy()
for a, bb in folds:
    Xf[bb, CHECK] = (
        checker_model()
        .fit([rows_[i] for i in a], y[a])
        .predict_proba([rows_[i] for i in bb])[:, 1]
    )
for label, M in [("current fit", Xf), ("+ BERT slot scores", np.c_[Xf, bs, bs - bo])]:
    p = np.zeros(len(y))
    for a, bb in folds:
        p[bb] = E.StackedFit(None, C).fit(M[a], y[a]).proba(M[bb])
    out[label] = {
        "auc_generated": float(roc_auc_score(y[gen], p[gen])),
        "log_loss_generated": float(log_loss(y[gen], np.clip(p[gen], 1e-6, 1 - 1e-6))),
    }
    print("FIT", label, json.dumps(out[label]), flush=True)
json.dump(
    out, open(ROOT / "runs/ensemble-v5-20260927/bert_compare.json", "w"), indent=2
)
