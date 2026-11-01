"""Replay the saved evaluator CWI baseline without copying raw data into Git."""

import argparse
import json, sys, pathlib, hashlib

r = pathlib.Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", type=pathlib.Path, required=True)
args = ap.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(r / "evals"))
import detection_baseline as b

weights = json.loads((r / "evals/trained_weights.json").read_text())["word_weights"]
scorer, desc = b.make_scorer("linear", weights)
rows = {
    s: b.load_cwi2018(r / "models/data/raw" / f"cwi_english_{n}.tsv")
    for s, n in [("train", "Train"), ("dev", "Dev")]
}
scores = {s: b.score_rows(rs, scorer) for s, rs in rows.items()}
threshold = 0.5
best = -1
t = 0.10
while t <= 0.90:
    candidate = round(t, 2)
    f = b.f1_at(scores["train"], candidate)[2]
    if f > best:
        threshold, best = candidate, f
    t += 0.05
out = {
    "word_model": "linear",
    "threshold_tuned_on": "cwi2018_en_train",
    "threshold": threshold,
    "train_n": len(rows["train"]),
    "dev_n": len(rows["dev"]),
    **{
        s: dict(zip(("precision", "recall", "f1"), b.f1_at(sc, threshold)))
        for s, sc in scores.items()
    },
}
old = json.loads((r / "evals/reports/20261021-v7/detection_baseline.json").read_text())
assert all(out[k] == old[k] for k in ["threshold", "train_n", "dev_n", "train", "dev"])
out["matches_archived_metrics_exactly"] = True
out["artifacts"] = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [
        pathlib.Path(__file__),
        r / "evals/trained_weights.json",
        r / "evals/detection_baseline.py",
        r / "evals/evaluate_base.py",
        r / "models/data/raw/cwi_english_Train.tsv",
        r / "models/data/raw/cwi_english_Dev.tsv",
    ]
}
args.out.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
