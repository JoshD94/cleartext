"""Ensemble v5: v4 with the context block retrained on SWORDS dev+test (round 9), the detector refit on CWI train+dev,
and part-of-speech decision features (round 10).

Chosen on all of BenchLS as dev (10-fold x3 grouped, net and log-loss agree). TSAR test is report-only.
"""

import sys, json, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.data import ROOT, tsar_data
from cleartext.features import parse, target_token
import cleartext.refine as R
from cleartext.ensemble_pipeline import EnsembleClearText

RUN_DIR = ROOT / "runs/ensemble-v5-20260927"
THRESHOLD = 0.40  # round 10 (with part-of-speech features)
names, S = R.tables("v5")
R.SPECIFICITY = True
X, y, _, _ = R.flat_specific(S["benchls_dev"] + S["benchls_holdout"])
model = R.AverageDecision().fit(X, y)
pickle.dump(model, open(RUN_DIR / "decision.pkl", "wb"))
# stacker, checker (SWORDS dev+test) and detector (CWI train+dev) were written by train_fit_v5.py and train_detector_final.py.
config = json.loads((ROOT / "runs/ensemble-v4-20260927/config.json").read_text())
config.update(
    max_word_edits=1,
    meaning_floor=0.1,
    noun_threshold=0.8,
    hardness_alpha=0.5,
    sentence_model="runs/ensemble-v5-20260927/sentence_model.pkl",
    checker=str((RUN_DIR / "checker.pkl").relative_to(ROOT)),
    detector=str((RUN_DIR / "detector.pkl").relative_to(ROOT)),
    decision=str((RUN_DIR / "decision.pkl").relative_to(ROOT)),
    decision_features="specific",
    decision_threshold=THRESHOLD,
)
(RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))
Xt, _, _, kt = R.flat_specific(S["tsar_test"])
table = R.evaluate(
    S["tsar_test"],
    R.accept_rule(dict(zip(kt, model.predict_proba(Xt)[:, 1])), THRESHOLD),
)
pipe = EnsembleClearText.load(RUN_DIR)
rows = tsar_data("test")
docs = parse([r["text"] for r in rows], "tsar-test")
e = c = 0
out = []
for r, d in zip(rows, docs):
    sel = pipe.rank_word(d, target_token(d, r["target"]))["selected"]
    if sel:
        e += 1
        c += sel["word"].lower() in r["gold"]
    out.append(
        {
            "target": r["target"],
            "replacement": sel["word"] if sel else None,
            "correct": bool(sel and sel["word"].lower() in r["gold"]),
        }
    )
print(
    "TABLE TSAR",
    json.dumps({k: table[k] for k in ["edits", "correct", "precision", "net"]}),
)
print(
    "PIPELINE TSAR",
    json.dumps(
        {"edits": e, "correct": c, "precision": round(c / e, 3), "net": 2 * c - e}
    ),
)
examples = [
    r["original"]
    for r in json.loads(
        (ROOT / "runs/initial-20260924/basic_examples_20.json").read_text()
    )
]
examples += [
    "The medication alleviates pain but can cause drowsiness.",
    "Heavy rainfall may exacerbate the flooding.",
    "She deposited the money at the bank.",
    "The gene encodes a protein that regulates growth.",
    "It was an auspicious start to the season.",
]
samples = [pipe.analyze(s, structure=False, phrases=False) for s in examples]
for s in samples:
    print("SAMPLE", ("* " if s["edits"] else "  ") + s["output"], flush=True)
json.dump(
    {
        "tsar_table": table,
        "tsar_pipeline": {"edits": e, "correct": c},
        "tsar_outputs": out,
        "samples": [
            {"original": s["original"], "output": s["output"]} for s in samples
        ],
        "threshold": THRESHOLD,
    },
    open(RUN_DIR / "evaluation.json", "w"),
    indent=2,
    default=float,
)
