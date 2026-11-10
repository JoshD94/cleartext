"""Ensemble v7: v5 plus BERT. The meaning model mixes in BERT's nearest sense vector (train_bert_senses.py) and the fit
stacker adds BERT slot scores (train_fit_v7.py). Decision model and protocol as v5; threshold from refine_bert_decision.py."""

import sys, json, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.data import ROOT, tsar_data
from cleartext.features import parse, target_token
import cleartext.refine as R
from cleartext.ensemble_pipeline import EnsembleClearText

RUN_DIR = ROOT / "runs/ensemble-v7-20260928"
THRESHOLD = float(
    json.loads((RUN_DIR / "refine_bert_decision.json").read_text())["v7"]["threshold"]
)
names, S = R.tables("v7")
R.SPECIFICITY = True
X, y, _, _ = R.flat_specific(S["benchls_dev"] + S["benchls_holdout"])
model = R.AverageDecision().fit(X, y)
pickle.dump(model, open(RUN_DIR / "decision.pkl", "wb"))
config = json.loads((RUN_DIR / "config.json").read_text())  # written by train_fit_v7.py
config.update(
    decision=str((RUN_DIR / "decision.pkl").relative_to(ROOT)),
    decision_features="specific",
    decision_threshold=THRESHOLD,
)
(RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))
Xt, _, _, kt = R.flat_specific(S["tsar_test"])
P = dict(zip(kt, model.predict_proba(Xt)[:, 1]))
SENSE = names.index("sense_fit")
FLOOR = config.get("meaning_floor", 0.0)


def floor_rule(
    case,
):  # same rule as the pipeline: threshold, safety checks, meaning floor
    ok = [
        c
        for c in case["candidates"]
        if c["guard"]
        and P[case["id"], c["word"]] >= THRESHOLD
        and c["x"][SENSE] >= FLOOR
    ]
    return max(ok, key=lambda c: P[case["id"], c["word"]]) if ok else None


table = R.evaluate(S["tsar_test"], floor_rule)
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
    "Although this sentence is convoluted, the main idea is simple.",
    "The judge handed down a harsh sentence for the theft.",
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
