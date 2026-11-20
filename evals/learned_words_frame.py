"""Round 5 probe: do neighbour-surprisal features separate the rater's meaning errors?

  learned-words frame probe

Diagnostic only (it reads blind Claude marks on rated round-3/4 swaps, so
nothing here may set a threshold or train a model). Prints the AUC of each
frame feature for "changes meaning" vs "good", and vs all other marks, next
to the stage-4 checker score. A trained version must use public data and be
tested on a fresh slice.
"""

import csv
import json

import numpy as np

from .learned_words import REPORT, STATE


def rated():
    out = []
    mp = {m["id"]: m for m in json.load(open(STATE / "rating-r4/blind-mixed-map.json"))}
    marks = {r["id"]: r["mark"] for r in csv.DictReader(open(STATE / "rating-r4/blind-mixed-marks.csv"))}
    keys = {r: {f"w{n}": x for n, x in enumerate(json.load(open(STATE / f"rating-{r}/key.json"))["swaps"], 1)}
            for r in ("r3", "r4")}
    for i, m in mp.items():
        out.append((keys[m["round"]][m["src"]], marks[i]))
    return out


def probe(argv):
    from sklearn.metrics import roc_auc_score

    from cleartext.learned_words.frame import frame_features, swap_span

    rows = []
    for x, mark in rated():
        s, eb, ea = swap_span(x["before"], x["after"], x["word"], x["rep"])
        f = frame_features(x["before"], x["after"], s, eb, ea)
        f.update(check=x["check"], margin=x["margin"], gate=x["gate"])
        rows.append((f, mark))
    keys = sorted(rows[0][0])
    res = {"n": len(rows), "marks": {m: sum(r[1] == m for r in rows) for m in ("good", "notsimpler", "meaning")}}
    for k in keys:
        v = np.array([r[0][k] for r in rows])
        meaning = np.array([r[1] == "meaning" for r in rows])
        gm = np.array([r[1] in ("good", "meaning") for r in rows])
        res[k] = {"auc_meaning_vs_good": float(roc_auc_score(meaning[gm], v[gm])),
                  "auc_meaning_vs_rest": float(roc_auc_score(meaning, v)),
                  "mean_good": float(v[np.array([r[1] == "good" for r in rows])].mean()),
                  "mean_meaning": float(v[meaning].mean())}
    (REPORT / "frame-probe.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("COMPLETE", flush=True)


def main(argv):
    if argv[:1] == ["probe"]:
        return probe(argv[1:])
    raise SystemExit("usage: learned-words frame probe")
