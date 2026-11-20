"""Round 6 offline replay set: blind-rated swaps from rounds 3-5 (validation only).

  learned-words replay build

Joins each round's blind marks to its swap key by sentence and word pair.
Each swap keeps the mark from the round that generated it (round 3's own
rater for round-3 swaps, and so on). Round-2 swaps, Joshua's 60 marked
swaps and the human-gold controls are left out. The replay set is never
used to fit a model or choose a threshold. It only decides whether a
pre-registered arm is worth a fresh window (see ROUND6.md).
Output: replay/items.json in the state directory (raw text stays local).
"""

import csv
import json
from collections import Counter

from .learned_words import STATE

ROUNDS = {"r3": "rating-r3", "r4": "rating-r4", "r5": "rating-r5"}


def build(argv):
    out, seen = [], set()
    for rnd, d in ROUNDS.items():
        keys = json.load(open(STATE / d / "key.json"))["swaps"]
        by_text = {(k["before"], k["word"], k["rep"]): k for k in keys}
        items = {i["id"]: i for i in json.load(open(STATE / d / "blind-mixed.json"))}
        marks = {r["id"]: r for r in csv.DictReader(open(STATE / d / "blind-mixed-marks.csv"))}
        for iid, it in items.items():
            k = by_text.get((it["before"], it["word"], it["rep"]))
            if k is None or iid not in marks:
                continue  # an earlier round's swap or a control
            sig = (k["before"], k["word"], k["rep"])
            if sig in seen:
                continue
            seen.add(sig)
            out.append({"round": rnd, "pid": k["pid"], "before": k["before"], "after": k["after"],
                        "word": k["word"], "rep": k["rep"], "pos": k.get("pos"), "arms": k.get("arms", []),
                        "margin": k.get("margin"), "check": k.get("check"), "gate": k.get("gate"),
                        "domain": k.get("domain"), "mark": marks[iid]["mark"]})
    (STATE / "replay").mkdir(exist_ok=True)
    json.dump(out, open(STATE / "replay/items.json", "w"), indent=1)
    stage = [o for o in out if "b" in o["arms"]]
    print(json.dumps({"items": len(out), "by_round": Counter(o["round"] for o in out),
                      "stage_b": len(stage), "stage_b_marks": Counter(o["mark"] for o in stage)}, indent=1))
    print("COMPLETE")


def main(argv):
    if argv[:1] == ["build"]:
        return build(argv[1:])
    raise SystemExit("usage: learned-words replay build")
