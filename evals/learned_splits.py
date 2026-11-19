"""Sentence-split checker study (stage 4b).

  learned-words splits featurize [--max-seconds N]
      WikiSplit train/validation/test pairs (human splits + labelled
      synthetic corruptions) and every F2 dev split edit, checkpointed
  learned-words splits fit
      fit on train, threshold on validation, report test by corruption type,
      score F2's dev splits and write a pre-commitment for the 60 F2 splits
      on Joshua's first rating page (part 2, still unmarked) — keys and
      decisions only, so his marks can score the checker later

Threshold rule (fixed here): the highest validation recall of human splits
whose specificity on corrupted splits is >= 0.90.
"""

import difflib
import glob
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .learned_words import REPORT, STATE, read_jsonl

WIKISPLIT = Path.home() / ".local/state/cleartext/structure-expanded-20261026/wikisplit-cache"
F2_OUTPUTS = Path.home() / ".local/state/cleartext/swap-veto-20261118/dev/outputs"
PANEL = Path.home() / ".local/state/cleartext/testset-v3-20261116/panel/dev.json"
HUMAN_KEY = Path.home() / ".local/state/cleartext/human-rating-20261118/key.json"
SPLIT_STATE = STATE / "splits"
SIZES = {"train": ("tune.tsv", 2500), "validation": ("validation.tsv", 800), "test": ("test.tsv", 800)}
MIN_SPECIFICITY = 0.90


def sent(t):
    return [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"(])', t.strip()) if s]


def f2_dev_splits():
    """Every sentence-level change F2 makes on the dev passages (same unit and
    key as the November 18 rating page)."""
    panel = {r["id"]: r["original"] for r in json.load(open(PANEL))}
    out = []
    for f in sorted(glob.glob(str(F2_OUTPUTS / "*/*.json"))):
        d = json.load(open(f))
        c = d["identity"]["config"]
        if not (c.startswith("words=off") and '"t5.min_entailment":0.5' in c and d["edits"]):
            continue
        pid = d["identity"]["id"]
        a, b = sent(panel[pid]), sent(d["output"])
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
            if tag == "equal":
                continue
            bef, aft = " ".join(a[i1:i2]), b[j1:j2]
            if not bef or not aft:
                continue
            key = hashlib.sha256(f"{pid}\n{bef}".encode()).hexdigest()[:16]
            out.append({"pid": pid, "source": bef, "parts": aft, "key": key, "kind": "f2", "label": -1})
    return out


def pairs():
    from cleartext.learned_words import splits as SP

    out = {}
    for name, (fname, n) in SIZES.items():
        rows = SP.wikisplit(WIKISPLIT / fname, limit=n)
        out[name] = SP.build_pairs(rows, seed=4701 + len(name))
    out["f2_dev"] = f2_dev_splits()
    return out


def featurize(argv):
    import torch

    from cleartext.learned_words import splits as SP

    torch.set_num_threads(2)
    max_seconds = float(argv[argv.index("--max-seconds") + 1]) if "--max-seconds" in argv else 1500
    SPLIT_STATE.mkdir(parents=True, exist_ok=True)
    path = SPLIT_STATE / "features.jsonl"
    done = {(r["set"], r["i"]) for r in read_jsonl(path)}
    deadline = time.monotonic() + max_seconds
    todo = [(name, i, p) for name, ps in pairs().items() for i, p in enumerate(ps) if (name, i) not in done]
    print(f"{len(done)} done, {len(todo)} to do", flush=True)
    with open(path, "a") as f:
        for k in range(0, len(todo), 64):
            if time.monotonic() > deadline:
                break
            chunk = todo[k:k + 64]
            feats = SP.features([p for _, _, p in chunk])
            for (name, i, p), x in zip(chunk, feats):
                f.write(json.dumps({"set": name, "i": i, "label": p["label"], "kind": p["kind"],
                                    "key": p.get("key"), "x": {a: round(b, 5) for a, b in x.items()}}) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {k + len(chunk)}/{len(todo)}", flush=True)
    left = len({(n, i) for n, i, _ in todo} - {(r["set"], r["i"]) for r in read_jsonl(path)})
    print("COMPLETE" if left == 0 else f"PARTIAL {left} left", flush=True)


def fit(argv):
    import pickle

    from sklearn.metrics import roc_auc_score

    from cleartext.learned_words.splits import SplitChecker

    rows = read_jsonl(SPLIT_STATE / "features.jsonl")
    by = {s: [r for r in rows if r["set"] == s] for s in ("train", "validation", "test", "f2_dev")}
    m = SplitChecker().fit([r["x"] for r in by["train"]], [r["label"] for r in by["train"]])
    pv = m.predict([r["x"] for r in by["validation"]])
    yv = np.array([r["label"] for r in by["validation"]])
    best = None
    for t in np.arange(0.05, 0.96, 0.01):
        spec = float((pv[yv == 0] < t).mean())
        rec = float((pv[yv == 1] >= t).mean())
        if spec >= MIN_SPECIFICITY and (best is None or rec > best[1]):
            best = (float(round(t, 2)), rec, spec)
    m.threshold = best[0]
    pt = m.predict([r["x"] for r in by["test"]])
    yt = np.array([r["label"] for r in by["test"]])
    kinds = np.array([r["kind"] for r in by["test"]])
    test = {"auc": float(roc_auc_score(yt, pt)), "human_split_acceptance": float((pt[yt == 1] >= m.threshold).mean()),
            "rejection_by_kind": {k: float((pt[kinds == k] < m.threshold).mean()) for k in sorted(set(kinds) - {"human"})},
            "n": dict(Counter(kinds.tolist()))}
    # NLI-only reference (the small-NLI floor F2 uses per split is 0.5).
    nli = np.array([r["x"]["nli_part_min"] for r in by["test"]])
    test["nli_part_min_auc"] = float(roc_auc_score(yt, nli))
    test["nli_floor_05_rejection_by_kind"] = {k: float((nli[kinds == k] < 0.5).mean()) for k in sorted(set(kinds) - {"human"})}
    pf = m.predict([r["x"] for r in by["f2_dev"]])
    f2 = {"splits": len(pf), "rejected": int((pf < m.threshold).sum())}
    human = json.load(open(HUMAN_KEY))["splits"]
    score = {r["key"]: float(p) for r, p in zip(by["f2_dev"], pf)}
    pre = [{"key": h["key"], "score": round(score[h["key"]], 4), "accept": score[h["key"]] >= m.threshold}
           for h in human if h["key"] in score]
    commit = {"note": "Split-checker decisions on the 60 F2 splits of rate-edits.html part 2, written before "
                      "Joshua's marks on them exist. Score with evals/learned_splits.py score-marks.",
              "threshold": m.threshold, "n": len(pre), "rejected": sum(not p["accept"] for p in pre),
              "sha256": hashlib.sha256(json.dumps(pre, sort_keys=True).encode()).hexdigest(), "decisions": pre}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "split-checker-precommit.json").write_text(json.dumps(commit, indent=1))
    res = {"threshold_rule": f"max validation recall with specificity >= {MIN_SPECIFICITY}",
           "threshold": best[0], "validation_recall": best[1], "validation_specificity": best[2],
           "train_pairs": len(by["train"]), "test": test, "f2_dev": f2,
           "precommit_60": {"n": len(pre), "rejected": commit["rejected"], "sha256": commit["sha256"]}}
    (REPORT / "split-checker.json").write_text(json.dumps(res, indent=1))
    with open(STATE / "split-checker.pkl", "wb") as f:
        pickle.dump(m, f)
    json.dump({r["key"]: float(p) for r, p in zip(by["f2_dev"], pf)}, open(SPLIT_STATE / "f2-dev-scores.json", "w"))
    print(json.dumps(res, indent=1), flush=True)
    print("COMPLETE", flush=True)


def score_marks(argv):
    """After Joshua marks part 2: precision of the checker's accept decisions."""
    import csv

    path = Path(argv[0]) if argv else Path.home() / ".local/state/cleartext/human-rating-20261118/cleartext-marks.csv"
    marks = {r["key"]: r["mark"] for r in csv.DictReader(open(path)) if r["id"].startswith("s") and r["mark"]}
    pre = json.load(open(REPORT / "split-checker-precommit.json"))["decisions"]
    rows = [(p["accept"], marks[p["key"]]) for p in pre if p["key"] in marks]
    acc = [m for a, m in rows if a]
    rej = [m for a, m in rows if not a]
    out = {"marked": len(rows), "all_good_rate": float(np.mean([m == "good" for _, m in rows])) if rows else None,
           "accepted": len(acc), "accepted_good_rate": float(np.mean([m == "good" for m in acc])) if acc else None,
           "rejected": len(rej), "rejected_marks": dict(Counter(rej))}
    print(json.dumps(out, indent=1))
    return out


def main(argv):
    cmd, rest = argv[0], argv[1:]
    return {"featurize": featurize, "fit": fit, "score-marks": score_marks}[cmd](rest)
