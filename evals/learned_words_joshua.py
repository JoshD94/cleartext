"""Joshua's blind marking page for the learned word stage (see JOSHUA-PAGE.md).

  learned-words joshua apply [--max-seconds N]   supplementary openFDA window (once), round 11's generators
  learned-words joshua page                      the page from round 21's Joshua window plus the supplement

Round 21's Joshua window came up short of the planned ~60 best-arm swaps: the Federal Register
corpus had only 16 unused documents (30 planned) and p50 swaps rarely on news and encyclopedic
text. The supplement adds openFDA passages under the same exclusions. No AI rater sees either
window. The key stays in the state directory; Joshua's marks are never used to train, tune or
choose anything.
"""

import hashlib
import json
import random
import time
from collections import Counter

from . import learned_words_r12 as P12
from . import learned_words_r21 as Q
from . import learned_words_r22 as Q22
from . import learned_words_r9 as R
from .learned_words import REPORT, STATE, read_jsonl

SDIR = STATE / "joshua-r21b"
SEED = 20261232
PER_SOURCE = {"openfda": 40}
PRECOMMIT = REPORT / "joshua-precommit.json"
BEST = "p50"  # round 21's pre-registered rule: q50 missed its bars, so p50
SOURCES = ("evals/learned_words_joshua.py",) + Q22.SOURCES


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def panel():
    from cleartext.learned_words.rewrite import tokens

    path = SDIR / "passages.json"
    if path.exists():
        raise SystemExit(f"REFUSED: {path} exists")
    prev = []
    for p in (R.PANEL, Q.P12.PANEL, Q.P13.PANEL, Q.P17.PANEL, Q.P18.PANEL, Q.P19.PANEL, Q.P20.PANEL,
              Q.DIR / "passages.json", Q.JDIR / "passages.json", Q22.PANEL):
        prev += json.load(open(p))
    used = [x["original"] for x in prev]
    for p in ("panel-r4", "panel-r5", "panel-r8"):
        used += [x["original"] for x in json.load(open(STATE / p / "passages.json"))]
    used += [s["before"] for s in R.dev_swaps()]
    used += [x["original"] for x in json.load(open(Q.V3_PANEL))]
    grams = P12._grams(used)
    prev_docs = {(x["domain"], x["source_doc"]) for x in prev}
    out = []
    for src, want in PER_SOURCE.items():
        docs = Q._docs(src)
        order = sorted(docs)
        rng = random.Random(f"{SEED}:{src}")
        rng.shuffle(order)
        n = 0
        for doc in order:
            if n >= want:
                break
            if (src, doc) in prev_docs:
                continue
            sents = docs[doc]
            starts = [i for i in range(len(sents) - P12.PASSAGE_SENTENCES + 1)
                      if all(R._ok_sentence(x) for x in sents[i:i + P12.PASSAGE_SENTENCES])]
            if not starts:
                continue
            i = rng.choice(starts)
            text = " ".join(sents[i:i + P12.PASSAGE_SENTENCES])
            tt = tokens(text)
            if any(tuple(tt[k:k + 8]) in grams for k in range(len(tt) - 7)):
                continue
            n += 1
            out.append({"id": f"joshua2-{src}-{n:02d}", "domain": src, "kind": "standard", "source_doc": doc,
                        "first_sentence": i, "original": text, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    SDIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({"passages": len(out), "by_source": Counter(p["domain"] for p in out)}), flush=True)


def apply(argv):
    """Round 21's generator path on the supplement (checkpointed)."""
    import pickle

    from cleartext.learned_words import senses as SN

    from . import learned_words_stacker as S
    from .learned_splits import sent
    from .learned_words_rewrite import load
    from .learned_words_simpler import SIMPLER

    pre = json.load(open(PRECOMMIT))
    if _sha(Q.PRECOMMIT) != pre["r21_precommit_sha256"]:
        raise SystemExit("REFUSED: round 21's pre-commitment changed")
    Q.load_models()
    R._threads(argv)
    deadline = time.monotonic() + R._seconds(argv, 1200)
    if not (SDIR / "passages.json").exists():
        panel()
    pan = json.load(open(SDIR / "passages.json"))
    skip = _overlap(pan)
    out = SDIR / "apply.jsonl"
    done = {r["pid"] for r in read_jsonl(out)}
    todo = [p for p in pan if p["id"] not in done and p["id"] not in skip]
    print(f"{len(done)} done, {len(todo)} to do, {len(skip)} skipped for rewrite-corpus overlap", flush=True)
    if todo:
        clf_s, _ = S.load_filter()
        sn = SN.load(R.TABLE)
        _, T, ids, _ = load()
        model = pickle.load(open(STATE / "model.pkl", "rb"))
        simpler = pickle.load(open(SIMPLER, "rb"))
    with open(out, "a") as f:
        for p in todo:
            if time.monotonic() > deadline:
                break
            variants, seen = R.word_swaps_two(p["original"], model, p["id"], sn)
            rec = {"pid": p["id"], "domain": p["domain"], "content_words": seen, "sentences": len(sent(p["original"]))}
            for name, swaps in variants.items():
                got, rows = [], []
                for s in swaps:
                    r = S.swap_row(model, simpler, s)
                    if r is not None:
                        got.append(s)
                        rows.append(r)
                if got:
                    X = Q.P13._x31(T, ids, got, rows)
                    for s, r, x, sc in zip(got, rows, X, clf_s.predict_proba(X)[:, 1]):
                        s.update(margin=r["f"]["margin"], stack=float(sc), x31=[float(v) for v in x],
                                 lemma=r["lemma"], start=r["start"])
                rec[name] = [s for s in swaps if "stack" in s]
                rec[f"unresolved_{name}"] = len(swaps) - len(rec[name])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {p['id']} all={len(rec['all'])} sense={len(rec['sense'])}", flush=True)
    left = len({p["id"] for p in pan} - skip - {r["pid"] for r in read_jsonl(out)})
    print("COMPLETE" if not left else f"PARTIAL {left} left", flush=True)


def _overlap(pan):
    from cleartext.learned_words.rewrite import tokens

    from .learned_words_rewrite import pairs

    path = SDIR / "panel-overlap.json"
    if path.exists():
        return set(json.load(open(path)))
    index = {}
    for p in pan:
        t = tokens(p["original"])
        for i in range(len(t) - 7):
            index.setdefault(tuple(t[i:i + 8]), set()).add(p["id"])
    hit = set()
    for _, _, a, c in pairs():
        for x in (a, c):
            t = tokens(x)
            for i in range(len(t) - 7):
                hit |= index.get(tuple(t[i:i + 8]), set())
    json.dump(sorted(hit), open(path, "w"))
    return hit


def page(argv):
    """Round 21's page rule over both Joshua windows: up to Q.J_BEST best-arm swaps and Q.J_B arm-b
    swaps the best arm rejects, one seeded shuffle, no scores or arm labels."""
    from .learned_words_apply import _skey, choose

    models, pre = Q.load_models()
    items = {}
    for path in (Q.JDIR / "apply.jsonl", SDIR / "apply.jsonl"):
        for r in read_jsonl(path):
            for arm, chosen in Q.picks(r, models, pre["thresholds"], pre["arms"], choose).items():
                for s in chosen:
                    k = _skey(s)
                    it = items.setdefault(k, dict({x: v for x, v in s.items() if x != "x31"}, key=k, arms=[],
                                                  domain=r["domain"], shown=Q.realized(s)))
                    if arm not in it["arms"]:
                        it["arms"].append(arm)
    items = sorted(items.values(), key=lambda x: x["key"])
    rng = random.Random(1231)
    a = [x for x in items if BEST in x["arms"]]
    rest = [x for x in items if "b" in x["arms"] and BEST not in x["arms"]]
    rng.shuffle(a)
    rng.shuffle(rest)
    chosen = [dict(x, group="best") for x in a[:Q.J_BEST]] + [dict(x, group="b_rejected") for x in rest[:Q.J_B]]
    rng.shuffle(chosen)
    for n, x in enumerate(chosen, 1):
        x["id"] = f"j{n}"
    key_path = Q.JDIR / "key.json"
    if key_path.exists() and json.load(open(key_path))["items"] != [x["key"] for x in chosen]:
        raise SystemExit("REFUSED: a different key exists")
    window = {"best_arm_swaps": len(a), "arm_b_swaps": sum("b" in x["arms"] for x in items),
              "arm_b_rejected_by_best": len(rest),
              "best_by_domain": Counter(x["domain"] for x in a), "page_by_domain": Counter(x["domain"] for x in chosen),
              "page_groups": Counter(x["group"] for x in chosen), "realized": sum(x["shown"] != x["after"] for x in chosen)}
    json.dump({"best_arm": BEST, "items": [x["key"] for x in chosen], "window": window,
               "swaps": chosen}, open(key_path, "w"), indent=1)
    data = [{"id": x["id"], "before": x["before"], "after": x["shown"], "word": x["word"], "rep": x["rep"]}
            for x in chosen]
    Q.JPAGE.parent.mkdir(parents=True, exist_ok=True)
    Q.JPAGE.write_text(Q._html(data))
    summary = {"best_arm": BEST, "page": str(Q.JPAGE), "items": len(chosen), **window}
    json.dump(summary, open(REPORT / "joshua-page.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


def precommit(argv):
    if PRECOMMIT.exists():
        raise SystemExit(f"REFUSED: {PRECOMMIT.name} exists")
    pre21 = json.load(open(Q.PRECOMMIT))
    json.dump({"best_arm": BEST, "reason": "round 21: q50 missed its bars (round21.json success_q50)",
               "r21_precommit_sha256": _sha(Q.PRECOMMIT), "model_p50_sha256": pre21["model_p50_sha256"],
               "thresholds": pre21["thresholds"], "supplement": {"seed": SEED, "per_source": PER_SOURCE},
               "page": {"best_arm_swaps": Q.J_BEST, "arm_b_rejected_swaps": Q.J_B, "shuffle_seed": 1231},
               "source_sha256": {p: _sha(p) for p in SOURCES}, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")},
              open(PRECOMMIT, "w"), indent=1)
    print(open(PRECOMMIT).read())


def main(argv):
    cmd = {"precommit": precommit, "apply": apply, "page": page}
    if argv[:1] and argv[0] in cmd:
        return cmd[argv[0]](argv[1:])
    raise SystemExit("usage: learned-words joshua precommit|apply|page")
