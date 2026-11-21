"""Round 5: the fine-tuned judge on fresh dev-document passages.

  learned-words r5 apply [--max-seconds N]   # checkpointed; rerun until COMPLETE
  learned-words r5 page                      # blind items + key (+ controls)
  learned-words r5 score                     # Good rate per arm from blind marks

For every passage of panel-r5 (built by `panel build-r5`) and its top
MAX_TARGETS eligible targets, one run records the stage's own decision (as in
round 4) and, for every generated candidate, the ranker score, the judge's
probability and the simplicity margin. Arms are post-hoc filters fixed in
judge-precommit.json before any round-5 swap exists (m = margin, judge_t and
strict_t from `judge eval`):

  b  round-3 stage: stage swap and margin >= m            (baseline)
  e  b and judge(stage's word) >= judge_t
  f  judge picks: eligible target, the candidate with the highest judge
     score among those with margin >= m; swap if judge >= judge_t
  g  f with strict_t (only if strict_t exists)

At most one swap per sentence and MAX_SWAPS per passage, highest score first
(b, e: gate x check; f, g: judge).

The blind page also carries 20 controls (TSAR-2022 test sentences with their
most frequent human substitute, a ceiling estimate) and 20 round-4 items
(rater consistency). Neither is ever used to fit or choose anything.
"""

import json
import pickle
import random
import time

from .learned_words import STATE, read_jsonl

APPLY_R5 = STATE / "apply-r5"
R5 = STATE / "rating-r5"


def arms(pre):
    m, jt, st = pre["margin"], pre["judge_t"], pre["strict_t"]
    out = {"b": ("stage", lambda c: c["stage_swap"] and c["margin"] >= m),
           "e": ("stage", lambda c: c["stage_swap"] and c["margin"] >= m and c["judge"] >= jt),
           "f": ("judge", lambda c: c["judge"] >= jt)}
    if st is not None:
        out["g"] = ("judge", lambda c: c["judge"] >= st)
    return out


def target_records(text, model, pid, weights, simpler, limit):
    from cleartext.features import nlp
    from cleartext.learned_words import features as F
    from cleartext.learned_words.judge import pair, score

    from .learned_words_apply import sent, targets
    from .learned_words_study import decide

    pool, seen = [], 0
    for si, s in enumerate(sent(text)):
        found = targets(nlp()(s), model, limit=1000)
        if found:
            elig, allc = found
            seen += len(allc)
            pool += [(pc * (1 - pt), si, s, t, pc, pt) for t, pc, pt in elig]
    out = []
    for _, si, s, t, pc, pt in sorted(pool, key=lambda x: -x[0])[:limit]:
        iid = f"{pid}:{si}:{t.idx}"
        start, end = t.idx, t.idx + len(t)
        inst = {"id": iid, "dataset": "apply", "group": t.lemma_.lower(), "text": s, "start": start,
                "end": end, "target": t.text, "gold": {}, "rated": []}
        rows = F.featurize(inst, given=False)
        if not rows:
            continue
        model["extra"][iid] = {"p_complex": pc, "p_term": pt}
        for r in rows:
            r["f"].update(model["extra"][iid])
        rank = model["stage"].rank(rows)
        top = max(zip(rank, rows), key=lambda x: x[0])[1]
        a, b = F.window(s, start, end, top["word"])
        fw, bw = F.entail([(a, b), (b, a)])
        top["f"].update({"nli_fw": fw, "nli_bw": bw, "nli_min": min(fw, bw)})
        d = decide(model, rows)[iid]
        if d["word"] != top["word"]:
            continue
        js = score(weights, [pair(s, start, end, r["word"]) for r in rows])
        ms = simpler.margin([t.text] * len(rows), [r["word"] for r in rows])
        cands = [{"rep": r["word"], "rank": float(k), "judge": float(j), "margin": float(m),
                  "sources": r["sources"]} for r, k, j, m in zip(rows, rank, js, ms)]
        out.append({"pid": pid, "sentence": si, "before": s, "word": t.text, "start": start, "end": end,
                    "pos": d["pos"], "p_complex": pc, "p_term": pt, "eligible": d["eligible"],
                    "stage": {"rep": d["word"], "swap": d["swap"], "gate": d["gate"], "check": d["check"]},
                    "cands": cands})
    return out, seen


def apply(argv):
    import torch

    from .learned_words_judge import load
    from .learned_words_panel import PANEL_R5
    from .learned_words_apply import MAX_TARGETS
    from .learned_words_simpler import SIMPLER

    torch.set_num_threads(int(argv[argv.index("--threads") + 1]) if "--threads" in argv else 3)
    max_seconds = float(argv[argv.index("--max-seconds") + 1]) if "--max-seconds" in argv else 1440
    weights, jt, st = load()
    simpler = pickle.load(open(SIMPLER, "rb"))
    model = pickle.load(open(STATE / "model.pkl", "rb"))
    panel = json.load(open(PANEL_R5))
    APPLY_R5.mkdir(parents=True, exist_ok=True)
    path = APPLY_R5 / "passages.jsonl"
    done = {r["pid"] for r in read_jsonl(path)}
    deadline = time.monotonic() + max_seconds
    todo = [p for p in panel if p["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to do; judge_t {jt}, strict_t {st}", flush=True)
    with open(path, "a") as f:
        for p in todo:
            if time.monotonic() > deadline:
                break
            recs, seen = target_records(p["original"], model, p["id"], weights, simpler, MAX_TARGETS)
            f.write(json.dumps({"pid": p["id"], "domain": p["domain"], "kind": p["kind"], "content_words": seen,
                                "targets": recs, "input": p["original"]}) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {p['id']} targets={len(recs)}", flush=True)
    left = len({p["id"] for p in panel} - {r["pid"] for r in read_jsonl(path)})
    print("COMPLETE" if not left else f"PARTIAL {left} left", flush=True)


def swaps_for(rec, arm_kind, ok, m):
    """Candidate swaps of one passage under one arm (before choose())."""
    out = []
    for t in rec["targets"]:
        by = {c["rep"]: c for c in t["cands"]}
        if arm_kind == "stage":
            c = by.get(t["stage"]["rep"])
            if c is None:
                continue
            c = dict(c, stage_swap=t["stage"]["swap"])
        else:
            ok_m = [c for c in t["cands"] if c["margin"] >= m]
            if not t["eligible"] or not ok_m:
                continue
            c = dict(max(ok_m, key=lambda c: c["judge"]), stage_swap=t["stage"]["swap"])
        if not ok(c):
            continue
        s = t["before"]
        out.append({"pid": t["pid"], "sentence": t["sentence"], "before": s,
                    "after": s[:t["start"]] + c["rep"] + s[t["end"]:], "word": t["word"], "rep": c["rep"],
                    "pos": t["pos"], "gate": t["stage"]["gate"], "check": t["stage"]["check"],
                    "judge": c["judge"], "margin": c["margin"], "stage_rep": t["stage"]["rep"]})
    return out


def picks():
    from .learned_words_apply import choose
    from .learned_words_judge import PRECOMMIT

    pre = json.load(open(PRECOMMIT))
    items = {}
    from .learned_words_apply import _skey

    for r in read_jsonl(APPLY_R5 / "passages.jsonl"):
        for arm, (kind, ok) in arms(pre).items():
            key = (lambda s: s["gate"] * s["check"]) if kind == "stage" else (lambda s: s["judge"])
            for s in choose(swaps_for(r, kind, ok, pre["margin"]), key=key):
                k = _skey(s)
                items.setdefault(k, dict(s, key=k, arms=[], domain=r["domain"], kind=r["kind"]))["arms"].append(arm)
    return sorted(items.values(), key=lambda x: x["key"])


def controls(n=20, seed=1121):
    """TSAR-2022 test sentences with their most frequent single-word human substitute."""
    from .learned_words import all_instances

    rng = random.Random(seed)
    inst = [i for i in all_instances() if i["dataset"] == "tsar_test" and i["gold"]]
    rng.shuffle(inst)
    out = []
    for i in inst:
        g = max(i["gold"], key=lambda w: (i["gold"][w], w))
        if " " in g or g.lower() == i["target"].lower():
            continue
        out.append({"before": i["text"], "after": i["text"][:i["start"]] + g + i["text"][i["end"]:],
                    "word": i["target"], "rep": g, "source": "control", "ref": i["id"]})
        if len(out) == n:
            break
    return out


def page(argv):
    items = picks()
    rng = random.Random(1121)
    old = json.load(open(STATE / "rating-r4/key.json"))["swaps"]
    old_marks = {}
    import csv

    mp = {m["id"]: m for m in json.load(open(STATE / "rating-r4/blind-mixed-map.json"))}
    for r in csv.DictReader(open(STATE / "rating-r4/blind-mixed-marks.csv")):
        m = mp[r["id"]]
        if m["round"] == "r4":
            old_marks[m["src"]] = r["mark"]
    olds = [dict(x, source="r4", ref=f"w{n}", old_mark=old_marks.get(f"w{n}"))
            for n, x in enumerate(old, 1) if f"w{n}" in old_marks]
    rng.shuffle(olds)
    mixed = [dict(x, source="r5") for x in items] + controls() + olds[:20]
    rng.shuffle(mixed)
    R5.mkdir(parents=True, exist_ok=True)
    json.dump({"swaps": items}, open(R5 / "key.json", "w"), indent=1)
    json.dump([{"id": f"m{n}", "source": x["source"], "key": x.get("key"), "ref": x.get("ref"),
                "old_mark": x.get("old_mark"), "arms": x.get("arms")} for n, x in enumerate(mixed, 1)],
              open(R5 / "blind-mixed-map.json", "w"), indent=1)
    json.dump([{"id": f"m{n}", "before": x["before"], "after": x["after"], "word": x["word"], "rep": x["rep"]}
               for n, x in enumerate(mixed, 1)], open(R5 / "blind-mixed.json", "w"), indent=1)
    from collections import Counter

    print(json.dumps({"r5_items": len(items), "mixed": len(mixed), "by_source": Counter(x["source"] for x in mixed),
                      "per_arm": Counter(a for x in items for a in x["arms"])}, indent=1))


def score(argv):
    import csv
    from collections import Counter

    from .learned_words import REPORT
    from .learned_words_study import wilson

    mp = json.load(open(R5 / "blind-mixed-map.json"))
    marks = {r["id"]: r for r in csv.DictReader(open(R5 / "blind-mixed-marks.csv"))}
    out = {"rated": len(marks)}
    r5 = [(m, marks[m["id"]]["mark"]) for m in mp if m["source"] == "r5" and m["id"] in marks]
    for arm in sorted({a for m, _ in r5 for a in m["arms"]}):
        ms = [k for m, k in r5 if arm in m["arms"]]
        c = Counter(ms)
        out["arm_" + arm] = {"swaps": len(ms), **c, "good_rate": c["good"] / len(ms), "good_ci95": wilson(c["good"], len(ms))}
    ctl = Counter(marks[m["id"]]["mark"] for m in mp if m["source"] == "control" and m["id"] in marks)
    out["controls_tsar_gold"] = {**ctl, "good_rate": ctl["good"] / max(1, sum(ctl.values()))}
    agree = [(m["old_mark"], marks[m["id"]]["mark"]) for m in mp if m["source"] == "r4" and m["id"] in marks]
    out["r4_agreement"] = {"items": len(agree), "agree": sum(a == b for a, b in agree)}
    (REPORT / "round5.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv):
    cmd = {"apply": apply, "page": page, "score": score}.get(argv[0] if argv else None)
    if cmd is None:
        raise SystemExit("usage: learned-words r5 apply|page|score")
    return cmd(argv[1:])
