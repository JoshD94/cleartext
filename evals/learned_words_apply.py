"""Apply the learned word stage to F2's dev outputs and build a blind rating page.

  learned-words apply [--max-seconds N]
      For each test-set-v3 dev passage: F2's output (with --split-filter, any
      split the split checker rejects reverted to its source sentence), then the learned word
      stage on every sentence (at most MAX_SWAPS swaps per passage, one per
      sentence). Checkpointed per passage. The locked test split is never read.
  learned-words page
      ~/Documents/cleartext-rating/rate-learned-swaps.html and key.json
      (same card and CSV design as the November 18 page; marks download as
      cleartext-marks-learned-swaps.csv)
  learned-words apply --panel onestop-dev [--targets N] [--max-seconds N]
      Round 3: the same stage on F2's output for the 181 OneStopEnglish dev
      sentences (never seen by the word stage), into apply-r3-onestop/. Every
      proposed swap records the simplicity margin (simpler.pkl, fixed in
      simpler-precommit.json); arm A picks swaps as in round 2, arm B only
      among swaps whose margin passes. The locked OneStopEnglish test split
      is never read.

The candidate ranker excludes NLI, so window NLI is computed only for each
target's top candidate here (the checker's input); featurize() is called with
NaN NLI for the rest. Swaps identical to one of Joshua's 60 marked
swaps (same sentence, word and replacement) keep their known mark and are left
off the page.
"""

import difflib
import hashlib
import html
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .learned_splits import F2_OUTPUTS, PANEL, SPLIT_STATE, sent
from .learned_words import REPORT, STATE, read_jsonl

MAX_SWAPS = 3
MAX_TARGETS = 10
APPLY = STATE / "apply"
PAGE_DIR = Path.home() / "Documents/cleartext-rating"
PAGE = PAGE_DIR / "rate-learned-swaps.html"
KEY = STATE / "rating/key.json"


SWEEP_DEV = Path.home() / ".local/state/cleartext/param-sweeps-20261116/dev/outputs"
ONESTOP_DEV = Path.home() / ".local/state/cleartext/testset-v3-20261116/panel/onestop-dev.json"
APPLY_R3 = STATE / "apply-r3-onestop"


def f2_outputs(root=F2_OUTPUTS, ids=None):
    import glob

    out = {}
    for f in sorted(glob.glob(str(root / "*/*.json"))):
        d = json.load(open(f))
        c = d["identity"]["config"]
        if c.startswith("words=off") and '"t5.min_entailment":0.5' in c and "compound_tagger" in c:
            if ids is None or d["identity"]["id"] in ids:
                out[d["identity"]["id"]] = d["output"]
    return out


def onestop_panel():
    """(panel, F2 outputs) for OneStopEnglish dev; refuses if any is missing."""
    rows = json.load(open(ONESTOP_DEV))
    panel = {r["id"]: {"original": r["original"], "domain": "onestop_news", "kind": "standard"} for r in rows}
    outputs = f2_outputs(SWEEP_DEV, set(panel))
    if set(outputs) != set(panel):
        raise SystemExit(f"REFUSED: F2 outputs for {len(set(panel) - set(outputs))} OneStopEnglish dev ids missing")
    return panel, outputs


def choose(swaps, ok=lambda s: True, key=lambda s: s["gate"] * s["check"]):
    chosen, used = [], set()
    for s in sorted(swaps, key=lambda s: -key(s)):
        if ok(s) and s["sentence"] not in used and len(chosen) < MAX_SWAPS:
            chosen.append(s)
            used.add(s["sentence"])
    return chosen


def filtered(original, output, scores, threshold):
    """F2's output with rejected splits reverted (same unit and key as the page)."""
    a, b = sent(original), sent(output)
    res, rejected = [], 0
    pid = None
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            res += b[j1:j2]
            continue
        bef = " ".join(a[i1:i2])
        key = hashlib.sha256(f"{filtered.pid}\n{bef}".encode()).hexdigest()[:16]
        if bef and b[j1:j2] and key in scores and scores[key] < threshold:
            res += a[i1:i2]
            rejected += 1
        else:
            res += b[j1:j2]
    return " ".join(res), rejected


def targets(doc, model, limit=MAX_TARGETS):
    from cleartext.generation import hyphen_fragment
    from cleartext.learned_words import eligibility as E

    toks = [t for t in doc if t.is_alpha and t.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and not t.is_stop
            and not t.ent_type_ and len(t) >= 4 and not (t.text.isupper() and len(t) > 1) and not hyphen_fragment(t)]
    if not toks:
        return []
    pc = model["complex"].predict([E.context_features(doc, t) for t in toks])
    pt = model["term"].predict([t.lemma_.lower() for t in toks])
    out = [(t, float(a), float(b)) for t, a, b in zip(toks, pc, pt)]
    elig = [x for x in out if x[1] >= model["complex"].threshold and x[2] < 0.5]
    return sorted(elig, key=lambda x: -x[1])[:limit], out


def word_swaps(text, model, pid, limit=MAX_TARGETS, score_top=None):
    """Swaps the stage accepts in one passage. Eligible words across all its
    sentences are ranked by p_complex * (1 - p_term); the top `limit` are
    featurized and decided (featurization is the costly step)."""
    from cleartext.features import nlp
    from cleartext.learned_words import features as F
    from .learned_words_study import decide

    pool, seen = [], 0
    sentences = sent(text)
    for si, s in enumerate(sentences):
        found = targets(nlp()(s), model, limit=1000)
        if found:
            elig, allc = found
            seen += len(allc)
            pool += [(pc * (1 - pt), si, s, t, pc, pt) for t, pc, pt in elig]
    swaps = []
    for _, si, s, t, pc, pt in sorted(pool, key=lambda x: -x[0])[:limit]:
        iid = f"{pid}:{si}:{t.idx}"
        inst = {"id": iid, "dataset": "apply", "group": t.lemma_.lower(), "text": s, "start": t.idx,
                "end": t.idx + len(t), "target": t.text, "gold": {}, "rated": []}
        rows = F.featurize(inst, given=False)  # NLI left NaN; filled for the top candidate below
        if not rows:
            continue
        model["extra"][iid] = {"p_complex": pc, "p_term": pt}
        for r in rows:
            r["f"].update(model["extra"][iid])
        scores = model["stage"].rank(rows)
        top = max(zip(scores, rows), key=lambda x: x[0])[1]
        a, b = F.window(s, inst["start"], inst["end"], top["word"])
        fw, bw = F.entail([(a, b), (b, a)])
        top["f"].update({"nli_fw": fw, "nli_bw": bw, "nli_min": min(fw, bw)})
        d = decide(model, rows)[iid]
        if d["word"] != top["word"]:
            continue  # decide() re-ranks with the same ranker; this guards the NLI fill
        if d["swap"]:
            after = s[:inst["start"]] + d["word"] + s[inst["end"]:]
            swaps.append({"pid": pid, "sentence": si, "before": s, "after": after, "word": t.text,
                          "rep": d["word"], "pos": d["pos"], "gate": d["gate"], "check": d["check"],
                          "p_complex": pc, "p_term": pt, "sources": top["sources"],
                          **(score_top(top) if score_top else {})})
    return swaps, seen


def apply(argv):
    import torch

    torch.set_num_threads(2)
    max_seconds = float(argv[argv.index("--max-seconds") + 1]) if "--max-seconds" in argv else 1500
    limit = int(argv[argv.index("--targets") + 1]) if "--targets" in argv else MAX_TARGETS
    if "--panel" in argv:
        name = argv[argv.index("--panel") + 1]
        if name == "dev-extra":
            return apply_r4(argv)
        if name != "onestop-dev":
            raise SystemExit("only --panel onestop-dev|dev-extra is supported")
        return apply_r3(max_seconds, limit)
    # The split checker rejects 38 of F2's 73 dev splits at its pre-registered
    # threshold, far more than the ~15% a review found broken, so by default
    # the word stage runs on F2's output unchanged (opt in with --split-filter).
    split_filter = "--split-filter" in argv
    model = pickle.load(open(STATE / "model.pkl", "rb"))
    checker = pickle.load(open(STATE / "split-checker.pkl", "rb"))
    scores = json.load(open(SPLIT_STATE / "f2-dev-scores.json"))
    panel = {r["id"]: r for r in json.load(open(PANEL))}
    outputs = f2_outputs()
    APPLY.mkdir(parents=True, exist_ok=True)
    path = APPLY / "passages.jsonl"
    done = {r["pid"] for r in read_jsonl(path)}
    deadline = time.monotonic() + max_seconds
    todo = [p for p in sorted(outputs) if p not in done]
    print(f"{len(done)} done, {len(todo)} to do", flush=True)
    with open(path, "a") as f:
        for pid in todo:
            if time.monotonic() > deadline:
                break
            filtered.pid = pid
            if split_filter:
                text, rejected = filtered(panel[pid]["original"], outputs[pid], scores, checker.threshold)
            else:
                text, rejected = outputs[pid], 0
            swaps, seen = word_swaps(text, model, pid)
            chosen = choose(swaps)
            final = sent(text)
            for s in chosen:
                final[s["sentence"]] = s["after"]
            f.write(json.dumps({"pid": pid, "domain": panel[pid]["domain"], "kind": panel[pid]["kind"],
                                "splits_rejected": rejected, "content_words": seen, "proposed": len(swaps),
                                "swaps": chosen, "output": " ".join(final)}) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {pid} swaps={len(chosen)} rejected_splits={rejected}", flush=True)
    left = len(set(outputs) - {r["pid"] for r in read_jsonl(path)})
    print("COMPLETE" if not left else f"PARTIAL {left} left", flush=True)


def apply_r3(max_seconds, limit):
    import torch

    from .learned_words_simpler import SIMPLER, active_margin

    torch.set_num_threads(2)
    margin, rule = active_margin()
    pre = {"margin": margin}
    print(f"simplicity margin {margin} (rule {rule})", flush=True)
    simpler = pickle.load(open(SIMPLER, "rb"))
    model = pickle.load(open(STATE / "model.pkl", "rb"))
    panel, outputs = onestop_panel()
    APPLY_R3.mkdir(parents=True, exist_ok=True)
    path = APPLY_R3 / "passages.jsonl"
    done = {r["pid"] for r in read_jsonl(path)}
    deadline = time.monotonic() + max_seconds
    todo = [p for p in sorted(outputs) if p not in done]
    print(f"{len(done)} done, {len(todo)} to do, targets/item {limit}, margin {pre['margin']}", flush=True)
    with open(path, "a") as f:
        for pid in todo:
            if time.monotonic() > deadline:
                break
            swaps, seen = word_swaps(outputs[pid], model, pid, limit=limit)
            if swaps:
                ms = simpler.margin([s["word"] for s in swaps], [s["rep"] for s in swaps])
                for s, m in zip(swaps, ms):
                    s["margin"] = float(m)
            a = choose(swaps)
            b = choose(swaps, lambda s: s["margin"] >= pre["margin"])
            f.write(json.dumps({"pid": pid, "domain": "onestop_news", "kind": "standard",
                                "content_words": seen, "proposed": swaps,
                                "arm_a": [_skey(s) for s in a],
                                "arm_b": [_skey(s) for s in b], "input": outputs[pid]}) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {pid} proposed={len(swaps)} a={len(a)} b={len(b)}", flush=True)
    left = len(set(outputs) - {r["pid"] for r in read_jsonl(path)})
    print("COMPLETE" if not left else f"PARTIAL {left} left", flush=True)


APPLY_R4 = STATE / "apply-r4"


def apply_r4(argv):
    """Round 4: the stage on the 84 fresh dev-document passages (source text;
    no F2 output exists for them). Every proposed swap records the round-3
    margin, the strict checker score and the target's predicted CWI share;
    arms are chosen afterwards from those fields (see page_r4)."""
    import torch

    from .learned_words_panel import PANEL_R4
    from .learned_words_simpler import SIMPLER, active_margin
    from . import learned_words_strict as ST

    torch.set_num_threads(2)
    max_seconds = float(argv[argv.index("--max-seconds") + 1]) if "--max-seconds" in argv else 1440
    margin, _ = active_margin()
    strict, floor = ST.load()
    simpler = pickle.load(open(SIMPLER, "rb"))
    model = pickle.load(open(STATE / "model.pkl", "rb"))
    panel = json.load(open(PANEL_R4))
    APPLY_R4.mkdir(parents=True, exist_ok=True)
    path = APPLY_R4 / "passages.jsonl"
    done = {r["pid"] for r in read_jsonl(path)}
    deadline = time.monotonic() + max_seconds
    todo = [p for p in panel if p["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to do; margin {margin}, strict {strict.threshold}, floor {floor}", flush=True)
    score = lambda top: {"strict": float(strict.score([top])[0])}
    with open(path, "a") as f:
        for p in todo:
            if time.monotonic() > deadline:
                break
            swaps, seen = word_swaps(p["original"], model, p["id"], score_top=score)
            if swaps:
                ms = simpler.margin([s["word"] for s in swaps], [s["rep"] for s in swaps])
                tc = simpler.raw([s["word"] for s in swaps], "cwi")
                for s, m, c in zip(swaps, ms, tc):
                    s["margin"], s["target_cwi"] = float(m), float(c)
            f.write(json.dumps({"pid": p["id"], "domain": p["domain"], "kind": p["kind"], "content_words": seen,
                                "proposed": swaps, "input": p["original"]}) + "\n")
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} {p['id']} proposed={len(swaps)}", flush=True)
    left = len({p["id"] for p in panel} - {r["pid"] for r in read_jsonl(path)})
    print("COMPLETE" if not left else f"PARTIAL {left} left", flush=True)


def r4_arms(margin, strict_t, floor):
    return {"b": lambda s: s["margin"] >= margin,
            "c": lambda s: s["margin"] >= margin and s["target_cwi"] >= floor,
            "d": lambda s: s["margin"] >= margin and s["target_cwi"] >= floor and s["strict"] >= strict_t}


def _skey(s):
    return hashlib.sha256(f"{s['pid']}\n{s['before']}\n{s['word']}\n{s['rep']}".encode()).hexdigest()[:16]


def summary():
    from .learned_words_study import wilson  # noqa: F401  (kept for report symmetry)

    rows = read_jsonl(APPLY / "passages.jsonl")
    swaps = [s for r in rows for s in r["swaps"]]
    return {"passages": len(rows), "passages_with_swaps": sum(bool(r["swaps"]) for r in rows),
            "swaps": len(swaps), "by_pos": dict(Counter(s["pos"] for s in swaps)),
            "by_domain": dict(Counter(r["domain"] for r in rows for _ in r["swaps"])),
            "simple_control_swaps": sum(len(r["swaps"]) for r in rows if r["kind"] != "standard"),
            "splits_rejected": sum(r["splits_rejected"] for r in rows),
            "content_words_seen": sum(r["content_words"] for r in rows)}


def page(argv):
    import random

    from cleartext.learned_words import data as D

    rows = read_jsonl(APPLY / "passages.jsonl")
    old = {(j["text"], j["target"].lower(), j["old_replacement"].lower()): j["mark"] for j in D.joshua60()}
    items, known = [], []
    for r in rows:
        for s in r["swaps"]:
            k = (s["before"], s["word"].lower(), s["rep"].lower())
            if k in old:
                known.append({"word": s["word"], "rep": s["rep"], "mark": old[k]})
                continue
            key = hashlib.sha256(f"{s['pid']}\n{s['before']}\n{s['word']}\n{s['rep']}".encode()).hexdigest()[:16]
            items.append(dict(s, kind="swap", key=key))
    random.Random(1118).shuffle(items)
    KEY.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"swaps": items, "known_from_60": known}, open(KEY, "w"), indent=1)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(render(items))
    agg = summary()
    agg.update({"page_items": len(items), "identical_to_marked_60": known})
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "dev-application.json").write_text(json.dumps(agg, indent=1))
    print(json.dumps(agg, indent=1))
    print(f"page {PAGE} ({len(items)} swaps), key {KEY}")


R3 = STATE / "rating-r3"
PAGE_R3 = PAGE_DIR / "rate-learned-swaps-r3.html"


def page_r3(argv):
    """Blind items (union of arms A and B, shuffled, no scores) for the
    round-3 rater, a page for Joshua, and the key that maps items to arms."""
    import random

    rows = read_jsonl(APPLY_R3 / "passages.jsonl")
    items = {}
    for r in rows:
        for s in r["proposed"]:
            k = _skey(s)
            arms = [a for a in ("a", "b") if k in r["arm_" + a]]
            if arms:
                items[k] = dict(s, key=k, arms=arms)
    items = sorted(items.values(), key=lambda x: x["key"])
    random.Random(1119).shuffle(items)
    R3.mkdir(parents=True, exist_ok=True)
    json.dump({"swaps": items}, open(R3 / "key.json", "w"), indent=1)
    blind = [{"id": f"w{n}", "before": x["before"], "after": x["after"], "word": x["word"], "rep": x["rep"]}
             for n, x in enumerate(items, 1)]
    json.dump(blind, open(R3 / "blind-items.json", "w"), indent=1)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_R3.write_text(render(items, "Rate the word swaps (round 3)", "cleartext-rate-learned-r3-20261119",
                              "cleartext-marks-learned-swaps-r3.csv"))
    print(f"{len(items)} items (arm A {sum('a' in x['arms'] for x in items)}, arm B "
          f"{sum('b' in x['arms'] for x in items)}) -> {R3 / 'blind-items.json'}, page {PAGE_R3}")


def score_r3(argv):
    """Good-rate per arm from a marks CSV (id,mark[,...]) over the blind items."""
    import csv

    path = argv[0]
    key = json.load(open(R3 / "key.json"))["swaps"]
    ids = {f"w{n}": x for n, x in enumerate(key, 1)}
    marks = {r["id"]: r["mark"] for r in csv.DictReader(open(path)) if r.get("mark")}
    from .learned_words_study import wilson

    out = {"marks_file": Path(path).name, "rated": len(marks), "items": len(ids)}
    for arm in ("a", "b"):
        ms = [marks[i] for i, x in ids.items() if arm in x["arms"] and i in marks]
        c = Counter(ms)
        out["arm_" + arm] = {"swaps": len(ms), **c, "good_rate": c["good"] / len(ms) if ms else None,
                             "good_ci95": wilson(c["good"], len(ms))}
    blocked = [marks[i] for i, x in ids.items() if x["arms"] == ["a"] and i in marks]
    out["blocked_by_margin"] = dict(Counter(blocked))
    print(json.dumps(out, indent=1))
    return out


R4 = STATE / "rating-r4"
PAGE_R4 = PAGE_DIR / "rate-learned-swaps-r4.html"


def page_r4(argv):
    """Union of the swaps each round-4 arm picks (b: round-3 stage, c: + target
    floor, d: + strict checker), shuffled and blind, plus key and page."""
    import random

    from .learned_words_simpler import active_margin
    from . import learned_words_strict as ST

    margin, _ = active_margin()
    strict, floor = ST.load()
    arms = r4_arms(margin, strict.threshold, floor)
    items = {}
    for r in read_jsonl(APPLY_R4 / "passages.jsonl"):
        for arm, ok in arms.items():
            for s in choose(r["proposed"], ok):
                k = _skey(s)
                items.setdefault(k, dict(s, key=k, arms=[], domain=r["domain"], kind=r["kind"]))["arms"].append(arm)
    items = sorted(items.values(), key=lambda x: x["key"])
    random.Random(1120).shuffle(items)
    R4.mkdir(parents=True, exist_ok=True)
    json.dump({"swaps": items}, open(R4 / "key.json", "w"), indent=1)
    blind = [{"id": f"w{n}", "before": x["before"], "after": x["after"], "word": x["word"], "rep": x["rep"]}
             for n, x in enumerate(items, 1)]
    json.dump(blind, open(R4 / "blind-items.json", "w"), indent=1)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_R4.write_text(render(items, "Rate the word swaps (round 4)", "cleartext-rate-learned-r4-20261119",
                              "cleartext-marks-learned-swaps-r4.csv"))
    print(f"{len(items)} items " + ", ".join(f"arm {a} {sum(a in x['arms'] for x in items)}" for a in arms)
          + f" -> {R4 / 'blind-items.json'}, page {PAGE_R4}")


def score_arms(key_path, marks_path):
    """Good rate per arm from a marks CSV (id,mark,...) over blind items w1..wN."""
    import csv

    from .learned_words_study import wilson

    key = json.load(open(key_path))["swaps"]
    ids = {f"w{n}": x for n, x in enumerate(key, 1)}
    marks = {r["id"]: r["mark"] for r in csv.DictReader(open(marks_path)) if r.get("mark")}
    arms = sorted({a for x in key for a in x["arms"]})
    out = {"rated": len(marks), "items": len(ids)}
    for arm in arms:
        ms = [marks[i] for i, x in ids.items() if arm in x["arms"] and i in marks]
        c = Counter(ms)
        out["arm_" + arm] = {"swaps": len(ms), **c, "good_rate": c["good"] / len(ms) if ms else None,
                             "good_ci95": wilson(c["good"], len(ms))}
    return out


def render(items, title="Rate the word swaps (round 2)", store="cleartext-rate-learned-20261118",
           download="cleartext-marks-learned-swaps.csv"):
    import re

    def mark(s, w):
        return re.sub(r"\b(%s)\b" % re.escape(html.escape(w)), r"<mark>\1</mark>", html.escape(s), count=1)

    opts = [("good", "Good"), ("meaning", "Changes meaning"), ("notsimpler", "Not simpler")]
    cards = []
    for n, r in enumerate(items, 1):
        btns = "".join(f'<button data-v="{v}">{t}</button>' for v, t in opts)
        cards.append(f'<div class="card" data-id="w{n}" data-key="{r["key"]}"><div class="n">W{n} '
                     f'<b>{html.escape(r["word"])} → {html.escape(r["rep"])}</b></div>'
                     f'<div class="lbl">Before</div><p>{mark(r["before"], r["word"])}</p>'
                     f'<div class="lbl">After</div><p>{mark(r["after"], r["rep"])}</p><div class="btns">{btns}</div></div>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rate Word Swaps</title><style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--acc:#2563eb;--good:#16a34a;--bad:#dc2626;--meh:#d97706}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mut:#999;--line:#2a2a2a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 -apple-system,system-ui,sans-serif}}
main{{max-width:720px;margin:0 auto;padding:16px}}
h1{{font-size:22px;margin:8px 0}} .intro{{color:var(--mut)}}
.card{{border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:12px 0}}
.n{{font-size:13px;color:var(--mut);margin-bottom:6px}} .n b{{color:var(--fg)}}
.lbl{{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}} p{{margin:2px 0 10px}}
mark{{background:#fde68a;color:#111;border-radius:3px;padding:0 2px}}
.btns{{display:flex;gap:8px;flex-wrap:wrap}} button{{font:inherit;font-size:14px;padding:6px 12px;border-radius:8px;border:1px solid var(--line);background:none;color:var(--fg);cursor:pointer}}
.card[data-mark] {{opacity:.55}} .card[data-mark]:hover{{opacity:1}}
.card[data-mark=good] button[data-v=good]{{background:var(--good);color:#fff;border-color:var(--good)}}
.card[data-mark=meaning] button[data-v=meaning]{{background:var(--bad);color:#fff;border-color:var(--bad)}}
.card[data-mark=notsimpler] button[data-v=notsimpler]{{background:var(--meh);color:#fff;border-color:var(--meh)}}
.bar{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 0;display:flex;justify-content:space-between;align-items:center;gap:8px}}
.bar button{{background:var(--acc);color:#fff;border-color:var(--acc)}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="intro">These are new swaps from a retrained word stage. Each card shows a sentence before and after one word was replaced. Tap the one answer that fits: is the highlighted replacement correct here, and simpler? Your marks save in this browser as you go. When you're done, tap <b>Download marks</b> at the top (it saves <code>{download}</code>, separate from earlier pages' files). Scores are hidden.</p>
<div class="bar"><span id="count"></span><button id="dl">Download marks</button></div>
{chr(10).join(cards)}
</main><script>
const KEY='{store}';const cs=[...document.querySelectorAll('.card')];
let m={{}};try{{m=JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(e){{}}
function paint(){{cs.forEach(c=>{{const v=m[c.dataset.id];if(v)c.dataset.mark=v;else delete c.dataset.mark}});
document.getElementById('count').textContent=Object.keys(m).length+' of '+cs.length+' rated';
try{{localStorage.setItem(KEY,JSON.stringify(m))}}catch(e){{}}}}
cs.forEach(c=>c.querySelectorAll('button').forEach(b=>b.onclick=()=>{{const id=c.dataset.id;
if(m[id]===b.dataset.v)delete m[id];else m[id]=b.dataset.v;paint()}}));
document.getElementById('dl').onclick=()=>{{const rows=['id,key,mark',...cs.map(c=>c.dataset.id+','+c.dataset.key+','+(m[c.dataset.id]||''))].join('\\n');
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows],{{type:'text/csv'}}));a.download='{download}';a.click()}};
paint();
</script></body></html>"""
