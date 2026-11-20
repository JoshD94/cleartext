"""Fresh passages for later word-stage rounds: unused windows of test-set v3 DEV documents.

  learned-words panel build-r4
  learned-words panel build-r5


Re-runs `evals/build_testset_v3.select` offline (download cache only), so it
opens exactly the documents of the v3 panel, and keeps every candidate window
of a DEV document that overlaps no panel passage. Test documents are never
read beyond what `select` itself opens, and none of their text is kept.
From those windows: per standard domain 14 passages, per simple-control
domain 6, picked round-robin over documents from a seeded shuffle (seed
20261119), pairwise non-overlapping. Text stays in the state directory.

build-r5 (round 5) does the same with seed 20261120 and also skips every
window that overlaps a round-4 passage.
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .learned_words import STATE

V3_STATE = Path.home() / ".local/state/cleartext/testset-v3-20261116"
PANEL_R4 = STATE / "panel-r4/passages.json"
PANEL_R5 = STATE / "panel-r5/passages.json"
SEED = 20261119
SEED_R5 = 20261120
PER_DOMAIN = {"standard": 14, "simple_control": 6}


def _overlap(a, b):
    return a[0] == b[0] and not (a[2] < b[1] or a[1] > b[2])


def build_r4(argv):
    return build(PANEL_R4, SEED, "r4", [])


def build_r5(argv):
    return build(PANEL_R5, SEED_R5, "r5", json.load(open(PANEL_R4)))


def build(path, seed, prefix, previous):
    from . import build_testset_v3 as B

    if path.exists():
        raise SystemExit(f"REFUSED: {path} exists")
    panel = json.load(open(V3_STATE / "panel/all.json"))
    dev_docs = {p["source_doc"] for p in panel if p["split"] == "dev"}
    used = defaultdict(list)
    for p in panel + previous:
        used[p["source_doc"]].append(tuple(p["sentence_span"]))
    texts = [p["original"] for p in panel + previous]
    info = {p["source_doc"]: (p["domain"], p["kind"]) for p in panel}

    seen, metas = [], []
    pick = B.pick_windows

    def record(found, k, rng):
        seen.append(found)
        return pick(found, k, rng)

    B.pick_windows = record
    for src, (pool, load) in list(B.SOURCES.items()):
        def wrapped(fetcher, key, load=load):
            r = load(fetcher, key)
            if r is not None:
                metas.append(r[0])
            return r
        B.SOURCES[src] = (pool, wrapped)
    B.select(B.Fetcher(V3_STATE / "raw/http", offline=True), log=lambda *a: None)
    if len(seen) != len(metas):
        raise SystemExit("REFUSED: could not align windows with documents")

    by_domain = defaultdict(lambda: defaultdict(list))
    for meta, found in zip(metas, seen):
        doc = meta["source_doc"]
        if doc not in dev_docs:
            continue
        for block, first, last, text in found:
            span = (block, first, last)
            if any(_overlap(span, u) for u in used[doc]) or any(text in t or t in text for t in texts):
                continue
            by_domain[info[doc]][doc].append((span, text, meta))
    out = []
    for (domain, kind), docs in sorted(by_domain.items()):
        rng = random.Random(f"{seed}:{domain}")
        order = sorted(docs)
        rng.shuffle(order)
        for d in order:
            rng.shuffle(docs[d])
        chosen, want = defaultdict(list), PER_DOMAIN[kind]
        n = 0
        while n < want and any(docs[d] for d in order):
            for d in order:
                while docs[d] and n < want:
                    span, text, meta = docs[d].pop()
                    if all(not _overlap(span, s) for s, _ in chosen[d]):
                        chosen[d].append((span, text))
                        n += 1
                        out.append({"id": f"{prefix}-{domain[:4]}-{n:02d}", "domain": domain, "kind": kind,
                                    "source_doc": d, "sentence_span": list(span), "original": text,
                                    "sha256": hashlib.sha256(text.encode()).hexdigest()})
                        break
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    from collections import Counter

    print(json.dumps({"passages": len(out), "by_domain": Counter(p["domain"] for p in out),
                      "documents": len({p["source_doc"] for p in out})}, indent=1))
    print("COMPLETE")


def main(argv):
    if argv[:1] == ["build-r4"]:
        return build_r4(argv[1:])
    if argv[:1] == ["build-r5"]:
        return build_r5(argv[1:])
    raise SystemExit("usage: learned-words panel build-r4|build-r5")
