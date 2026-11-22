"""Structure v2 study (November 22, 2026): sentence operations beyond
splitting, measured one by one on fresh public text with blind ratings and on
the test set v3 dev split on top of F2. No LLM is used in any operation.

Subcommands (raw text stays in --state, outside the checkout):
  pool     fresh/build sentence pools from the public corpora, excluding every
           document used by earlier panels (word-stage rounds, test set v3) and
           any sentence sharing an 8-gram with them
  apply    run operations over a pool: every firing, accepted or rejected
  panel    a blind rating panel: up to N accepted items per operation, round
           robin over sources, each rater's own shuffle (texts only)
  tally    majority marks, Wilson intervals, per-operation/per-rule results
  dev      apply operations to F2's cached dev outputs (100 passages)
  score    evaluation metrics for dev outputs (queue job: DeBERTa NLI)
  judge    dev gates and paired deltas against F2

The locked test split is never read: dev passages come from panel/dev.json and
`dev` refuses any ID listed in test_ids.txt.
"""

import argparse
import gzip
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_ROOT = Path.home() / ".local/state/cleartext"
WORDS = STATE_ROOT / "learned-words-20261118"
V3 = STATE_ROOT / "testset-v3-20261116"
SWEEP_DEV = STATE_ROOT / "param-sweeps-20261116/dev"
SOURCES = {"news": WORDS / "genre/news.jsonl.gz", "wiki": WORDS / "genre/wiki.jsonl.gz",
           "rfc": WORDS / "corpus/rfc.jsonl.gz", "fedreg": WORDS / "corpus/fedreg.jsonl.gz",
           "openfda": WORDS / "corpus/openfda.jsonl.gz"}
# Downloaded for this study (structure_v2_fetch.py): sources no earlier panel drew from.
DOWNLOAD = STATE_ROOT / "structure-20261122/download"
FRESH_SOURCES = {"news2": DOWNLOAD / "news.jsonl.gz", "goodwiki": DOWNLOAD / "goodwiki.jsonl.gz",
                 "fedreg2": DOWNLOAD / "fedreg.jsonl.gz"}
F2 = ('words=off|phrases=authored|structure=t5+compound_tagger|negation=off|grammar=off|terms=off#'
      '{"t5.min_entailment":0.5,"t5.passage":"passage","t5.waive":["argument_roles","complement_scope",'
      '"complete_clauses","content_retained","discourse_markers","finite_predicates","generation_terminated",'
      '"logical_connectives","modality","no_new_content","operator_scope","predicate_tense",'
      '"relative_attachments","sentence_count","subject_groups","uncertainty_scope"]}')
MARKS = ("good", "notsimpler", "meaning", "ungrammatical")
SEVERITY = {"good": 0, "notsimpler": 1, "ungrammatical": 2, "meaning": 3}
SEED = 20261122


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False))
    tmp.replace(path)


def outside_checkout(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise SystemExit("State must be outside the checkout; raw text stays local.")
    return path


def bucket(doc):
    """0-9 from the document name: 0-2 build, 3-6 round A, 7-9 round B."""
    return int(hashlib.sha256(doc.encode()).hexdigest(), 16) % 10


def grams(text, n=8):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


# --- pools ------------------------------------------------------------------------

def used_texts():
    """(documents, texts) used by earlier panels: every passages/pool file of
    the word-stage rounds and every test set v3 passage (all splits)."""
    docs, texts = set(), []
    files = sorted(WORDS.glob("*/passages.json")) + sorted(WORDS.glob("*/pool.json"))
    files += [V3 / "panel/all.json"]
    for f in files:
        rows = json.loads(f.read_text())
        rows = rows if isinstance(rows, list) else rows.get("passages", rows.get("rows", []))
        for r in rows:
            if not isinstance(r, dict):
                continue
            if r.get("source_doc"):
                docs.add(str(r["source_doc"]))
            for k in ("original", "text", "s", "sentence"):
                if isinstance(r.get(k), str):
                    texts.append(r[k])
    return docs, texts, [str(f) for f in files]


def clean(s):
    words = s.split()
    return (8 <= len(words) <= 45 and s[:1].isupper() and s.rstrip()[-1:] in ".!?"
            and "http" not in s and "www." not in s and sum(c.isalpha() for c in s) / max(1, len(s)) > 0.7
            and not re.search(r"\s{2,}|_{2,}|\.{3,}|[|<>{}]", s)
            and not re.match(r"(?:[A-Z]{2,}\s+(?:&\s+)?){2,}", s))  # an all-caps heading run into the sentence


def run_pool(args):
    state = outside_checkout(args.state)
    docs_used, texts_used, files = used_texts()
    used = set()
    for t in texts_used:
        used |= grams(t)
    rng = random.Random(SEED)
    out, manifest = {"build": [], "A": [], "B": []}, {"excluded_files": files, "sources": {}}
    sources = {**SOURCES, **{k: v for k, v in FRESH_SOURCES.items() if v.exists()}}
    for name, path in sources.items():
        rows = [json.loads(line) for line in gzip.open(path, "rt")]
        by_doc = defaultdict(list)
        for k, r in enumerate(rows):
            by_doc[r["doc"]].append(r["s"])
        stats = Counter()
        for split, buckets in (("build", range(0, 3)), ("A", range(3, 7)), ("B", range(7, 10))):
            docs = sorted(d for d in by_doc if bucket(d) in buckets)
            rng.shuffle(docs)
            kept, wide = [], []
            for d in docs:
                if d in docs_used:
                    stats["docs_used_elsewhere"] += 1
                    continue
                sents = by_doc[d]
                idx = [i for i, s in enumerate(sents) if clean(s)]
                rng.shuffle(idx)
                full = len(kept) < args.per_source
                for rank, i in enumerate(idx):
                    if grams(sents[i]) & used:
                        stats["sentences_8gram_overlap"] += 1
                        continue
                    row = {"id": f"{name}:{hashlib.sha256((d + str(i)).encode()).hexdigest()[:12]}",
                           "source": name, "doc": d, "index": i, "sentence": sents[i],
                           "context": sents[max(0, i - 3): i]}
                    # The standard pool: at most per_doc sentences per document
                    # until per_source; "wide" keeps every sentence of the
                    # documents the standard pool drew from (rare operations).
                    if full and rank < args.per_doc and len(kept) < args.per_source:
                        kept.append(row)
                    if full:
                        wide.append(row)
                if len(kept) >= args.per_source:
                    break
            out[split] += kept
            out.setdefault(split + "wide", []).extend(wide)
            stats[f"{split}_sentences"] = len(kept)
            stats[f"{split}wide_sentences"] = len(wide)
        manifest["sources"][name] = {"sha256": sha(path), **stats}
        print(name, dict(stats), flush=True)
    for split, rows in out.items():
        write_json(state / f"pool-{split}.json", rows)
        manifest[f"pool_{split}_sha256"] = sha(state / f"pool-{split}.json")
    write_json(state / "pool-manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "excluded_files"}, indent=1))


# --- applying operations -----------------------------------------------------------

def make_stage(op, entailment=None, min_entailment=None, gate=None):
    from cleartext.structure_v2 import StructureV2
    from cleartext.structure_v2.stage import build
    operator = build(op)
    if gate is not None:
        operator.gate = gate
    return StructureV2(ops=(op,), operators=[operator], entailment=entailment, min_entailment=min_entailment)


class SmallNLI:
    """The F2 split floor's scorer (Chhoeng's cross-encoder/nli-deberta-v3-small
    min of both directions), memoized in a JSONL file."""

    def __init__(self, memo):
        self.memo_path, self.memo, self.model = Path(memo), {}, None
        if self.memo_path.exists():
            for line in self.memo_path.read_text().splitlines():
                r = json.loads(line)
                self.memo[(r["a"], r["b"])] = r["score"]

    def __call__(self, a, b):
        if (a, b) not in self.memo:
            if self.model is None:
                from cleartext.teammate_stages import EntailmentVeto
                hub = Path.home() / ".cache/huggingface/hub"
                path = hub / "models--cross-encoder--nli-deberta-v3-small/snapshots/fa2804872c3b4bd748f38c0185cc85775361e735"
                self.model = EntailmentVeto(model=path)
            score = float(self.model.score(a, b))
            self.memo[(a, b)] = score
            with self.memo_path.open("a") as f:
                f.write(json.dumps({"a": a, "b": b, "score": score}) + "\n")
        return self.memo[(a, b)]


def chhoeng_agentive(placeholder=None):
    """Chhoeng's converter, unchanged, as a comparator arm: placeholder=None
    (passives with a by-phrase only) or "auto" (his default: "someone" /
    "people" for agentless passives)."""
    from cleartext.teammate_stages import PassiveStage
    name = "chhoeng_agentive" if placeholder is None else "chhoeng_auto"

    class Arm:
        names = (name,)

        def __init__(self):
            self.stage = PassiveStage(placeholder=placeholder)

        def rewrite_sentence(self, sentence, context=()):
            out = self.stage.converter.transform(sentence)
            if out == sentence:
                return sentence, [], []
            return out, [{"op": name, "rule": placeholder or "agentive", "before": sentence, "after": out,
                          "features": {}}], []
    return Arm()


def kea_negation():
    """Kea-Roy's double-negation rule (rewrites.negation_rewrite), unchanged,
    for coverage."""
    from cleartext.rewrites import negation_rewrite

    class Arm:
        def rewrite_sentence(self, sentence, context=()):
            out, changes = negation_rewrite(sentence)
            if out == sentence:
                return sentence, [], []
            return out, [{"op": "kea_negation", "rule": (changes[0].get("rule") if changes else "negation"),
                          "before": sentence, "after": out, "features": {}}], []
    return Arm()


def run_apply(args):
    from cleartext.structure_v2.common import parse
    state = outside_checkout(args.state)
    pool = json.loads(Path(args.pool).read_text())
    if args.limit:
        pool = pool[: args.limit]
    nli = SmallNLI(state / "small-nli.jsonl") if args.min_entailment is not None else None
    gate = None
    if args.gate:
        from cleartext.structure_v2.gate import load_gate
        gate = load_gate(args.gate)
    out_path = state / args.out
    done = set()
    if out_path.exists():
        done = {(r["id"], r["arm"]) for r in map(json.loads, out_path.read_text().splitlines())}
    started = time.monotonic()
    stages = {}
    with out_path.open("a") as f:
        for arm in args.ops:
            if arm == "kea_negation":
                stages[arm] = kea_negation()
            elif arm in ("chhoeng_agentive", "chhoeng_auto"):
                stages[arm] = chhoeng_agentive(None if arm == "chhoeng_agentive" else "auto")
            else:
                stages[arm] = make_stage(arm, nli, args.min_entailment,
                                         gate if arm.startswith("passive") else None)
        for k, row in enumerate(pool):
            if time.monotonic() - started > args.max_seconds:
                print("PARTIAL", k, flush=True)
                return
            context = [parse(c) for c in row["context"]]
            for arm, stage in stages.items():
                if (row["id"], arm) in done:
                    continue
                new, accepted, rejected = stage.rewrite_sentence(row["sentence"], context)
                rec = {"id": row["id"], "arm": arm, "source": row["source"], "changed": new != row["sentence"],
                       "after": new, "accepted": accepted,
                       "rejected": [{k2: v for k2, v in r.items() if k2 != "features"} for r in rejected]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 500 == 0:
                f.flush()
                print(f"{k}/{len(pool)} {time.monotonic() - started:.0f}s", flush=True)
    print("COMPLETE", flush=True)


def load_apply(path):
    recs = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        recs[r["arm"]][r["id"]] = r
    return recs


# --- rating panel ------------------------------------------------------------------

def run_panel(args):
    state = outside_checkout(args.state)
    pool = {r["id"]: r for r in json.loads(Path(args.pool).read_text())}
    recs = load_apply(state / args.applied)
    rng = random.Random(args.seed)
    items, seen = [], set()
    per_arm = {}
    for arm in args.ops:
        changed = [r for r in recs[arm].values() if r["changed"]]
        by_source = defaultdict(list)
        for r in changed:
            by_source[r["source"]].append(r)
        for v in by_source.values():
            rng.shuffle(v)
        order = sorted(by_source)
        rng.shuffle(order)
        picked = []
        cap = dict(x.split("=") for x in args.cap).get(arm) if args.cap else None
        cap = int(cap) if cap else args.per_op
        while len(picked) < cap and any(by_source.values()):
            for s in order:
                if by_source[s] and len(picked) < cap:
                    picked.append(by_source[s].pop())
        per_arm[arm] = {"changed": len(changed), "rated": len(picked),
                        "sentences": len(recs[arm]),
                        "by_source": dict(Counter(r["source"] for r in changed))}
        for r in picked:
            p = pool[r["id"]]
            key = (p["sentence"], r["after"])
            items.append({"arm": arm, "id": r["id"], "source": r["source"], "context": p["context"],
                          "before": p["sentence"], "after": r["after"], "duplicate": key in seen,
                          "rules": [e["rule"] for e in r["accepted"]]})
            seen.add(key)
    # One card per distinct (before, after) pair; arms producing the same pair share its marks.
    cards, index = [], {}
    for it in items:
        key = (it["before"], it["after"])
        if key not in index:
            # Hashed IDs: no arm order leaks through the card name.
            index[key] = "c" + hashlib.sha256(f"{args.seed}|{key[0]}|{key[1]}".encode()).hexdigest()[:8]
            cards.append({"card": index[key], "context": it["context"], "before": it["before"],
                          "after": it["after"]})
        it["card"] = index[key]
    key_rows = [{k: v for k, v in it.items() if k not in ("context",)} for it in items]
    write_json(state / f"{args.name}-key.json", {"items": key_rows, "per_arm": per_arm})
    # Each rater gets its own directory holding only its cards (no key, no arms).
    for rater in range(1, args.raters + 1):
        order = list(cards)
        random.Random(args.seed * 31 + rater).shuffle(order)
        rows = [{"n": k + 1, **c} for k, c in enumerate(order)]
        write_json(state / args.name / f"rater{rater}" / "cards.json", rows)
    print(json.dumps(per_arm, indent=1))
    print(f"{len(cards)} cards for {len(items)} items")


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return [max(0.0, c - h), min(1.0, c + h)]


def majority(marks):
    """The majority of three marks; when all differ, the most severe."""
    c = Counter(marks)
    top, n = c.most_common(1)[0]
    if n >= 2:
        return top
    return max(marks, key=lambda m: SEVERITY[m])


def read_marks(path):
    rows = {}
    for line in Path(path).read_text().splitlines()[1:]:
        if not line.strip():
            continue
        card, mark = [x.strip() for x in line.split(",")[:2]]
        if mark not in MARKS:
            raise ValueError(f"{path}: unknown mark {mark!r} for {card}")
        rows[card] = mark
    return rows


def tally(items, marks, arms, rated_sentences):
    out = {}
    for arm in arms:
        sel = [it for it in items if it["arm"] == arm]
        votes = [majority([m[it["card"]] for m in marks]) for it in sel]
        n = len(votes)
        c = Counter(votes)
        per_rater = [Counter(m[it["card"]] for it in sel) for m in marks]
        by_rule = defaultdict(list)
        for it, v in zip(sel, votes):
            by_rule["+".join(sorted(set(it["rules"]))) or "-"].append(v)
        by_source = defaultdict(list)
        for it, v in zip(sel, votes):
            by_source[it["source"]].append(v)
        info = rated_sentences.get(arm, {})
        out[arm] = {
            "n": n, "good": c["good"], "good_rate": c["good"] / n if n else None,
            "good_ci95": wilson(c["good"], n), "marks": dict(c),
            "per_rater_good_rate": [r["good"] / n if n else None for r in per_rater],
            "unanimous_good": sum(all(m[it["card"]] == "good" for m in marks) for it in sel),
            "by_rule": {k: {"n": len(v), "good": v.count("good"), "good_ci95": wilson(v.count("good"), len(v)),
                            "marks": dict(Counter(v))} for k, v in by_rule.items()},
            "by_source": {k: {"n": len(v), "good": v.count("good")} for k, v in by_source.items()},
            "edits_per_100_sentences": 100 * info.get("changed", 0) / info["sentences"] if info.get("sentences") else None,
            "changed": info.get("changed"), "sentences": info.get("sentences"),
        }
    cards = sorted({it["card"] for it in items})
    agree = {"all_three": sum(len({m[c] for m in marks}) == 1 for c in cards) / len(cards),
             "pairwise": sum(marks[a][c] == marks[b][c] for c in cards for a in range(len(marks))
                             for b in range(a + 1, len(marks))) / (len(cards) * 3)}
    return out, agree


def run_tally(args):
    state = outside_checkout(args.state)
    key = json.loads((state / f"{args.name}-key.json").read_text())
    marks = [read_marks(state / args.name / f"rater{r}" / "marks.csv") for r in range(1, args.raters + 1)]
    cards = {it["card"] for it in key["items"]}
    for k, m in enumerate(marks, 1):
        missing = cards - set(m)
        if missing:
            raise SystemExit(f"Rater {k} is missing {len(missing)} cards")
    arms = list(dict.fromkeys(it["arm"] for it in key["items"]))
    rates = key["per_arm"]
    if args.rates_from:
        # Edits per 100 sentences from the standard pool's apply file (rare
        # arms draw extra items from a wider pool, but their rate is not from it).
        recs = load_apply(state / args.rates_from)
        rates = {a: {**key["per_arm"].get(a, {}), "changed": sum(r["changed"] for r in recs[a].values()),
                     "sentences": len(recs[a])} for a in arms}
    items = key["items"]
    if args.gate:
        # A derived arm: the rated cards of `arm` whose every edit passes the
        # gate (no extra cards). Its per-100 rate applies the gate to the apply file.
        from cleartext.structure_v2.gate import load_gate
        arm, path = args.gate.split(":", 1)
        gate = load_gate(path)
        recs = load_apply(state / (args.rates_from or args.applied_for_gate))
        passes = {i: bool(r["accepted"]) and all(gate(e["features"]) for e in r["accepted"])
                  for i, r in recs[arm].items() if r["changed"]}
        derived = [{**it, "arm": arm + "_gated"} for it in items if it["arm"] == arm and passes.get(it["id"])]
        items = items + derived
        arms = arms + [arm + "_gated"]
        rates[arm + "_gated"] = {"changed": sum(passes.values()), "sentences": len(recs[arm])}
    result, agree = tally(items, marks, arms, rates)
    out = {"panel": args.name, "arms": result, "agreement": agree, "cards": len(cards),
           "key_sha256": sha(state / f"{args.name}-key.json"),
           "marks_sha256": [sha(state / args.name / f"rater{r}" / "marks.csv") for r in range(1, args.raters + 1)]}
    write_json(Path(args.out), out)
    for arm, r in result.items():
        lo, hi = r["good_ci95"]
        print(f"{arm:20s} n={r['n']:3d} good={r['good_rate']:.2f} [{lo:.2f},{hi:.2f}] "
              f"per100={r['edits_per_100_sentences'] or 0:.2f} {r['marks']}")
    print("agreement", agree)


# --- dev (test set v3 dev split, on top of F2) --------------------------------------

def locked_ids():
    ids = set()
    for name in ("test_ids.txt", "onestop_test_ids.txt"):
        ids |= set((ROOT / "evals/data/testset_v3" / name).read_text().split())
    return ids


def f2_dev_outputs():
    """{passage id: F2 output} from the parameter-sweep dev state's cache."""
    from .checkpoints import digest
    dev = json.loads((V3 / "panel/dev.json").read_text())
    if {p["id"] for p in dev} & locked_ids():
        raise SystemExit("dev.json contains locked test IDs")
    sig = next((SWEEP_DEV / "outputs").iterdir()).name
    out = {}
    for p in dev:
        path = SWEEP_DEV / "outputs" / sig / (digest([p["id"], F2]) + ".json")
        d = json.loads(path.read_text())
        if d["identity"]["config"] != F2 or d["identity"]["id"] != p["id"]:
            raise ValueError("F2 cache identity mismatch")
        out[p["id"]] = d["output"]
    return dev, out


def run_dev(args):
    state = outside_checkout(args.state)
    dev, f2 = f2_dev_outputs()
    nli = SmallNLI(state / "small-nli.jsonl") if args.min_entailment is not None else None
    gate = None
    if args.gate:
        from cleartext.structure_v2.gate import load_gate
        gate = load_gate(args.gate)
    rows = {}
    path = state / f"dev-{args.name}.json"
    if path.exists():
        rows = json.loads(path.read_text())
    if args.ops in (["chhoeng_agentive"], ["chhoeng_auto"]):
        arm = chhoeng_agentive(None if args.ops == ["chhoeng_agentive"] else "auto")
        stage = type("S", (), {"rewrite": lambda self, t: _passive_passage(arm, t)})()
    else:
        from cleartext.structure_v2 import StructureV2
        from cleartext.structure_v2.stage import build
        operators = [build(o) for o in args.ops]
        for o, op in zip(args.ops, operators):
            if o.startswith("passive") and gate is not None:
                op.gate = gate
        stage = StructureV2(ops=args.ops, operators=operators, entailment=nli, min_entailment=args.min_entailment)
    for p in dev:
        if p["id"] in rows:
            continue
        text, edits = stage.rewrite(f2[p["id"]])
        rows[p["id"]] = {"output": text, "edits": edits}
    write_json(path, rows)
    changed = sum(rows[p["id"]]["output"] != f2[p["id"]] for p in dev)
    print(f"{args.name}: {changed} passages changed, {sum(len(r['edits']) for r in rows.values())} edits")


def _passive_passage(arm, text):
    from cleartext.structure_v2.common import parse
    parts, edits, offset = [], [], 0
    for s in parse(text).sents:
        new, acc, _ = arm.rewrite_sentence(s.text)
        parts += [text[offset:s.start_char], new]
        offset = s.end_char
        edits += acc
    parts.append(text[offset:])
    return "".join(parts), edits


def cached_f2_metric(source, output):
    """The sweep dev state's cached metrics for a (source, output) pair, if any."""
    from .checkpoints import digest
    for sig_dir in (SWEEP_DEV / "metrics").iterdir():
        path = sig_dir / (digest([source, output]) + ".json")
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("pair") == {"source": source, "output": output}:
                return {k: value[k] for k in ("readability", "preservation", "sentence_nli")}
    return None


def scoring_plan():
    plan = json.loads((SWEEP_DEV / "plan.json").read_text())
    return {"evaluation": plan["evaluation"]}


def run_score(args):
    """Evaluation metrics (Week 7 readability/preservation + sentence NLI)
    for each dev arm's changed passages; unchanged passages reuse F2's cached
    metrics from the sweep dev state (same scoring sources)."""
    from . import ensemble_study as S
    state = outside_checkout(args.state)
    dev, f2 = f2_dev_outputs()
    import torch
    torch.set_num_threads(1)
    scorer = None
    started = time.monotonic()
    metrics_path = state / "dev-metrics.jsonl"
    have = set()
    if metrics_path.exists():
        have = {(r["source"], r["output"]) for r in map(json.loads, metrics_path.read_text().splitlines())}
    # Reproduction check: three F2 passages are rescored here and must match the cache.
    todo = [(p["original"], f2[p["id"]]) for p in dev[:3]]
    for name in args.arms:
        rows = json.loads((state / f"dev-{name}.json").read_text())
        todo += [(p["original"], rows[p["id"]]["output"]) for p in dev]
    with metrics_path.open("a") as f:
        for source, output in todo:
            if (source, output) in have:
                continue
            if time.monotonic() - started > args.max_seconds:
                print("PARTIAL", flush=True)
                return
            if scorer is None:
                plan = scoring_plan()
                scorer = S.StudyScorer(plan, state, "structure-v2-dev", threads=1)
            m = scorer(source, output)
            f.write(json.dumps({"source": source, "output": output, "metrics": m}, ensure_ascii=False) + "\n")
            f.flush()
            have.add((source, output))
    print("COMPLETE", flush=True)


def run_judge(args):
    from .testset_v3 import gates, over_edit_gate, paired, passage_row, summarize
    state = outside_checkout(args.state)
    dev, f2 = f2_dev_outputs()
    metrics = {(r["source"], r["output"]): r["metrics"]
               for r in map(json.loads, (state / "dev-metrics.jsonl").read_text().splitlines())}
    repro = []
    for p in dev:
        cached = cached_f2_metric(p["original"], f2[p["id"]])
        key = (p["original"], f2[p["id"]])
        if cached is None:
            continue
        if key in metrics:
            repro.append(abs(metrics[key]["preservation"]["nli_mean"] - cached["preservation"]["nli_mean"]) < 1e-6
                         and abs(metrics[key]["readability"]["flesch_kincaid_grade"]
                                 - cached["readability"]["flesch_kincaid_grade"]) < 1e-9)
        else:
            metrics[key] = cached
    if not repro or not all(repro):
        raise SystemExit(f"F2 metric reproduction failed or missing: {repro}")
    systems = {"F2": {p["id"]: f2[p["id"]] for p in dev}}
    for name in args.arms:
        rows = json.loads((state / f"dev-{name}.json").read_text())
        systems[name] = {p["id"]: rows[p["id"]]["output"] for p in dev}
    rows = {}
    for name, outs in systems.items():
        missing = [p["id"] for p in dev if (p["original"], outs[p["id"]]) not in metrics]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} passages unscored; never drop missing outputs")
        rows[name] = [passage_row(p, outs[p["id"]], metrics[(p["original"], outs[p["id"]])]) for p in dev]
    rb_entities = 1.0  # rule_based scores 1.0 on dev (param-sweeps README, gate 4)
    result = {}
    for name, r in rows.items():
        s = summarize(r)
        res = {"summary": s, "gates": gates(s, rb_entities),
               "over_edit": over_edit_gate(summarize([x for x in r if x["kind"] == "simple_control"])),
               "changed_vs_F2": sum(systems[name][p["id"]] != f2[p["id"]] for p in dev)}
        if name != "F2":
            res["vs_F2"] = {m: paired(r, rows["F2"], m) for m in ("fk_grade", "kea_S", "nli_mean",
                                                                   "sent_nli_mean", "entities")}
            changed = [k for k, p in enumerate(dev) if systems[name][p["id"]] != f2[p["id"]]]
            res["changed_only"] = {
                "n": len(changed),
                "nli_mean": sum(r[k]["nli_mean"] for k in changed) / len(changed) if changed else None,
                "f2_nli_mean": sum(rows["F2"][k]["nli_mean"] for k in changed) / len(changed) if changed else None,
                "fk": sum(r[k]["fk_grade"] for k in changed) / len(changed) if changed else None,
                "f2_fk": sum(rows["F2"][k]["fk_grade"] for k in changed) / len(changed) if changed else None,
                "newly_below_080": sum(r[k]["nli_mean"] < 0.80 <= rows["F2"][k]["nli_mean"] for k in changed),
            }
        result[name] = res
    per_passage = {name: [{k: x[k] for k in ("id", "fk_grade", "kea_S", "nli_mean", "entities", "changed")}
                          for x in r] for name, r in rows.items()}
    write_json(Path(args.out), {"systems": result, "per_passage": per_passage,
                                "f2_metrics_reproduced": f"{sum(repro)}/{len(repro)}"})
    for name, r in result.items():
        s = r["summary"]
        d = r.get("vs_F2", {})
        print(f"{name:28s} FK {s['fk_grade']:.3f} S {s['kea_S']:.4f} NLI {s['nli_mean']:.4f} "
              f"<0.8 {s['nli_below_080']} ent {s['entities']:.4f} num/neg {s['numbers_failed']}/{s['negations_failed']} "
              f"gates {all(r['gates'].values())} changed {r['changed_vs_F2']} "
              + (f"dFK {d['fk_grade']['mean']:+.3f} {d['fk_grade']['ci95']} dNLI {d['nli_mean']['mean']:+.4f} "
                 f"{d['nli_mean']['ci95']}" if d else ""))


def run_synthetic(args):
    """Synthetic active->passive pairs for the passive gate: development
    sentences (document buckets 0-2 only, never pool A/B) are passivized by
    rule and converted back; label 1 when the original returns exactly."""
    from cleartext.structure_v2.common import parse
    from cleartext.structure_v2.gate import passivize
    from cleartext.structure_v2.passive import ActiveVoice
    state = outside_checkout(args.state)
    rng = random.Random(7)
    av = ActiveVoice()
    sources = {**SOURCES, **{k: v for k, v in FRESH_SOURCES.items() if v.exists()}}
    with (state / args.out).open("w") as out:
        for name, path in sources.items():
            rows = [json.loads(line)["s"] for line in gzip.open(path, "rt")
                    if bucket(json.loads(line)["doc"]) < 3]
            rows = [r for r in rows if clean(r)]
            rng.shuffle(rows)
            for active in rows[: args.per_source]:
                passive = passivize(active)
                if not passive:
                    continue
                sent = list(parse(passive).sents)
                cands = [c for c in av.candidates(sent[0]) if c["kind"] == "agentive"] if len(sent) == 1 else []
                if not cands:
                    continue
                output, _ = av.rewrite_sentence(passive)
                out.write(json.dumps({"source": name, "active": active, "passive": passive,
                                      "features": cands[0]["features"], "reason": cands[0]["reason"],
                                      "output": output, "label": int(output == active)}) + "\n")
    print("COMPLETE")


def run_gate_train(args):
    from cleartext.structure_v2.gate import train
    state = outside_checkout(args.state)
    rows = [json.loads(line) for line in (state / args.synthetic).read_text().splitlines()]
    rows = [r for r in rows if r["reason"] is None]
    report = train(rows, state / args.out, threshold_precision=args.precision)
    print(json.dumps(report, indent=1))


def run_precommit(args):
    """Hashes of the sources, brief and pools a round depends on, plus its
    thresholds, written before the round's outputs exist."""
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "models/src/cleartext/structure_v2").glob("*.py"))
    files += ["evals/structure_v2.py", "evals/reports/20261122-structure-v2/RATER_BRIEF.md",
              "evals/reports/20261122-structure-v2/RATER_PROMPT.md", *args.extra]
    out = {"round": args.round, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "source_sha256": {f: sha(ROOT / f) for f in files},
           "inputs_sha256": {str(p): sha(p) for p in args.inputs},
           "thresholds": json.loads(args.thresholds)}
    write_json(Path(args.out), out)
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--per-source", type=int, default=1500)
    p.add_argument("--per-doc", type=int, default=40)
    p = sub.add_parser("apply")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--ops", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--min-entailment", type=float)
    p.add_argument("--gate", type=Path)
    p.add_argument("--max-seconds", type=float, default=1600)
    p = sub.add_parser("panel")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--applied", required=True)
    p.add_argument("--ops", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--per-op", type=int, default=40)
    p.add_argument("--cap", nargs="*", help="arm=N overrides of --per-op")
    p.add_argument("--raters", type=int, default=3)
    p.add_argument("--seed", type=int, default=SEED)
    p = sub.add_parser("tally")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--raters", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rates-from", help="apply file (in --state) for edits per 100 sentences")
    p.add_argument("--gate", help="ARM:GATE.pkl adds ARM_gated, the rated ARM cards the gate keeps")
    p.add_argument("--applied-for-gate", help="apply file for the gate when --rates-from is not given")
    p = sub.add_parser("dev")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--ops", nargs="+", required=True)
    p.add_argument("--min-entailment", type=float)
    p.add_argument("--gate", type=Path)
    p = sub.add_parser("score")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--max-seconds", type=float, default=1600)
    p = sub.add_parser("judge")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("synthetic")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-source", type=int, default=4000)
    p = sub.add_parser("gate-train")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--synthetic", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--precision", type=float, default=0.92)
    p = sub.add_parser("precommit")
    p.add_argument("--round", required=True)
    p.add_argument("--inputs", nargs="*", type=Path, default=[])
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--thresholds", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    {"pool": run_pool, "apply": run_apply, "panel": run_panel, "tally": run_tally, "dev": run_dev,
     "score": run_score, "judge": run_judge, "precommit": run_precommit,
     "synthetic": run_synthetic, "gate-train": run_gate_train}[args.cmd](args)


if __name__ == "__main__":
    main()
