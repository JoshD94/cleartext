"""Does the NLI meaning test agree with humans on single-word swaps? (November 17, 2026)

Calibrates the evaluation NLI (DeBERTa-v3-base-mnli-fever-anli, both
directions) and the word stage's entailment veto (Chhoeng's
nli-deberta-v3-small, min of both directions) against human judgments of
lexical substitutes, then re-gates word-swap outputs that earlier studies
already generated and scored. Nothing here generates a simplification; no LLM
is used anywhere.

Subcommands:
  gold      calibration pairs from SWORDS test (human-rated acceptable and
            unacceptable substitutes), TSAR-2022 test gold and MultiLS trial
            gold (positives) plus random same-POS, frequency-matched words
            (easy negatives)
  swaps     one row per word swap in existing outputs, with the sentence
            before and after, from the output and its words=off twin
  score     checkpointed NLI scoring of gold and swap pairs (bounded batches)
  analyze   aggregate calibration and re-gating tables (no raw text)
  review    phone-readable HTML of sampled swaps for Joshua to mark
  marks     NLI precision/recall per threshold from Joshua's exported marks

Raw text (sentences, outputs, scores per pair) stays in the local state
directory; only aggregates go to Git. The locked test split is never read:
every panel goes through ensemble_study.load_panel, which refuses locked IDs.
"""

import argparse
import csv
import gzip
import hashlib
import html
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HUB = Path.home() / ".cache/huggingface/hub"
BASE_NLI = HUB / "models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli/snapshots/6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
SMALL_NLI = HUB / "models--cross-encoder--nli-deberta-v3-small/snapshots/fa2804872c3b4bd748f38c0185cc85775361e735"
THRESHOLDS = (0.97, 0.95, 0.9, 0.8, 0.7, 0.575, 0.5)
SEED = 20261117
SCORES = ("base_mean", "base_min", "small_min")
BANDS = ((0.95, 1.01, "≥0.95"), (0.9, 0.95, "0.90–0.95"), (0.8, 0.9, "0.80–0.90"), (-0.01, 0.8, "<0.80"))
_WORD = re.compile(r"\w+|[^\w\s]")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def norm(text):
    return " ".join(text.split())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# --- gold calibration pairs ---------------------------------------------------

def _nlp():
    from cleartext.features import nlp
    return nlp()


def _inflect(lemma, tag, original):
    """cleartext.generation.inflect for one-word substitutes; None if no form."""
    from cleartext.generation import inflect
    return inflect(lemma, tag, original)


def _sentence_at(text, offset):
    """(start, end) of the sentence containing offset (preservation.split_sentences)."""
    from .preservation import split_sentences
    for a, b in split_sentences(text):
        if a <= offset < b:
            return a, b
    return 0, len(text)


def swords_items(path, nlp, per_target=(4, 4, 2), rng=None):
    """SWORDS: positives score >= 0.5 (share of annotators saying acceptable,
    10 labels), hard negatives score 0, borderline 0 < score < 0.5. Single-word
    substitutes only, inflected to the target's tag."""
    rng = rng or random.Random(SEED)
    data = json.load(gzip.open(path))
    subs = defaultdict(list)
    for sid, s in data["substitutes"].items():
        labels = data["substitute_labels"][sid]
        score = sum(label == "TRUE" for label in labels) / len(labels)
        if " " in s["substitute"].strip() or not s["substitute"].isalpha():
            continue
        subs[s["target_id"]].append((sid, s["substitute"].strip(), score, len(labels)))
    items = []
    for tid, target in sorted(data["targets"].items()):
        context = data["contexts"][target["context_id"]]["context"]
        offset, word = target["offset"], target["target"]
        if context[offset:offset + len(word)] != word or " " in word:
            continue
        a, b = _sentence_at(context, offset)
        sentence = context[a:b]
        local = offset - a
        doc = nlp(sentence)
        token = next((t for t in doc if t.idx == local), None)
        if token is None or token.text != word:
            continue
        groups = {"pos": [], "neg": [], "border": []}
        for sid, lemma, score, n in sorted(subs[tid]):
            if lemma.lower() == word.lower() or lemma.lower() == token.lemma_.lower():
                continue
            kind = "pos" if score >= 0.5 and n >= 10 else "neg" if score == 0 else "border"
            groups[kind].append((sid, lemma, score, n))
        for kind, cap in zip(("pos", "neg", "border"), per_target):
            chosen = rng.sample(groups[kind], min(cap, len(groups[kind])))
            for sid, lemma, score, n in chosen:
                form = _inflect(lemma, token.tag_, word)
                if not form or " " in form:
                    continue
                after = sentence[:local] + form + sentence[local + len(word):]
                items.append({"id": f"swords:{sid}", "dataset": "swords_test", "kind": kind,
                              "human": score, "labels": n, "pos": target["pos"], "target": word,
                              "substitute": form, "group": tid,
                              "before": norm(sentence), "after": norm(after)})
    return items


def _replace_word(sentence, target, substitute):
    m = re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", sentence)
    if m is None:
        return None
    return sentence[:m.start()] + substitute + sentence[m.end():]


def _wordnet_pool(nlp):
    """Base-form content words (wordfreq top 30k) by WordNet POS, with zipf."""
    from nltk.corpus import wordnet as wn
    from wordfreq import top_n_list, zipf_frequency
    pool = defaultdict(list)
    for w in top_n_list("en", 30000):
        if not w.isalpha() or len(w) < 3:
            continue
        poses = {s.pos().replace("s", "a") for s in wn.synsets(w)}
        for p in poses:
            if w in {l.name().lower() for s in wn.synsets(w, pos=p) for l in s.lemmas()}:
                pool[p].append((w, zipf_frequency(w, "en")))
    return pool


_COARSE = {"NOUN": "n", "PROPN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


def ls_items(name, instances, nlp, pool, rng, min_count=2, singletons=1, easy=2):
    """TSAR/MultiLS: gold substitutes named by >= min_count annotators are
    positives ("pos"); one random singleton per instance ("pos1"); `easy`
    random same-POS words within 0.5 zipf of the top gold substitute are
    easy negatives ("easy_neg")."""
    from wordfreq import zipf_frequency
    items = []
    for k, inst in enumerate(instances):
        doc = nlp(inst.sentence)
        token = next((t for t in doc if t.text == inst.target), None)
        gold = {s.lower() for s, _ in inst.ranked} | {inst.target.lower()}
        chosen = [(s, c, "pos") for s, c in inst.ranked if c >= min_count]
        ones = [(s, c, "pos1") for s, c in inst.ranked if c == 1]
        chosen += rng.sample(ones, min(singletons, len(ones)))
        coarse = _COARSE.get(token.pos_) if token is not None else None
        if coarse and inst.ranked:
            z = zipf_frequency(inst.ranked[0][0], "en")
            near = [w for w, wz in pool[coarse] if abs(wz - z) <= 0.5 and w not in gold]
            for w in rng.sample(near, min(easy, len(near))):
                form = _inflect(w, token.tag_, inst.target)
                if form and " " not in form and form.lower() not in gold:
                    chosen.append((form, 0, "easy_neg"))
        for sub, count, kind in chosen:
            after = _replace_word(inst.sentence, inst.target, sub)
            if after is None or " " in sub.strip():
                continue
            items.append({"id": f"{name}:{k}:{sub}", "dataset": name, "kind": kind,
                          "human": count, "pos": token.pos_ if token is not None else None,
                          "target": inst.target, "substitute": sub, "group": f"{name}:{k}",
                          "before": norm(inst.sentence), "after": norm(after)})
    return items


def build_gold(args):
    from . import datasets as D
    rng = random.Random(SEED)
    nlp = _nlp()
    import nltk
    nltk.data.path[:] = [str(Path.home() / "nltk_data")]
    pool = _wordnet_pool(nlp)
    items = swords_items(args.swords, nlp, rng=rng)
    bench = ROOT / "evals/data/benchmarks"
    items += ls_items("tsar2022_test", D.load_tsar2022_en(bench / "tsar2022_en_test_gold.tsv"), nlp, pool, rng)
    items += ls_items("multils_trial", D.load_multils_en_ls(bench / "multils2024_en_trial_ls.tsv"), nlp, pool, rng)
    seen, unique = set(), []
    for it in items:
        if it["id"] not in seen and it["before"] != it["after"]:
            seen.add(it["id"])
            unique.append(it)
    write_jsonl(args.out, unique)
    print(json.dumps(Counter((i["dataset"], i["kind"]) for i in unique).most_common(), indent=0))


# --- swaps in existing outputs ------------------------------------------------

def twin_name(name):
    """The same config with words=off and no w.* parameters."""
    head, _, tail = name.partition("#")
    parts = dict(p.split("=", 1) for p in head.split("|"))
    parts["words"] = "off"
    params = {k: v for k, v in (json.loads(tail) if tail else {}).items() if not k.startswith("w.")}
    twin = "|".join(f"{k}={v}" for k, v in parts.items())
    return twin + ("#" + json.dumps(params, sort_keys=True, separators=(",", ":")) if params else "")


def single_swap(before, after):
    """(char start/end in before, in after) of the one changed token run, or None."""
    import difflib
    ta = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(before)]
    tb = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(after)]
    ops = [o for o in difflib.SequenceMatcher(None, [t[0] for t in ta], [t[0] for t in tb],
                                             autojunk=False).get_opcodes() if o[0] != "equal"]
    if len(ops) != 1 or ops[0][0] != "replace":
        return None
    _, i1, i2, j1, j2 = ops[0]
    a, b = (ta[i1][1], ta[i2 - 1][2]), (tb[j1][1], tb[j2 - 1][2])
    if before[:a[0]] != after[:b[0]] or before[a[1]:] != after[b[1]:]:
        return None
    return a, b


def state_outputs(state):
    """{config: {passage id: output record}} for every output in a state."""
    found = defaultdict(dict)
    for path in (Path(state) / "outputs").glob("*/*.json"):
        d = json.loads(path.read_text())
        found[d["identity"]["config"]][d["identity"]["id"]] = d
    return found


SOURCES = {
    # label: (state, panel name in its plan, word configs: None = every words!=off config)
    "dev": ("param-sweeps-20261116/dev", "dev"),
    "old90_sweep_smoke": ("param-sweeps-20261116/smoke", "smoke"),
    "old90_v2": ("ensemble-v2-20261110/run1", "sourced"),
    "old90_nov9": ("ensemble-study-20261109/run1", "sourced"),
    "paper": ("paper-run-20261116/run1", "paper"),
}
STATE_ROOT = Path.home() / ".local/state/cleartext"
PANEL_FILES = {  # the plan's panel paths point into removed worktrees; same SHA-256 checked below
    "smoke": ROOT.parent / "cleartext/evals/data/week7_sourced_passages.json",
    "sourced": ROOT.parent / "cleartext/evals/data/week7_sourced_passages.json",
}


def load_source_panel(label):
    from . import ensemble_study as S
    state, panel = SOURCES[label]
    plan = json.loads((STATE_ROOT / state / "plan.json").read_text())
    path = Path(plan["panels"][panel])
    if not path.exists():
        path = PANEL_FILES[panel]
    if S.sha(path) != plan["panel_sha256"][panel]:
        raise SystemExit(f"{label}: panel file {path} does not match the plan's SHA-256")
    rows = S.load_panel({"panels": {panel: str(path)}}, panel)  # refuses locked test IDs
    return plan, {r["id"]: r for r in rows}


def build_swaps(args):
    rows, skipped = [], Counter()
    for label in SOURCES:
        _, passages = load_source_panel(label)
        outputs = state_outputs(STATE_ROOT / SOURCES[label][0])
        for name, by_id in sorted(outputs.items()):
            if not name.startswith("words=") or name.startswith("words=off"):
                continue
            twin = outputs.get(twin_name(name), {})
            for pid, rec in sorted(by_id.items()):
                if pid not in passages:
                    skipped["not in panel"] += 1
                    continue
                words = [e for e in rec.get("edits", []) if e.get("stage") == "word"]
                if not words:
                    continue
                if pid not in twin:
                    skipped["no twin output"] += 1
                    continue
                before, after = twin[pid]["output"], rec["output"]
                span = single_swap(before, after) if len(words) == 1 else None
                if span is None:
                    skipped["not a single swap vs twin" if len(words) == 1 else "multiple word edits"] += 1
                    continue
                (a0, a1), (b0, b1) = span
                s0, s1 = _sentence_at(before, a0)
                delta = len(after) - len(before)
                sb, sa = before[s0:s1], after[s0:s1 + delta]
                rows.append({"id": f"{label}:{pid}:{sha(name)[:10]}", "source": label, "passage": pid,
                             "config": name, "twin": twin_name(name), "domain": passages[pid].get("domain"),
                             "word": before[a0:a1], "replacement": after[b0:b1],
                             "edit": words[0], "before": norm(sb), "after": norm(sa),
                             "swap_key": sha(norm(sb) + "\n" + norm(sa))})
    write_jsonl(args.out, rows)
    print(len(rows), "swaps;", len({r["swap_key"] for r in rows}), "unique sentence pairs; skipped", dict(skipped))


# --- scoring ------------------------------------------------------------------

class Scorer:
    """Entailment probabilities memoized per (model, premise, hypothesis) in
    an append-only JSONL file.

    base: the evaluation NLI's entailment probability, computed with the model
    in float32 (the evaluator's transformers pipeline loads the float16
    weights as float16, which is ~25x slower on this CPU; `fidelity` records
    the measured difference). small: the CrossEncoder softmax that
    phrase_replacer.entailment computes, batched."""

    def __init__(self, memo, threads=1):
        import torch
        torch.set_num_threads(threads)
        self.memo_path = Path(memo)
        self.memo = {r["k"]: r["v"] for r in read_jsonl(self.memo_path)}
        self.new = []
        self._base = self._small = None

    def key(self, model, premise, hypothesis):
        return sha(f"{model}\n{premise}\n{hypothesis}")

    def flush(self):
        if self.new:
            with self.memo_path.open("a") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in self.new)
            self.new = []

    def _put(self, k, v):
        self.memo[k] = v
        self.new.append({"k": k, "v": v})

    def base_model(self):
        if self._base is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(BASE_NLI))
            model = AutoModelForSequenceClassification.from_pretrained(str(BASE_NLI), dtype=torch.float32).eval()
            entail = next(i for i, n in model.config.id2label.items() if "entail" in n.lower())
            self._base = (tok, model, entail)
        return self._base

    def small_model(self):
        if self._small is None:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(str(SMALL_NLI))
            labels = {i: n.lower() for i, n in model.model.config.id2label.items()}
            self._small = (model, next(i for i, n in labels.items() if n.startswith("entail")))
        return self._small

    def score(self, model, pairs, batch=16):
        todo = sorted({(p, h) for p, h in pairs if self.key(model, p, h) not in self.memo},
                      key=lambda x: len(x[0]) + len(x[1]))
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            for (p, h), v in zip(chunk, self._run(model, chunk)):
                self._put(self.key(model, p, h), v)
        return [self.memo[self.key(model, p, h)] for p, h in pairs]

    def _run(self, model, chunk):
        import torch
        if model == "base":
            tok, net, entail = self.base_model()
            enc = tok([p for p, _ in chunk], [h for _, h in chunk], return_tensors="pt", padding=True,
                      truncation=False)
            if enc["input_ids"].shape[1] > 512:
                return [None if len(tok(p, h)["input_ids"]) > 512 else self._run(model, [(p, h)])[0]
                        for p, h in chunk]
            with torch.inference_mode():
                probs = torch.softmax(net(**enc).logits.float(), -1)[:, entail]
            return [float(x) for x in probs]
        net, entail = self.small_model()
        logits = np.asarray(net.predict(chunk, batch_size=len(chunk), show_progress_bar=False))
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        return [float(x) for x in probs[:, entail]]


def pair_scores(scorer, rows):
    fwd = scorer.score("base", [(r["before"], r["after"]) for r in rows])
    bwd = scorer.score("base", [(r["after"], r["before"]) for r in rows])
    sf = scorer.score("small", [(r["before"], r["after"]) for r in rows])
    sb = scorer.score("small", [(r["after"], r["before"]) for r in rows])
    out = []
    for f, b, x, y in zip(fwd, bwd, sf, sb):
        ok = f is not None and b is not None
        out.append({"base_fwd": f, "base_bwd": b, "base_mean": (f + b) / 2 if ok else None,
                    "base_min": min(f, b) if ok else None, "small_min": min(x, y)})
    return out


def run_score(args):
    state = Path(args.state)
    deadline = time.monotonic() + args.max_seconds
    scorer = Scorer(state / "nli-memo.jsonl", args.threads)
    rows = [r for f in args.items for r in read_jsonl(f)]
    done = 0
    for i in range(0, len(rows), 128):
        if time.monotonic() > deadline:
            break
        pair_scores(scorer, rows[i:i + 128])
        scorer.flush()
        done = i + 128
        print(f"{time.strftime('%H:%M:%S')} scored {min(done, len(rows))}/{len(rows)}", flush=True)
    complete = done >= len(rows)
    fidelity = state / "fidelity.json"
    if complete and not fidelity.exists():
        fidelity.write_text(json.dumps(check_fidelity(scorer, rows), indent=1))
    print("COMPLETE" if complete else "PARTIAL", flush=True)


def check_fidelity(scorer, rows, n=60):
    """Max |difference| between this module's scores and the pipeline's own
    functions (preservation.entailment_probability, EntailmentVeto.score)."""
    from cleartext.teammate_stages import EntailmentVeto
    from .preservation import entailment_probability
    rng = random.Random(SEED)
    sample = rng.sample(rows, min(n, len(rows)))
    veto = EntailmentVeto(model=str(SMALL_NLI))
    mine = pair_scores(scorer, sample)
    base_diff = max(abs(entailment_probability(r["before"], r["after"], str(BASE_NLI)) - m["base_fwd"])
                    for r, m in zip(sample, mine))
    small_diff = max(abs(veto.score(r["before"], r["after"]) - m["small_min"]) for r, m in zip(sample, mine))
    return {"n": len(sample), "base_fwd_max_abs_diff_vs_pipeline_fp16": base_diff,
            "small_min_max_abs_diff_vs_EntailmentVeto": small_diff}


# --- analysis -----------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney)."""
    if not pos or not neg:
        return None
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def clustered_auc_ci(rows, score, pos_kind, neg_kind, resamples=2000):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(resamples):
        pick = [r for k in rng.choice(len(keys), len(keys)) for r in groups[keys[k]]]
        vals.append(auc([r[score] for r in pick if r["kind"] == pos_kind],
                        [r[score] for r in pick if r["kind"] == neg_kind]))
    vals = [v for v in vals if v is not None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def rates(values, thresholds=THRESHOLDS):
    """Share of values below each threshold (k, n, rate, Wilson CI)."""
    out = {}
    for t in thresholds:
        k = sum(v < t for v in values)
        out[str(t)] = {"k": k, "n": len(values), "rate": k / len(values) if values else None,
                       "ci": wilson(k, len(values))}
    return out


def youden(pos, neg):
    """Threshold maximizing TPR - FPR (accept when score >= t)."""
    best = (-1, None)
    for t in sorted(set(np.round(np.concatenate([pos, neg]), 4))):
        j = np.mean(np.asarray(pos) >= t) - np.mean(np.asarray(neg) >= t)
        if j > best[0]:
            best = (float(j), float(t))
    return {"J": best[0], "threshold": best[1]}


def calibration(gold):
    out = {"counts": Counter(f"{r['dataset']}/{r['kind']}" for r in gold), "by_score": {}}
    from scipy.stats import spearmanr
    for score in SCORES:
        rows = [r for r in gold if r[score] is not None]
        res = {}
        for ds in ("swords_test", "tsar2022_test", "multils_trial", "all_gold_positives"):
            sel = rows if ds == "all_gold_positives" else [r for r in rows if r["dataset"] == ds]
            pos = [r[score] for r in sel if r["kind"] == "pos"]
            res[ds] = {"positives_rejected": rates(pos)}
            if ds != "all_gold_positives":
                for kind in ("pos1", "border", "neg", "easy_neg"):
                    vals = [r[score] for r in sel if r["kind"] == kind]
                    if vals:
                        accepted = {t: {"rate": 1 - v["rate"], "k": v["n"] - v["k"], "n": v["n"],
                                        "ci": [1 - v["ci"][1], 1 - v["ci"][0]]} for t, v in rates(vals).items()}
                        res[ds][f"{kind}_accepted"] = accepted
                for kind in ("neg", "easy_neg"):
                    neg = [r[score] for r in sel if r["kind"] == kind]
                    if pos and neg:
                        res[ds][f"auc_pos_vs_{kind}"] = auc(pos, neg)
                        res[ds][f"auc_pos_vs_{kind}_ci"] = clustered_auc_ci(sel, score, "pos", kind)
                        res[ds][f"youden_pos_vs_{kind}"] = youden(pos, neg)
        sw = [r for r in rows if r["dataset"] == "swords_test" and r["labels"] >= 10]
        res["swords_spearman_vs_human_share"] = float(spearmanr([r[score] for r in sw],
                                                                [r["human"] for r in sw]).statistic)
        by_pos = {}
        for p in ("NOUN", "VERB", "ADJ", "ADV"):
            sel = [r for r in rows if r["dataset"] == "swords_test" and r["pos"] == p]
            pos = [r[score] for r in sel if r["kind"] == "pos"]
            neg = [r[score] for r in sel if r["kind"] == "neg"]
            by_pos[p] = {"n_pos": len(pos), "n_neg": len(neg), "auc": auc(pos, neg),
                         "pos_rejected_at_0.95": float(np.mean(np.asarray(pos) < 0.95)) if pos else None}
        res["swords_by_pos"] = by_pos
        # Human-share curve: mean score and share >= 0.9 by acceptability bin.
        bins = defaultdict(list)
        for r in sw:
            bins[round(r["human"], 1)].append(r[score])
        res["swords_by_human_share"] = {str(k): {"n": len(v), "mean": float(np.mean(v)),
                                                  "share_ge_0.9": float(np.mean(np.asarray(v) >= 0.9))}
                                        for k, v in sorted(bins.items())}
        out["by_score"][score] = res
    return out


def passage_rows(label, plan, passages, outputs, name, ids, metric_roots):
    """{passage id: testset_v3.passage_row} for one config; passages without a
    cached metric are left out."""
    from .checkpoints import digest
    from .testset_v3 import passage_row
    rows = {}
    for pid in ids:
        p = dict(passages[pid])
        p.setdefault("domain", "all")
        p.setdefault("kind", "standard")
        out = outputs[name][pid]["output"]
        m = None
        for root in metric_roots:
            path = root / (digest([p["original"], out]) + ".json")
            if path.exists():
                m = json.loads(path.read_text())
                break
        if m is not None and m.get("pair") == {"source": p["original"], "output": out}:
            rows[pid] = passage_row(p, out, m)
    return rows


def metric_roots(label):
    state = STATE_ROOT / SOURCES[label][0]
    plan = json.loads((state / "plan.json").read_text())
    roots = [d for d in (state / "metrics").iterdir()]
    reuse = plan.get("reuse_metrics") or plan.get("reuse") or {}
    while reuse.get("state"):
        older = Path(reuse["state"])
        roots += [d for d in (older / "metrics").iterdir()]
        reuse = json.loads((older / "plan.json").read_text()).get("reuse") or {}
    return roots


GATES = {
    # name: (min passage NLI mean, passage floor, max share of passages below the floor)
    "current (≥0.95; ≤5% <0.80)": (0.95, 0.80, 0.05),
    "mean ≥0.90; ≤5% <0.80": (0.90, 0.80, 0.05),
    "mean ≥0.90; ≤5% <0.70": (0.90, 0.70, 0.05),
    "mean ≥0.90; ≤10% <0.80": (0.90, 0.80, 0.10),
}


def gate(rows, rb_entities, spec):
    lo, floor, share = spec
    nli = [r["nli_mean"] for r in rows]
    mean = float(np.mean([v for v in nli if v is not None]))
    below = sum(v is None or v < floor for v in nli)
    numneg = sum(r["numbers"] != 1.0 or r["negations"] != 1.0 for r in rows)
    ent = float(np.mean([r["entities"] for r in rows]))
    return mean >= lo and below <= share * len(rows) and numneg == 0 and ent >= rb_entities - 1e-12


# (label, score, threshold, explicit-check guard); "+checks" also vetoes a swap
# whose sentence loses a number, negation or entity under the evaluator's checks.
VETO_RULES = [("none", None, None, False)] + [
    (f"{sc} ≥ {t}", sc, t, False) for sc in ("base_mean", "small_min") for t in (0.97, 0.95, 0.9, 0.8, 0.7, 0.5)
] + [("small_min ≥ 0.97 +checks", "small_min", 0.97, True), ("base_mean ≥ 0.9 +checks", "base_mean", 0.9, True),
     ("checks only", "small_min", -1.0, True)]


def checks_pass(swap):
    from .preservation import explicit_checks
    c = explicit_checks(swap["before"], swap["after"])
    return all(c[k] == 1.0 for k in ("numbers_preserved", "negations_preserved", "entities_preserved"))


def swap_passes(swap, score, sc, t, guard):
    """A1/A2-style per-swap gate: sentence score >= t (and the checks, if guarded)."""
    return (score[sc] if score[sc] is not None else -1.0) >= t and (not guard or checks_pass(swap))


def regate(swaps, scores):
    """Every word config x veto rule x gate, from cached outputs and metrics."""
    from .testset_v3 import paired, summarize
    by_cfg = defaultdict(dict)
    for s in swaps:
        by_cfg[(s["source"], s["config"])][s["passage"]] = s
    results = []
    cache = {}
    for label in SOURCES:
        plan, passages = load_source_panel(label)
        outputs = state_outputs(STATE_ROOT / SOURCES[label][0])
        roots = metric_roots(label)
        for name in sorted(outputs):
            if not name.startswith("words=") or name.startswith("words=off"):
                continue
            twin = twin_name(name)
            ids = sorted(set(outputs[name]) & set(outputs.get(twin, {})) & set(outputs.get("rule_based", {}))
                         & set(passages))
            if not ids:
                continue
            word_rows = passage_rows(label, plan, passages, outputs, name, ids, roots)
            for ref in (twin, "rule_based"):
                if (label, ref) not in cache:
                    cache[(label, ref)] = passage_rows(label, plan, passages, outputs, ref,
                                                       sorted(set(outputs[ref]) & set(passages)), roots)
            scored = [pid for pid in ids if pid in word_rows and pid in cache[(label, twin)]
                      and pid in cache[(label, "rule_based")]]
            unscored, ids = len(ids) - len(scored), scored
            word_rows = [word_rows[pid] for pid in ids]
            twin_rows = [cache[(label, twin)][pid] for pid in ids]
            rb_rows = [cache[(label, "rule_based")][pid] for pid in ids]
            if not ids:
                continue
            rb_ent = float(np.mean([r["entities"] for r in rb_rows]))
            edited = [any(e.get("stage") == "word" for e in outputs[name][pid].get("edits", [])) for pid in ids]
            sw = by_cfg.get((label, name), {})
            for rule, sc, t, guard in VETO_RULES:
                if sc is not None and any(e and pid not in sw for e, pid in zip(edited, ids)):
                    continue  # a multi-edit or unmatched swap: per-swap veto not simulable
                rows, kept = [], 0
                for e, pid, wr, tr in zip(edited, ids, word_rows, twin_rows):
                    keep = e and (sc is None or swap_passes(sw[pid], scores[sw[pid]["swap_key"]], sc, t, guard))
                    rows.append(wr if keep or not e else tr)
                    kept += keep
                summ = summarize(rows)
                entry = {"source": label, "config": name, "veto": rule, "n": len(ids),
                         "unscored_passages": unscored, "complete_panel": len(ids) == len(passages),
                         "swaps": kept,
                         "fk": summ["fk_grade"], "kea_S": summ["kea_S"], "nli_mean": summ["nli_mean"],
                         "nli_below_080": summ["nli_below_080"],
                         "nli_below_070": sum(r["nli_mean"] is None or r["nli_mean"] < 0.70 for r in rows),
                         "numbers_failed": summ["numbers_failed"], "negations_failed": summ["negations_failed"],
                         "entities": summ["entities"], "rule_based_entities": rb_ent,
                         "gates": {g: gate(rows, rb_ent, spec) for g, spec in GATES.items()}}
                for metric in ("fk_grade", "kea_S", "nli_mean"):
                    d = paired(rows, twin_rows, metric)
                    entry[f"d_{metric}_vs_twin"] = {"mean": d["mean"], "ci95": d["ci95"]}
                results.append(entry)
    return results


def run_analyze(args):
    state = Path(args.state)
    scorer = Scorer(state / "nli-memo.jsonl")
    gold = read_jsonl(state / "gold.jsonl")
    swaps = read_jsonl(state / "swaps.jsonl")
    for rows in (gold, swaps):
        for r, s in zip(rows, pair_scores(_ReadOnly(scorer), rows)):
            r.update(s)
    scores = {s["swap_key"]: {k: s[k] for k in SCORES} for s in swaps}
    unique = list({s["swap_key"]: s for s in swaps}.values())
    swap_bands = {}
    for sc in SCORES:
        vals = [s[sc] for s in unique if s[sc] is not None]
        swap_bands[sc] = {"n_unique": len(vals), "below": {str(t): float(np.mean(np.asarray(vals) < t))
                                                           for t in THRESHOLDS}}
    out = {"calibration": calibration(gold), "pipeline_swaps": swap_bands,
           "swap_sources": Counter(s["source"] for s in unique),
           "regate": regate(swaps, scores)}
    fid = state / "fidelity.json"
    out["fidelity"] = json.loads(fid.read_text()) if fid.exists() else None
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print("wrote", args.out)


class _ReadOnly:
    """Scorer view that fails on a pair the score step did not cover."""

    def __init__(self, scorer):
        self.s = scorer

    def score(self, model, pairs):
        missing = [p for p in pairs if self.s.key(model, *p) not in self.s.memo]
        if missing:
            raise SystemExit(f"{len(missing)} {model} pairs unscored; run `score` first")
        return [self.s.memo[self.s.key(model, *p)] for p in pairs]


# --- review page --------------------------------------------------------------

def band(v):
    return next(name for lo, hi, name in BANDS if lo <= v < hi)


def highlight(text, other):
    """HTML of text with the token run that differs from other marked."""
    span = single_swap(other, text)
    if span is None:
        return html.escape(text)
    _, (b0, b1) = span
    return f"{html.escape(text[:b0])}<mark>{html.escape(text[b0:b1])}</mark>{html.escape(text[b1:])}"


def run_review(args):
    state = Path(args.state)
    scorer = _ReadOnly(Scorer(state / "nli-memo.jsonl"))
    swaps = read_jsonl(state / "swaps.jsonl")
    unique = list({s["swap_key"]: s for s in swaps}.values())
    for r, s in zip(unique, pair_scores(scorer, unique)):
        r.update(s)
    rng = random.Random(SEED)
    by_band = defaultdict(list)
    for s in unique:
        by_band[band(s["base_mean"])].append(s)
    per = args.n // len(BANDS)
    sample = []
    for _, _, name in BANDS:
        pool = sorted(by_band[name], key=lambda s: s["swap_key"])
        sample += [(name, s) for s in rng.sample(pool, min(per, len(pool)))]
    short = len(sample) < args.n  # top up from the largest band
    if short:
        rest = sorted([s for s in unique if s not in [x for _, x in sample]], key=lambda s: s["swap_key"])
        sample += [(band(s["base_mean"]), s) for s in rng.sample(rest, args.n - len(sample))]
    order = {name: k for k, (_, _, name) in enumerate(BANDS)}
    sample.sort(key=lambda x: (order[x[0]], x[1]["swap_key"]))
    key = [{"n": k + 1, "swap_key": s["swap_key"], "band": b, **{sc: s[sc] for sc in SCORES},
            "word": s["word"], "replacement": s["replacement"], "source": s["source"]}
           for k, (b, s) in enumerate(sample)]
    write_jsonl(state / "review-key.jsonl", key)
    Path(args.out).write_text(review_html(sample))
    print("wrote", args.out, Counter(b for b, _ in sample))


def review_html(sample):
    cards, current = [], None
    for k, (b, s) in enumerate(sample, 1):
        if b != current:
            current = b
            cards.append(f'<h2>NLI {html.escape(b)}</h2>')
        cards.append(f"""<section class="card" data-n="{k}">
<div class="num">{k}</div>
<p class="swap"><b>{html.escape(s['word'])}</b> → <b>{html.escape(s['replacement'])}</b></p>
<p class="lab">Before</p><p class="txt">{highlight(s['before'], s['after'])}</p>
<p class="lab">After</p><p class="txt">{highlight(s['after'], s['before'])}</p>
<div class="btns"><button data-v="good">Good</button><button data-v="bad">Bad</button></div>
</section>""")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Word swap review</title>
<style>
:root{{--bg:#fff;--fg:#111;--muted:#666;--line:#ddd;--accent:#2563eb;--good:#15803d;--bad:#b91c1c;--mark:#fde68a}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--muted:#999;--line:#333;--mark:#854d0e}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:720px;margin:0 auto;padding:16px}}
h1{{font-size:20px;margin:8px 0}} h2{{font-size:16px;margin:24px 0 8px;color:var(--muted)}}
.card{{border:1px solid var(--line);border-radius:12px;padding:12px;margin:12px 0;position:relative}}
.num{{position:absolute;top:12px;right:12px;color:var(--muted);font-size:13px}}
.lab{{margin:8px 0 0;font-size:12px;color:var(--muted);text-transform:uppercase}} .txt{{margin:2px 0}}
.swap{{margin:0 32px 0 0}} mark{{background:var(--mark);color:inherit;border-radius:3px;padding:0 2px}}
.btns{{display:flex;gap:8px;margin-top:10px}}
button{{flex:1;min-height:44px;border-radius:8px;border:1px solid var(--line);background:none;color:inherit;font:inherit}}
.card[data-mark=good] button[data-v=good]{{background:var(--good);color:#fff;border-color:var(--good)}}
.card[data-mark=bad] button[data-v=bad]{{background:var(--bad);color:#fff;border-color:var(--bad)}}
.bar{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center}}
.bar span{{flex:1;color:var(--muted)}} .bar button{{flex:0 0 auto;padding:0 16px;background:var(--accent);color:#fff;border:0}}
</style></head><body><main>
<h1>Word swap review</h1>
<p>Mark each swap <b>Good</b> if the sentence after means the same as before, else <b>Bad</b>. Then export the CSV.</p>
<div class="bar"><span id="count"></span><button id="export">Export CSV</button></div>
{''.join(cards)}
</main><script>
const KEY='cleartext-nli-review-20261117';
let marks={{}};try{{marks=JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(e){{}}
const cards=[...document.querySelectorAll('.card')];
function paint(){{cards.forEach(c=>{{const m=marks[c.dataset.n];if(m)c.dataset.mark=m;else delete c.dataset.mark}});
document.getElementById('count').textContent=Object.keys(marks).length+' of '+cards.length+' marked';
try{{localStorage.setItem(KEY,JSON.stringify(marks))}}catch(e){{}}}}
cards.forEach(c=>c.querySelectorAll('button').forEach(b=>b.onclick=()=>{{
marks[c.dataset.n]=marks[c.dataset.n]===b.dataset.v?undefined:b.dataset.v;
if(!marks[c.dataset.n])delete marks[c.dataset.n];paint()}}));
document.getElementById('export').onclick=()=>{{
const rows=['n,mark',...cards.map(c=>c.dataset.n+','+(marks[c.dataset.n]||''))].join('\\n');
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows],{{type:'text/csv'}}));
a.download='word-swap-marks.csv';a.click()}};
paint();
</script></body></html>
"""


def run_marks(args):
    key = {r["n"]: r for r in read_jsonl(Path(args.state) / "review-key.jsonl")}
    with open(args.marks, newline="") as fh:
        marks = {int(r["n"]): r["mark"].strip().lower() for r in csv.DictReader(fh) if r["mark"].strip()}
    rows = [(key[n], m == "good") for n, m in marks.items() if m in ("good", "bad")]
    out = {"marked": len(rows), "good": sum(g for _, g in rows), "by_score": {}}
    for sc in SCORES:
        res = {}
        for t in THRESHOLDS:
            acc = [(r[sc] >= t, g) for r, g in rows]
            tp = sum(a and g for a, g in acc)
            fp = sum(a and not g for a, g in acc)
            fn = sum(g and not a for a, g in acc)
            res[str(t)] = {"accepted": tp + fp, "precision": tp / (tp + fp) if tp + fp else None,
                           "recall": tp / (tp + fn) if tp + fn else None,
                           "good_rejected": fn}
        out["by_score"][sc] = res
        out["by_score"][sc]["auc"] = auc([r[sc] for r, g in rows if g], [r[sc] for r, g in rows if not g])
    print(json.dumps(out, indent=1))


# --- pre-registered F2 + words judgement (PROTOCOL.md) -------------------------

JUDGED_RULES = {"A1": ("small_min", 0.97, True), "A2": ("base_mean", 0.90, True), "none": (None, None, False)}


def run_judge(args):
    """PROTOCOL.md: each stacked word config, under each word gate, against F2 on dev."""
    from . import param_sweeps as P
    from .testset_v3 import gates, over_edit_gate, paired, summarize
    state = Path(args.state)
    scorer = _ReadOnly(Scorer(state / "nli-memo.jsonl"))
    names = P.design_names(args.design)
    base, stacked = names[0], names[1:]
    plan, passages = load_source_panel("dev")
    outputs = state_outputs(STATE_ROOT / SOURCES["dev"][0])
    roots = metric_roots("dev")
    ids = sorted(passages)
    pending = {n: len(set(ids) - set(outputs.get(n, {}))) for n in names}
    rows = {n: passage_rows("dev", plan, passages, outputs, n, sorted(set(ids) & set(outputs.get(n, {}))), roots)
            for n in names + ["rule_based"]}
    unscored = {n: len(set(ids) - set(r)) for n, r in rows.items()}
    out = {"design": str(args.design), "n": len(ids), "pending_outputs": pending, "unscored": unscored,
           "results": {}}
    if any(pending.values()) or any(unscored.values()):
        out["complete"] = False
        Path(args.out).write_text(json.dumps(out, indent=1))
        print("INCOMPLETE", pending, unscored)
        return
    out["complete"] = True
    rb_ent = summarize([rows["rule_based"][p] for p in ids])["entities"]
    base_rows = [rows[base][p] for p in ids]
    for name in stacked:
        if twin_name(name) != base:
            raise SystemExit(f"{name} is not {base} plus words")
        swaps = {}
        for pid in ids:
            words = [e for e in outputs[name][pid].get("edits", []) if e.get("stage") == "word"]
            if not words:
                continue
            before, after = outputs[base][pid]["output"], outputs[name][pid]["output"]
            span = single_swap(before, after) if len(words) == 1 else None
            if span is None:
                raise SystemExit(f"{pid}: not a single swap over {base}")
            (a0, _), _ = span
            s0, s1 = _sentence_at(before, a0)
            swaps[pid] = {"before": norm(before[s0:s1]), "after": norm(after[s0:s1 + len(after) - len(before)])}
        scored = dict(zip(swaps, pair_scores(scorer, list(swaps.values()))))
        for rule, (sc, t, guard) in JUDGED_RULES.items():
            chosen, kept = [], 0
            for pid in ids:
                keep = pid in swaps and (sc is None or swap_passes(swaps[pid], scored[pid], sc, t, guard))
                chosen.append(rows[name][pid] if keep else rows[base][pid])
                kept += keep
            summ = summarize(chosen)
            controls = [r for r in chosen if r["kind"] == "simple_control"]
            g = gates(summ, rb_ent)
            over = over_edit_gate(summarize(controls))
            d = {m: paired(chosen, base_rows, m) for m in ("fk_grade", "kea_S", "nli_mean")}
            adopt = {"a_gates": all(g.values()), "b_dfk_le_-0.10": d["fk_grade"]["mean"] <= -0.10,
                     "c_dfk_ci_below_0": d["fk_grade"]["ci95"][1] < 0,
                     "d_dnli_ge_-0.010": d["nli_mean"]["mean"] >= -0.010, "e_over_edit": all(over.values())}
            out["results"][f"{name} | {rule}"] = {
                "swaps_proposed": len(swaps), "swaps_kept": kept,
                "fk": summ["fk_grade"], "kea_S": summ["kea_S"], "nli_mean": summ["nli_mean"],
                "nli_below_080": summ["nli_below_080"], "entities": summ["entities"],
                "numbers_failed": summ["numbers_failed"], "negations_failed": summ["negations_failed"],
                "gates": g, "over_edit": over,
                "vs_F2": {m: {k: v[k] for k in ("mean", "ci95")} for m, v in d.items()},
                "adoption": adopt, "beats_F2": all(adopt.values())}
    out["F2"] = {k: summarize(base_rows)[k] for k in ("fk_grade", "kea_S", "nli_mean", "nli_below_080")}
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    for k, v in out["results"].items():
        print(k[-40:], "beats F2" if v["beats_F2"] else "no", v["swaps_kept"], round(v["vs_F2"]["fk_grade"]["mean"], 3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gold")
    g.add_argument("--swords", type=Path, default=ROOT / "models/data/raw/swords_test.json.gz")
    g.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("swaps")
    s.add_argument("--out", type=Path, required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--state", type=Path, required=True)
    sc.add_argument("--items", type=Path, nargs="+", required=True)
    sc.add_argument("--max-seconds", type=float, default=1500)
    sc.add_argument("--threads", type=int, default=2)
    a = sub.add_parser("analyze")
    a.add_argument("--state", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("review")
    r.add_argument("--state", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--n", type=int, default=60)
    m = sub.add_parser("marks")
    m.add_argument("--state", type=Path, required=True)
    m.add_argument("--marks", type=Path, required=True)
    j = sub.add_parser("judge")
    j.add_argument("--state", type=Path, required=True)
    j.add_argument("--design", type=Path, required=True)
    j.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    {"gold": build_gold, "swaps": build_swaps, "score": run_score, "analyze": run_analyze,
     "review": run_review, "marks": run_marks,
     "judge": run_judge}[args.cmd](args)


if __name__ == "__main__":
    main()
