"""Per-swap window veto for the word stage: calibration, lexical metrics and a
dev run on top of F2 (November 18, 2026).

The veto itself is cleartext.swap_veto.WindowSwapVeto. This module binds it
to the evaluation NLI (DeBERTa-v3-base-mnli-fever-anli, float32 weights; see
nli_calibration.Scorer) and to the evaluator's own number, negation and
entity checks, and runs the study in evals/reports/20261118-swap-window-veto/.
No LLM is used anywhere.

Subcommands:
  gold       window NLI for every human-good and human-bad gold substitute
             (SWORDS, TSAR-2022, MultiLS; nli_calibration's gold.jsonl),
             checkpointed in a JSONL memo
  calibrate  precision/recall/balanced accuracy per (window, threshold) and
             the pre-registered choice (PROTOCOL.md)
  freeze     a dev plan that reuses the parameter-sweep dev state's memos
             (T5 beams, word decisions, small NLI, sentence NLI, metrics)
             after checking that every source and artifact they depend on is
             byte-identical, with the committed veto choice
  run        one bounded, checkpointed batch of a design on the dev panel
  judge      the pre-registered comparison of F2 with F2 + words
  page       a local phone-readable page of every swap applied on dev

Raw text (gold sentences, outputs, swap records) stays in the state
directories; only aggregates are written to Git. The locked test split is
never read: panels go through ensemble_study.load_panel and the parameter
sweeps' panel guard, which refuse locked IDs and files.
"""

import argparse
import copy
import difflib
import fcntl
import hashlib
import html
import inspect
import json
import socket
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cleartext.swap_veto import WindowSwapVeto, changed_region, window_pair

from . import ensemble_study as S
from . import nli_calibration as C
from . import param_sweeps as P
from .checkpoints import digest, write_json

ROOT = S.ROOT
REPORT = ROOT / "evals/reports/20261118-swap-window-veto"
CHOICE = REPORT / "veto-choice.json"
STATE_ROOT = Path.home() / ".local/state/cleartext"
CAL_STATE = STATE_ROOT / "nli-calibration-20261117"
SWEEP_DEV = STATE_ROOT / "param-sweeps-20261116/dev"
VERSION = 1

# Calibration grid (PROTOCOL.md). None = the whole sentence (the November 17 unit).
WINDOWS = (2, 3, 4, 6, 8, 10, None)
THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.50, 0.9951, 0.005), 3))
GOLD_KINDS = {"pos": True, "neg": False, "easy_neg": False}  # human-good / human-bad
MIN_PRECISION = 0.90
ZIPF_RARE = 3.5  # readability.technical_term_count's threshold
RESAMPLES, SEED = 2000, 20261118


# --- NLI ----------------------------------------------------------------------

class MemoScorer(C.Scorer):
    """nli_calibration.Scorer (evaluation NLI in float32, append-only JSONL
    memo) that also reads older memo files, read-only."""

    def __init__(self, memo, threads=1, readonly=()):
        super().__init__(memo, threads)
        for path in readonly:
            for row in C.read_jsonl(path) if Path(path).exists() else []:
                self.memo.setdefault(row["k"], row["v"])

    def entail(self, premise, hypothesis):
        return self.score("base", [(premise, hypothesis)])[0]


def sentence_checks(sentence, edited):
    """The evaluator's number, negation and entity checks (all must be 1.0)."""
    return C.checks_pass({"before": sentence, "after": edited})


def evaluation_veto(scorer, threshold, window, log=None):
    return WindowSwapVeto(scorer.entail, threshold, window, checks=sentence_checks, log=log)


# --- gold calibration -----------------------------------------------------------

def gold_rows(path):
    return [r for r in C.read_jsonl(path) if r["kind"] in GOLD_KINDS]


def gold_pairs(rows, windows=WINDOWS):
    pairs = []
    for r in rows:
        for w in windows:
            b, a = window_pair(r["before"], r["after"], w)
            pairs += [(b, a), (a, b)]
    return list(dict.fromkeys(pairs))


def run_gold(args):
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    scorer = MemoScorer(state / "nli-memo.jsonl", args.threads, readonly=[CAL_STATE / "nli-memo.jsonl"])
    pairs = gold_pairs(gold_rows(args.gold))
    todo = [p for p in pairs if scorer.key("base", *p) not in scorer.memo]
    print(f"{len(pairs)} unique window pairs, {len(todo)} to score", flush=True)
    deadline = time.monotonic() + args.max_seconds
    done = 0
    for i in range(0, len(todo), 256):
        if time.monotonic() > deadline:
            break
        scorer.score("base", todo[i:i + 256])
        scorer.flush()
        done = min(i + 256, len(todo))
        print(f"{time.strftime('%H:%M:%S')} scored {done}/{len(todo)}", flush=True)
    print("COMPLETE" if done >= len(todo) else "PARTIAL", flush=True)


def confusion(good, accept):
    good, accept = np.asarray(good, bool), np.asarray(accept, bool)
    tp, fn = int((good & accept).sum()), int((good & ~accept).sum())
    fp, tn = int((~good & accept).sum()), int((~good & ~accept).sum())
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "recall": recall,
            "precision": tp / (tp + fp) if tp + fp else None, "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None}


def choose(grid, min_precision=MIN_PRECISION):
    """PROTOCOL.md: the highest pooled balanced accuracy among settings with
    pooled precision >= min_precision; if none reaches it, the highest
    balanced accuracy overall. Ties: higher precision, smaller window
    (sentence last), higher threshold."""
    def key(g):
        p = g["pooled"]
        return (p["balanced_accuracy"], p["precision"] or 0.0, -(g["window"] if g["window"] is not None else 99),
                g["threshold"])
    eligible = [g for g in grid if (g["pooled"]["precision"] or 0.0) >= min_precision]
    pool = eligible or grid
    best = max(pool, key=key)
    return {"window": best["window"], "threshold": best["threshold"], "precision_constraint_met": bool(eligible),
            "pooled": best["pooled"]}


def cluster_ci(rows, good, accept, resamples=RESAMPLES, seed=SEED):
    """Percentile 95% CIs of precision, recall and balanced accuracy, resampling
    targets (a target contributes several substitutes)."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["group"]].append(i)
    keys = list(groups)
    good, accept = np.asarray(good, bool), np.asarray(accept, bool)
    rng = np.random.default_rng(seed)
    stats = defaultdict(list)
    for _ in range(resamples):
        idx = np.concatenate([groups[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        c = confusion(good[idx], accept[idx])
        for m in ("precision", "recall", "balanced_accuracy"):
            if c[m] is not None:
                stats[m].append(c[m])
    return {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in stats.items()}


def run_calibrate(args):
    state = Path(args.state)
    scorer = MemoScorer(state / "nli-memo.jsonl", readonly=[CAL_STATE / "nli-memo.jsonl"])
    rows = gold_rows(args.gold)
    good = np.array([GOLD_KINDS[r["kind"]] for r in rows])
    checks = np.array([sentence_checks(r["before"], r["after"]) for r in rows])
    datasets = sorted({r["dataset"] for r in rows})
    grid, means, missing = [], {}, 0
    for w in WINDOWS:
        vals = []
        for r in rows:
            b, a = window_pair(r["before"], r["after"], w)
            f, k = scorer.memo.get(scorer.key("base", b, a)), scorer.memo.get(scorer.key("base", a, b))
            missing += (f is None) + (k is None)
            vals.append(np.nan if f is None or k is None else (f + k) / 2)
        means[w] = np.array(vals)
    if missing:
        raise SystemExit(f"{missing} window scores missing; run `gold` to completion first")
    for w in WINDOWS:
        for t in THRESHOLDS:
            accept = (means[w] >= t) & checks
            entry = {"window": w, "threshold": t, "pooled": confusion(good, accept), "by_dataset": {}}
            for ds in datasets:
                sel = np.array([r["dataset"] == ds for r in rows])
                entry["by_dataset"][ds] = confusion(good[sel], accept[sel])
            grid.append(entry)
    choice = choose(grid)
    w, t = choice["window"], choice["threshold"]
    accept = (means[w] >= t) & checks
    choice["pooled_ci95"] = cluster_ci(rows, good, accept)
    choice["by_dataset"] = next(g for g in grid if g["window"] == w and g["threshold"] == t)["by_dataset"]
    swords = np.array([r["dataset"] == "swords_test" for r in rows])
    choice["swords_ci95"] = cluster_ci([r for r, s in zip(rows, swords) if s], good[swords], accept[swords])
    best_by_window = {}
    for win in WINDOWS:
        sub = [g for g in grid if g["window"] == win]
        top = max(sub, key=lambda g: g["pooled"]["balanced_accuracy"])
        prec = [g for g in sub if (g["pooled"]["precision"] or 0) >= MIN_PRECISION]
        best_by_window[str(win)] = {"best_balanced": {"threshold": top["threshold"], **top["pooled"]},
                                    "best_balanced_with_precision_ge_0.9":
                                        ({"threshold": max(prec, key=lambda g: g["pooled"]["balanced_accuracy"])["threshold"],
                                          **max(prec, key=lambda g: g["pooled"]["balanced_accuracy"])["pooled"]}
                                         if prec else None),
                                    "at_0.95": next(g["pooled"] for g in sub if g["threshold"] == 0.95),
                                    "at_0.97": next(g["pooled"] for g in sub if g["threshold"] == 0.97)}
    out = {"counts": Counter(f"{r['dataset']}/{r['kind']}" for r in rows), "checks_fail": int((~checks).sum()),
           "max_precision": max(g["pooled"]["precision"] or 0 for g in grid),
           "best_by_window": best_by_window, "choice": choice, "grid": grid,
           "rule": "max pooled balanced accuracy s.t. pooled precision >= 0.90, else max balanced accuracy"}
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    write_json(Path(args.choice), {"window": w, "threshold": t,
                                   "precision_constraint_met": choice["precision_constraint_met"],
                                   "source": "calibrate on gold.jsonl (SWORDS, TSAR-2022, MultiLS)",
                                   "gold_sha256": S.sha(args.gold)})
    print(json.dumps({k: choice[k] for k in ("window", "threshold", "precision_constraint_met", "pooled")},
                     indent=1, default=float))


# --- lexical simplicity metrics -------------------------------------------------

_GBM = None


def _gbm():
    global _GBM
    if _GBM is None:
        from . import word_models
        _GBM = word_models
    return _GBM


def content_words(text):
    """[(word, words of its sentence)] for alphabetic non-stopwords, casefolded."""
    from spacy.lang.en.stop_words import STOP_WORDS
    from .evaluate_base import words
    from .preservation import split_sentences
    out = []
    for a, b in split_sentences(text):
        ws = words(text[a:b])
        out += [(w, ws) for w in ws if w.isalpha() and w not in STOP_WORDS]
    return out


def zipf(word):
    from wordfreq import zipf_frequency
    return zipf_frequency(word, "en")


def lexical_row(source, output):
    """Per-passage lexical simplicity (no text): mean Kea-Roy GBM word
    complexity over content words, share of content words with Zipf < 3.5,
    and the mean Zipf of content words inserted minus deleted vs the source."""
    wm = _gbm()
    model, _ = wm._ensure()
    cw = content_words(output)
    if cw:
        feats = [wm.featurize6(w, ws) for w, ws in cw]
        complexity = float(np.clip(model.predict(feats), 0.0, 1.0).mean())
        rare = float(np.mean([zipf(w) < ZIPF_RARE for w, _ in cw]))
    else:
        complexity = rare = None
    a, b = [w for w, _ in content_words(source)], [w for w, _ in cw]
    deleted, inserted = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            deleted += a[i1:i2]
            inserted += b[j1:j2]
    gain = (float(np.mean([zipf(w) for w in inserted]) - np.mean([zipf(w) for w in deleted]))
            if inserted and deleted else None)
    return {"word_complexity": complexity, "rare_share": rare, "changed_zipf_gain": gain}


# --- frozen plan reusing the sweep's dev state ----------------------------------

def committed_choice(path=CHOICE):
    """The committed, unmodified veto choice (PROTOCOL.md)."""
    rel = str(Path(path).resolve().relative_to(ROOT))
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    if git("ls-files", "--error-unmatch", rel).returncode or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        raise SystemExit(f"{rel} must be committed and unmodified before a dev run")
    value = json.loads(Path(path).read_text())
    return {"window": value["window"], "threshold": value["threshold"], "path": rel, "sha256": S.sha(path),
            "commit": git("log", "-1", "--format=%H", "--", rel).stdout.strip()}


def rerooted(path, old_root):
    p = Path(path)
    return str(ROOT / p.relative_to(old_root)) if p.is_relative_to(old_root) else str(p)


def run_freeze(args):
    state, base = args.state.resolve(), args.base_state.resolve()
    if state == ROOT or ROOT in state.parents:
        raise SystemExit("State must be outside the checkout; raw text stays local.")
    old = json.loads((base / "plan.json").read_text())
    if old.get("unlock_test"):
        raise SystemExit("The base state must be a dev plan")
    old_root = Path(old["generation"]["runs"]["v7"]).parents[2]
    choice = committed_choice()
    panel = old["panels"][args.panel]
    P.guard_panel(panel)
    generation = copy.deepcopy(old["generation"])
    generation["runs"] = {k: rerooted(v, old_root) for k, v in generation["runs"].items()}
    problems = []
    artifacts = {}
    for path, h in old["generation_artifacts_sha256"].items():
        here = rerooted(path, old_root)
        artifacts[here] = S.sha(here) if Path(here).is_file() else None
        if artifacts[here] != h:
            problems.append(f"artifact {path}")
    for path, h in old["evaluation_artifacts_sha256"].items():
        if not Path(path).is_file() or S.sha(path) != h:
            problems.append(f"evaluation artifact {path}")
    sources = {str(p): S.sha(p) for p in S.generation_sources()}
    old_sources = {rerooted(p, old_root): h for p, h in old["generation_sources_sha256"].items()}
    problems += [f"source {p}" for p, h in old_sources.items() if sources.get(p) != h]
    added = sorted(set(sources) - set(old_sources))
    allowed = {str(ROOT / "models/src/cleartext/swap_veto.py")}
    problems += [f"unexpected new source {p}" for p in added if p not in allowed]
    scoring = {str(p): S.sha(p) for p in S.scoring_sources()}
    if sorted(scoring.items()) != sorted((rerooted(p, old_root), h) for p, h in old["scoring_sources_sha256"].items()):
        problems.append("scoring sources")
    if S.runtime() != old["runtime"]:
        problems.append("runtime")
    if S.sha(panel) != old["panel_sha256"][args.panel]:
        problems.append("panel")
    if problems:
        raise SystemExit("Cannot reuse " + str(base) + ": " + "; ".join(problems))
    old_gen, old_score = S.state_signatures(base)
    plan = {
        "created": datetime.now(timezone.utc).isoformat(),
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                               check=True).stdout.strip(),
        "panels": {args.panel: panel}, "panel_sha256": {args.panel: old["panel_sha256"][args.panel]},
        "generation": generation, "evaluation": old["evaluation"],
        "generation_artifacts_sha256": artifacts,
        "evaluation_artifacts_sha256": old["evaluation_artifacts_sha256"],
        "generation_sources_sha256": sources, "scoring_sources_sha256": scoring, "runtime": S.runtime(),
        "veto": choice,
    }
    if P.signatures(plan)[1] != old_score:
        raise SystemExit("Scoring signature differs from the base state's; metrics cannot be reused")
    # StudyGenerator reads plan["reuse"] for T5, grammar and decision memos;
    # VetoGenerator adds the beam and small-NLI memos; metrics fall back too.
    plan["reuse"] = {"state": str(base), "generation_signature": old_gen, "scoring_signature": old_score,
                     "memos": ["t5", "grammar", "decisions", "t5-beams*", "small-nli", "nli-pairs"],
                     "metrics": True, "sources_added": added, "old_head": old["head"]}
    plan["reuse_metrics"] = {"state": str(base), "scoring_signature": old_score}
    if str(Path(C.BASE_NLI).resolve()) != str(Path(plan["evaluation"]["nli"]).resolve()):
        raise SystemExit("The veto's NLI is not the evaluation NLI")
    state.mkdir(parents=True, exist_ok=True)
    path = state / "plan.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a frozen plan is never overwritten")
    write_json(path, plan)
    print(f"Froze {path}: veto window {choice['window']} threshold {choice['threshold']}; "
          f"reusing {base} (generation {old_gen[:16]}, scoring {old_score[:16]})")


def signatures(plan):
    gen, score = P.signatures(plan)
    # Everything outside the frozen sources that can change an output.
    code = [VetoGenerator, MemoScorer, C.Scorer, C.checks_pass, evaluation_veto, sentence_checks]
    harness = hashlib.sha256("".join(inspect.getsource(o) for o in code).encode()).hexdigest()
    return digest({"sweep": gen, "harness": harness, "veto": plan["veto"], "version": VERSION}), score


# --- generation -------------------------------------------------------------------

def design_names(path):
    """IDs of a design: sweep configs, with "@window" for the window veto."""
    rows = json.loads(Path(path).read_text())["configs"]
    names = []
    for c in rows:
        if isinstance(c, str):
            names.append(c)
            continue
        name = P.make_id(c["base"], c.get("params"))
        if c.get("veto"):
            if c["veto"] != "window" or c["base"]["words"] not in ("v7", "contextual"):
                raise ValueError("veto 'window' needs words=v7 or words=contextual")
            name += "@window"
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate configs in design")
    return names


def design_labels(path):
    rows = json.loads(Path(path).read_text())["configs"]
    return {n: (c.get("label") if isinstance(c, dict) else c) for n, c in zip(design_names(path), rows)}


class VetoGenerator(P.SweepGenerator):
    """SweepGenerator with the window veto ("<config>@window") and read-only
    fallback to the base state's beam and small-NLI memos."""

    def __init__(self, plan, state, signature, threads=1):
        super().__init__(plan, state, signature, threads)
        self._window, self._log, self._nli = False, None, None

    def beams(self, width):
        if width not in self._beams:
            memo = S.DiskMemo(self.state, f"t5-beams{width}", self.signature, self.fallback)
            loaded, study = [], self

            class Beams:
                def candidates(_, sentence):
                    if sentence not in memo:
                        if not loaded:
                            from cleartext.structure.seq2seq import SentenceGenerator
                            loaded.append(SentenceGenerator(study.plan["generation"]["t5"], device="cpu",
                                                            num_beams=width, num_return_sequences=width))
                            study._limit_threads()
                        result = loaded[0].predict(sentence)
                        memo[sentence] = result.get("candidates") or [
                            {"raw": result["raw"],
                             "terminated": result["raw_checks"]["checks"].get("generation_terminated", True)}]
                    return memo[sentence]

            self._beams[width] = Beams()
        return self._beams[width]

    def small_entailment(self, premise, hypothesis):
        memo = S.DiskMemo(self.state, "small-nli", self.signature, self.fallback)
        key = [premise, hypothesis]
        if key not in memo:
            if self._small_nli is None:
                from cleartext.teammate_stages import EntailmentVeto
                self._small_nli = EntailmentVeto(model=self.plan["generation"]["teammate_models"]["nli"])
                self._limit_threads()
            memo[key] = self._small_nli.score(premise, hypothesis)
        return memo[key]

    def nli(self):
        if self._nli is None:
            self._nli = MemoScorer(self.state / "window-nli.jsonl", self.threads)
        return self._nli

    def tuned_veto(self, level, params):
        if self._window:
            veto = self.plan["veto"]
            return evaluation_veto(self.nli(), veto["threshold"], veto["window"], log=self._log)
        return super().tuned_veto(level, params)

    def predict(self, name, text):
        base, _, veto = name.partition("@")
        if not veto:
            return super().predict(name, text)
        if veto != "window":
            raise ValueError(f"Unknown veto {veto}")
        self._window, self._log = True, []
        try:
            result = super().predict(base, text)
        finally:
            self._window = False
        result["veto_log"], self._log = self._log, None
        self.nli().flush()
        return result


def make_scorer(plan, state, score_sig, threads):
    scorer = S.StudyScorer(plan, state, score_sig, threads)
    reuse = plan["reuse"]
    scorer.memo = S.DiskMemo(state, "nli-pairs", score_sig, (reuse["state"], reuse["scoring_signature"]))
    return scorer


def run(args):
    plan = json.loads(args.plan.read_text())
    P.preflight(plan, check_artifacts=True)
    gen_sig, score_sig = signatures(plan)
    names = [*S.REFERENCES, *[n for n in design_names(args.design) if n not in S.REFERENCES]]
    passages = S.load_panel(plan, args.panel)
    state = args.state
    store = P.Store(state, gen_sig, score_sig, P.metric_fallbacks(plan))
    import torch
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        pass
    started, deadline = time.monotonic(), time.monotonic() + args.max_seconds
    (state / "locks").mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with (state / "locks" / f"{args.panel}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generator = scorer = None

        def log(message):
            print(f"{time.strftime('%H:%M:%S')} +{time.monotonic() - started:.0f}s {message}", flush=True)

        done = True
        for passage in passages:
            missing = [n for n in names if store.output(passage, n) is None]
            if not missing:
                continue
            generator = generator or VetoGenerator(plan, state, gen_sig, args.threads)
            generator.begin(passage)
            try:
                for name in missing:
                    if time.monotonic() >= deadline:
                        done = False
                        break
                    t0 = time.monotonic()
                    result = generator.predict(name, passage["original"])
                    result["seconds"] = time.monotonic() - t0
                    store.save_output(passage, name, result)
                    counts["outputs"] += 1
                    if result["seconds"] > 5:
                        log(f"output {passage['id']} {name[:40]} {result['seconds']:.1f}s")
            finally:
                generator.end(passage)
            if not done:
                break
        if done and args.score:
            # Score config by config, in design order, so the primary arms finish first.
            for name in names:
                for passage in passages:
                    out = store.output(passage, name)["output"]
                    if store.metrics(passage["original"], out) is not None:
                        continue
                    if time.monotonic() >= deadline:
                        done = False
                        break
                    scorer = scorer or make_scorer(plan, state, score_sig, args.threads)
                    store.save_metrics(passage["original"], out, scorer(passage["original"], out))
                    counts["scored"] += 1
                if not done:
                    break
            log(f"scored {counts['scored']} pairs")
        complete = all(store.output(p, n) is not None and (not args.score or store.metrics(
            p["original"], store.output(p, n)["output"]) is not None) for p in passages for n in names)
    P.preflight(plan, check_artifacts=True)
    receipt = {"host": socket.gethostname(), "panel": args.panel, "design": str(args.design),
               "design_sha256": S.sha(args.design), "configs": len(names),
               "seconds": time.monotonic() - started, "generation_signature": gen_sig,
               "scoring_signature": score_sig, "complete": complete, **counts, "frozen_inputs_rechecked": True}
    write_json(state / "batches" / f"{time.time_ns()}.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    print("COMPLETE" if complete else "PARTIAL", flush=True)


# --- judgement --------------------------------------------------------------------

def load_state(state, plan, panel):
    """({id: passage}, {config: {id: output record}}, metric Store)."""
    gen_sig, score_sig = signatures(plan)
    store = P.Store(state, gen_sig, score_sig, P.metric_fallbacks(plan))
    passages = {p["id"]: p for p in S.load_panel(plan, panel)}
    outputs = defaultdict(dict)
    for path in (Path(state) / "outputs" / gen_sig[:16]).glob("*.json"):
        d = json.loads(path.read_text())
        if d["identity"]["signature"] != gen_sig:
            raise ValueError(f"{path}: output from another generation signature")
        outputs[d["identity"]["config"]][d["identity"]["id"]] = d
    return passages, outputs, store


def swap_records(passage, rec):
    """Every veto decision of one @window output, with whether it was applied."""
    import re
    bare = lambda text: re.sub(r"[^\w\s]", "", text).strip().lower()
    words = [e for e in rec.get("edits", []) if e.get("stage") == "word"]
    rows = []
    for d in rec.get("veto_log", []):
        # The logged run is whole whitespace words, so it may carry punctuation or an article.
        applied = d["passes"] and any(bare(e["source"]) in bare(d["word"]).split()
                                      and bare(e["replacement"]) in bare(d["replacement"]).split() for e in words)
        rows.append({**d, "applied": bool(applied)})
    return rows


def bootstrap_mean(values, resamples=10000, seed=SEED):
    values = np.asarray([v for v in values if v is not None], float)
    if not len(values):
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(np.percentile(means, 2.5)),
                                                   float(np.percentile(means, 97.5))], "n": int(len(values))}


def judged(arm, base_rows, rb_ent):
    """PROTOCOL.md success rule for one F2 + words arm against F2."""
    from .testset_v3 import gates, over_edit_gate, paired, summarize
    summ = summarize(arm)
    g = gates(summ, rb_ent)
    over = over_edit_gate(summarize([r for r in arm if r["kind"] == "simple_control"]))
    d = {m: paired(arm, base_rows, m) for m in
         ("fk_grade", "kea_S", "nli_mean", "word_complexity", "rare_share")}
    rule = {"a_dnli_ci_low_ge_-0.005": d["nli_mean"]["ci95"][0] >= -0.005,
            "b_gates_1_4": all(g.values()), "c_over_edit": all(over.values()),
            "d_word_complexity_ci_below_0": d["word_complexity"]["ci95"][1] < 0,
            "e_dfk_le_0": d["fk_grade"]["mean"] <= 0}
    lex = {m: float(np.mean([r[m] for r in arm if r[m] is not None])) for m in ("word_complexity", "rare_share")}
    return {"fk": summ["fk_grade"], "kea_S": summ["kea_S"], "nli_mean": summ["nli_mean"],
            "nli_below_080": summ["nli_below_080"], "entities": summ["entities"],
            "numbers_failed": summ["numbers_failed"], "negations_failed": summ["negations_failed"],
            **lex, "gates": g, "over_edit": over,
            "vs_F2": {m: {k: v[k] for k in ("mean", "ci95", "passages_nonzero")} for m, v in d.items()},
            "rule": rule, "beats_F2": all(rule.values())}


def run_judge(args):
    from .testset_v3 import passage_row, summarize
    plan = json.loads((args.state / "plan.json").read_text())
    passages, outputs, store = load_state(args.state, plan, args.panel)
    labels = design_labels(args.design)
    names = design_names(args.design)
    base = names[0]
    ids = sorted(passages)
    rows, pending = {}, {}
    for name in names + ["rule_based", "original"]:
        rows[name] = {}
        for pid in ids:
            rec = outputs.get(name, {}).get(pid)
            m = rec and store.metrics(passages[pid]["original"], rec["output"])
            if m:
                r = passage_row(passages[pid], rec["output"], m)
                r.update(lexical_row(passages[pid]["original"], rec["output"]))
                rows[name][pid] = r
        pending[name] = len(ids) - len(rows[name])
    rb_ent = summarize(list(rows["rule_based"].values()))["entities"] if not pending["rule_based"] else None
    # Cache reuse check: configs the sweep state also generated must come out identical.
    sweep = C.state_outputs(plan["reuse"]["state"])
    same = {}
    for name in names + ["rule_based", "original"]:
        if name in sweep:
            both = set(outputs.get(name, {})) & set(sweep[name])
            diff = sorted(p for p in both if outputs[name][p]["output"] != sweep[name][p]["output"])
            same[labels.get(name, name)] = {"compared": len(both), "differ": len(diff)}
            if diff:
                raise SystemExit(f"{name}: {len(diff)} outputs differ from the reused sweep state (e.g. {diff[0]})")
    out = {"design": str(args.design.relative_to(ROOT)) if args.design.is_relative_to(ROOT) else str(args.design),
           "veto": {k: plan["veto"][k] for k in ("window", "threshold", "commit")},
           "n": len(ids), "pending": pending, "same_as_sweep_state": same, "systems": {}, "arms": {}}
    for name in names + ["original", "rule_based"]:
        if not pending[name]:
            s = summarize([rows[name][p] for p in ids])
            out["systems"][labels.get(name, name)] = {
                **{k: s[k] for k in ("fk_grade", "kea_S", "nli_mean", "nli_below_080", "entities",
                                     "numbers_failed", "negations_failed", "changed")},
                **{m: float(np.mean([rows[name][p][m] for p in ids if rows[name][p][m] is not None]))
                   for m in ("word_complexity", "rare_share")},
                "changed_zipf_gain": bootstrap_mean([rows[name][p]["changed_zipf_gain"] for p in ids])}
    swaps_local = []
    for name in names[1:]:
        done = sorted(set(rows[name]) & set(rows[base]))
        if not done or rb_ent is None:
            continue
        arm = [rows[name][p] for p in done]
        base_rows = [rows[base][p] for p in done]
        for p in done:
            if not any(e.get("stage") == "word" for e in outputs[name][p].get("edits", [])) and \
                    outputs[name][p]["output"] != outputs[base][p]["output"]:
                raise SystemExit(f"{p}: no word edit, but {name} differs from {base}")
        res = judged(arm, base_rows, rb_ent)
        res["passages"] = len(done)
        res["complete"] = len(done) == len(ids)
        words = [(p, e) for p in done for e in outputs[name][p].get("edits", []) if e.get("stage") == "word"]
        res["swaps_applied"] = len(words)
        res["swap_zipf"] = {"replaced": bootstrap_mean([C_zipf(e["source"]) for _, e in words]),
                            "replacement": bootstrap_mean([C_zipf(e["replacement"]) for _, e in words]),
                            "gain": bootstrap_mean([C_zipf(e["replacement"]) - C_zipf(e["source"])
                                                    for _, e in words])}
        if name.endswith("@window"):
            recs = [(p, r) for p in done for r in swap_records(passages[p], outputs[name][p])]
            res["veto"] = {"decisions": len(recs), "passed": sum(r["passes"] for _, r in recs),
                           "vetoed_nli": sum(not r["passes"] and r["checks"] for _, r in recs),
                           "vetoed_checks": sum(not r["checks"] for _, r in recs),
                           "applied": sum(r["applied"] for _, r in recs),
                           "passages_with_a_veto": len({p for p, r in recs if not r["passes"]}),
                           "refilled": sum(any(not r["passes"] for r in swap_records(passages[p], outputs[name][p]))
                                           and any(e.get("stage") == "word" for e in outputs[name][p]["edits"])
                                           for p in done)}
            swaps_local += [{"config": labels.get(name, name), "passage": p, "domain": passages[p]["domain"],
                             "kind": passages[p]["kind"], "passage_nli": rows[name][p]["nli_mean"],
                             "f2_passage_nli": rows[base][p]["nli_mean"], **r} for p, r in recs]
        out["arms"][labels.get(name, name)] = res
    if swaps_local:
        sentence_scores(swaps_local, args.state)
        C.write_jsonl(args.state / "swaps.jsonl", swaps_local)
        applied = [r for r in swaps_local if r["applied"]]
        out["applied_swaps_sentence_vs_window"] = {
            "n": len(applied),
            "sentence_mean_below_0.95": sum((r["sentence_mean"] or 0) < 0.95 for r in applied),
            "window_mean": bootstrap_mean([r["mean"] for r in applied]),
            "sentence_mean": bootstrap_mean([r["sentence_mean"] for r in applied])}
        out["applied_swaps_by_sentence_nli"] = diagnostics(applied)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    for k, v in out["arms"].items():
        print(f"{k}: {v['passages']} passages, beats F2: {v['beats_F2']} {v['rule']}")


def diagnostics(applied):
    """Passage-NLI cost of the applied swaps by their whole-sentence NLI (post
    hoc, aggregate): where do the passages that fall below 0.80 come from?"""
    out = {}
    for config in sorted({r["config"] for r in applied}):
        rows = [r for r in applied if r["config"] == config]
        bands = {}
        for lo, hi, label in C.BANDS:
            sel = [r for r in rows if r["sentence_mean"] is not None and lo <= r["sentence_mean"] < hi]
            d = [r["passage_nli"] - r["f2_passage_nli"] for r in sel
                 if r["passage_nli"] is not None and r["f2_passage_nli"] is not None]
            bands[label] = {"swaps": len(sel), "mean_d_passage_nli": float(np.mean(d)) if d else None,
                            "newly_below_080": sum(r["passage_nli"] < 0.80 <= r["f2_passage_nli"] for r in sel
                                                   if r["passage_nli"] is not None and r["f2_passage_nli"] is not None)}
        out[config] = bands
    return out


def C_zipf(word):
    return zipf(word.lower())


def sentence_scores(records, state):
    """Whole-sentence evaluation NLI of each logged swap (for the page and the
    sentence-vs-window comparison), from the shared memo when present."""
    scorer = MemoScorer(state / "window-nli.jsonl", readonly=[CAL_STATE / "nli-memo.jsonl"])
    for r in records:
        f, b = scorer.entail(r["sentence"], r["edited"]), scorer.entail(r["edited"], r["sentence"])
        r["sentence_mean"] = None if f is None or b is None else (f + b) / 2
    scorer.flush()


# --- applied-swaps page (local only) --------------------------------------------

def _marked(text, other):
    prefix, mid, _, suffix = changed_region(text, other)
    return f"{html.escape(prefix)}<mark>{html.escape(mid)}</mark>{html.escape(suffix)}"


def run_page(args):
    rows = C.read_jsonl(args.state / "swaps.jsonl")
    applied = sorted((r for r in rows if r["applied"]), key=lambda r: (r["config"], r["mean"]))
    vetoed = sorted((r for r in rows if not r["passes"]), key=lambda r: (r["config"], -(r["mean"] or 0)))

    def card(r):
        sent = "–" if r.get("sentence_mean") is None else f"{r['sentence_mean']:.2f}"
        why = "" if r["passes"] else (" · vetoed: checks" if not r["checks"] else " · vetoed: window NLI")
        return (f"<article><header><b>{html.escape(r['word'])} → {html.escape(r['replacement'])}</b>"
                f"<span>{html.escape(r['config'])} · {html.escape(r['passage'])}</span></header>"
                f"<p class=b>{_marked(r['sentence'], r['edited'])}</p>"
                f"<p class=a>{_marked(r['edited'], r['sentence'])}</p>"
                f"<p class=s>window NLI <b>{(r['mean'] or 0):.3f}</b> (→ {(r['forward'] or 0):.2f}, "
                f"← {(r['backward'] or 0):.2f}) · whole sentence {sent}{why}</p>"
                f"<p class=w>{html.escape(r['window_before'])}</p></article>")

    body = "".join(card(r) for r in applied)
    rest = "".join(card(r) for r in vetoed)
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Applied word swaps</title>
<style>
:root{{--bg:#fff;--fg:#111;--muted:#666;--line:#e4e4e4;--mark:#fde68a;--card:#fafafa}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--muted:#999;--line:#2a2a2a;--mark:#7c5e10;--card:#181818}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,sans-serif}}
main{{max-width:720px;margin:0 auto;padding:16px}}
h1{{font-size:20px;margin:8px 0}} h2{{font-size:17px;margin:24px 0 8px}}
article{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin:10px 0}}
header{{display:flex;flex-wrap:wrap;justify-content:space-between;gap:4px}}
header span,.s,.w{{color:var(--muted);font-size:13px}} .w{{font-style:italic}}
p{{margin:6px 0;overflow-wrap:anywhere}} .b{{opacity:.75}} mark{{background:var(--mark);color:inherit;border-radius:3px}}
summary{{cursor:pointer;font-weight:600;margin:16px 0}}
</style></head><body><main>
<h1>Applied word swaps on dev ({len(applied)})</h1>
<p class=s>Window veto: ±{args.window} words, mean of both NLI directions ≥ {args.threshold}, plus number, negation
and entity checks. First line: before; second: after. Sorted by window NLI, lowest first. Local only.</p>
{body}
<details><summary>Vetoed swaps ({len(vetoed)})</summary>{rest}</details>
</main></body></html>"""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(page)
    print(f"{args.out}: {len(applied)} applied, {len(vetoed)} vetoed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gold")
    g.add_argument("--state", type=Path, required=True)
    g.add_argument("--gold", type=Path, default=CAL_STATE / "gold.jsonl")
    g.add_argument("--max-seconds", type=float, default=1500)
    g.add_argument("--threads", type=int, default=4)
    c = sub.add_parser("calibrate")
    c.add_argument("--state", type=Path, required=True)
    c.add_argument("--gold", type=Path, default=CAL_STATE / "gold.jsonl")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--choice", type=Path, required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--state", type=Path, required=True)
    f.add_argument("--base-state", type=Path, default=SWEEP_DEV)
    f.add_argument("--panel", default="dev")
    r = sub.add_parser("run")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--state", type=Path, required=True)
    r.add_argument("--design", type=Path, required=True)
    r.add_argument("--panel", default="dev")
    r.add_argument("--max-seconds", type=float, default=1500)
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--no-score", dest="score", action="store_false")
    j = sub.add_parser("judge")
    j.add_argument("--state", type=Path, required=True)
    j.add_argument("--design", type=Path, required=True)
    j.add_argument("--panel", default="dev")
    j.add_argument("--out", type=Path, required=True)
    pg = sub.add_parser("page")
    pg.add_argument("--state", type=Path, required=True)
    pg.add_argument("--out", type=Path, required=True)
    pg.add_argument("--window", required=True)
    pg.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args(argv)
    {"gold": run_gold, "calibrate": run_calibrate, "freeze": run_freeze, "run": run, "judge": run_judge,
     "page": run_page}[args.cmd](args)


if __name__ == "__main__":
    main()
