"""Non-LLM ensemble study: planned designs over pipeline stage choices.

Subcommands:
  freeze   write a frozen plan (paths, hashes, runtime) into an external state dir
  run      one bounded batch: generate missing outputs, then score missing pairs
  report   aggregate complete configs, apply the frozen selection rule, write
           aggregate-only JSON/Markdown (raw text stays in the state dir)

Outputs are cached per (passage, config) under a generation signature, metrics
per exact (source, output) pair under a separate scoring signature, so configs
that produce the same text share scoring. Stage caches (lexical decision rows,
T5 sentences, grammar corrections) are reused across configs and batches.

A factor added later (e.g. passive-to-active) goes in FACTORS with its legacy
level in LEGACY: configs at that level keep their old IDs, so a rerun only adds
cells. `report` accepts several state dirs and refuses to merge conflicting
outputs for the same cell.
"""

import argparse
import copy
import fcntl
import hashlib
import inspect
import json
import pickle
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .checkpoints import digest, write_json

ROOT = Path(__file__).resolve().parents[1]

# Level order matters only for display. "simpleppdb" phrases are not on main.
FACTORS = {
    "words": ("v7", "contextual", "off"),
    "phrases": ("authored", "off"),
    "structure": ("rules", "t5", "compound_tagger", "off"),
    "negation": ("off", "on"),
    "grammar": ("off", "on"),
    "terms": ("off", "on"),
}
# Factors added after the first run, with the level that reproduces old behaviour.
LEGACY = {}
BASELINE = {"words": "v7", "phrases": "authored", "structure": "rules",
            "negation": "off", "grammar": "off", "terms": "off"}
REFERENCES = ("original", "rule_based")
GENERATION_VERSION = 1
SCORING_VERSION = 1
PACKAGES = ("torch", "transformers", "sentence-transformers", "textstat", "spacy",
            "en-core-web-sm", "numpy", "scikit-learn", "lemminflect", "wordfreq", "joblib")


def normalize(config):
    config = {**BASELINE, **{k: v for k, v in LEGACY.items()}, **config}
    unknown = set(config) - set(FACTORS)
    if unknown:
        raise ValueError(f"Unknown factors: {sorted(unknown)}")
    for name, value in config.items():
        if value not in FACTORS[name]:
            raise ValueError(f"Unknown level {name}={value}")
    return {name: config[name] for name in FACTORS}


def config_id(config):
    """Stable ID; later factors at their legacy level are omitted."""
    if isinstance(config, str):
        return config
    config = normalize(config)
    return "|".join(f"{k}={v}" for k, v in config.items()
                    if not (k in LEGACY and v == LEGACY[k]))


def parse_id(name):
    if name in REFERENCES:
        return name
    return normalize(dict(part.split("=", 1) for part in name.split("|")))


def ofat(base=BASELINE):
    """Baseline plus every single-factor change from it."""
    base = normalize(base)
    rows = [base]
    for name, levels in FACTORS.items():
        rows += [{**base, name: level} for level in levels if level != base[name]]
    return rows


def factorial(levels, base=BASELINE):
    """Full factorial over {factor: [levels]} with other factors at base."""
    rows = [normalize(base)]
    for name, values in levels.items():
        rows = [{**row, name: value} for row in rows for value in values]
    return [normalize(r) for r in rows]


# --- frozen plan --------------------------------------------------------------

def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _files(path):
    path = Path(path)
    return [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())


def _run_dependencies(run):
    run = Path(run)
    found = list(_files(run))
    config = json.loads((run / "config.json").read_text())

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, str) and (ROOT / "models" / value).is_file():
            found.append(ROOT / "models" / value)

    walk(config)
    found += [ROOT / "models/runs/initial-20260924/word_model.pkl",
              ROOT / "models/runs/initial-20260924/context_model.pkl",
              ROOT / "models/runs/context-20260924/phrase_model.pkl",
              ROOT / "models/runs/context-20260924/checker.pkl"]
    return [p for p in found if p.is_file()]


def generation_sources():
    return sorted([*(ROOT / "models/src/cleartext").rglob("*.py"), ROOT / "evals/baselines.py",
                   ROOT / "evals/pipeline_benchmark.py", ROOT / "evals/data/simplification_dict.json"])


def scoring_sources():
    return [ROOT / "evals" / n for n in ("preservation.py", "readability.py", "runner.py")]


def freeze(args):
    state = args.state.resolve()
    if state == ROOT or ROOT in state.parents:
        raise SystemExit("State must be outside the checkout; raw text stays local.")
    panels = {"sourced": ROOT / "evals/data/week7_sourced_passages.json"}
    if args.asset:
        panels["asset_test"] = args.asset.resolve()
    generation = {
        "runs": {"v7": str(ROOT / "models/runs/ensemble-v7-20260928"),
                 "contextual": str(ROOT / "models/runs/ensemble-context-similarity-20261002")},
        "t5": str(args.t5.resolve()), "tagger": str(args.tagger.resolve()),
        "encoder": str(args.encoder.resolve()), "nltk_data": str(args.nltk_data.resolve()),
        "languagetool": {"dir": str(args.languagetool.resolve()), "java": str(args.java.resolve()),
                         "port": args.languagetool_port},
    }
    artifacts = []
    for run in generation["runs"].values():
        artifacts += _run_dependencies(run)
    for key in ("t5", "tagger", "encoder"):
        artifacts += _files(generation[key])
    artifacts += _files(Path(generation["languagetool"]["dir"]) / "languagetool-server.jar")
    evaluation = {"embedding": str(args.embedding.resolve()), "nli": str(args.nli.resolve())}
    eval_artifacts = _files(args.embedding) + _files(args.nli)
    plan = {
        "created": datetime.now(timezone.utc).isoformat(),
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip(),
        "panels": {k: str(v) for k, v in panels.items()},
        "panel_sha256": {k: sha(v) for k, v in panels.items()},
        "generation": generation, "evaluation": evaluation,
        "generation_artifacts_sha256": {str(p): sha(p) for p in sorted(set(artifacts))},
        "evaluation_artifacts_sha256": {str(p): sha(p) for p in sorted(set(eval_artifacts))},
        "generation_sources_sha256": {str(p): sha(p) for p in generation_sources()},
        "scoring_sources_sha256": {str(p): sha(p) for p in scoring_sources()},
        "runtime": runtime(),
    }
    state.mkdir(parents=True, exist_ok=True)
    path = state / "plan.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a frozen plan is never overwritten")
    write_json(path, plan)
    print(f"Froze {path}: {len(plan['generation_artifacts_sha256'])} generation and "
          f"{len(plan['evaluation_artifacts_sha256'])} evaluation artifacts")


def runtime():
    return {"python": platform.python_version(), "host": socket.gethostname(),
            "packages": {n: version(n) for n in PACKAGES}}


def preflight(plan, check_artifacts=False):
    for key in ("generation_sources_sha256", "scoring_sources_sha256"):
        for path, expected in plan[key].items():
            if sha(path) != expected:
                raise ValueError(f"Frozen source changed: {path}; freeze a new plan")
    for key in ("generation_artifacts_sha256", "evaluation_artifacts_sha256"):
        for path, expected in plan[key].items():
            if not Path(path).is_file() or (check_artifacts and sha(path) != expected):
                raise ValueError(f"Frozen artifact missing/changed: {path}")
    for name, path in plan["panels"].items():
        if sha(path) != plan["panel_sha256"][name]:
            raise ValueError(f"Panel changed: {name}")
    if runtime() != plan["runtime"]:
        raise ValueError("Runtime differs from the frozen plan")


def _relative(hashes):
    return sorted((str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else p, h)
                  for p, h in hashes.items())


def signatures(plan):
    gen = digest({"generation": plan["generation"], "artifacts": plan["generation_artifacts_sha256"],
                  "sources": _relative(plan["generation_sources_sha256"]),
                  "harness": hashlib.sha256(inspect.getsource(StudyGenerator).encode()).hexdigest(),
                  "version": GENERATION_VERSION, "runtime": plan["runtime"]})
    score = digest({"evaluation": plan["evaluation"], "artifacts": plan["evaluation_artifacts_sha256"],
                    "sources": _relative(plan["scoring_sources_sha256"]),
                    "harness": hashlib.sha256(inspect.getsource(StudyScorer).encode()).hexdigest(),
                    "version": SCORING_VERSION, "runtime": plan["runtime"]})
    return gen, score


# --- storage ------------------------------------------------------------------

def _plain(value):
    return json.loads(json.dumps(value, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))


class StudyStore:
    """Outputs keyed by generation signature; metrics by scoring signature."""

    def __init__(self, root, generation_signature, scoring_signature):
        self.root = Path(root)
        self.gen, self.score = generation_signature, scoring_signature

    def _output_path(self, passage, name):
        return self.root / "outputs" / self.gen[:16] / (digest([passage["id"], name]) + ".json")

    def _metric_path(self, source, output):
        return self.root / "metrics" / self.score[:16] / (digest([source, output]) + ".json")

    def _identity(self, passage, name):
        return {"signature": self.gen, "id": passage["id"], "config": name,
                "source_sha256": digest(passage["original"])}

    def output(self, passage, name):
        path = self._output_path(passage, name)
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value.get("identity") != self._identity(passage, name) or not isinstance(value.get("output"), str):
            raise ValueError(f"Output checkpoint identity changed: {passage['id']} / {name}")
        return value

    def save_output(self, passage, name, result):
        if not isinstance(result.get("output"), str):
            raise ValueError("System returned a non-string output")
        entry = {"identity": self._identity(passage, name), **_plain(result)}
        write_json(self._output_path(passage, name), entry)
        return entry

    def metrics(self, source, output):
        path = self._metric_path(source, output)
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value.get("pair") != {"source": source, "output": output} or value.get("signature") != self.score:
            raise ValueError("Metric cache identity changed")
        return value

    def save_metrics(self, source, output, metrics):
        entry = {"signature": self.score, "pair": {"source": source, "output": output}, **_plain(metrics)}
        write_json(self._metric_path(source, output), entry)
        return entry


class DiskMemo:
    """JSON memo for deterministic stage calls, keyed by exact input."""

    def __init__(self, root, namespace, signature):
        self.dir = Path(root) / "memo" / signature[:16] / namespace
        self.signature = signature

    def _path(self, key):
        return self.dir / (digest(key) + ".json")

    def __contains__(self, key):
        return self._path(key).exists()

    def __getitem__(self, key):
        value = json.loads(self._path(key).read_text())
        if value["key"] != key or value["signature"] != self.signature:
            raise ValueError("Memo identity changed")
        return value["value"]

    def __setitem__(self, key, value):
        write_json(self._path(key), {"signature": self.signature, "key": key, "value": _plain(value)})


# --- generation ---------------------------------------------------------------

class StudyGenerator:
    """Loads each stage once and maps (config, text) to the pipeline result."""

    def __init__(self, plan, state, signature, threads=1):
        self.plan, self.state, self.signature, self.threads = plan, Path(state), signature, threads
        self.pipes, self.decisions, self.dirty = {}, {}, set()
        self._t5 = self._compound = self._grammar = self._dictionary = None

    def _limit_threads(self):
        import torch
        torch.set_num_threads(self.threads)

    def pipe(self, level):
        level = "v7" if level == "off" else level
        if level in self.pipes:
            return self.pipes[level]
        import nltk
        nltk.data.path[:] = [self.plan["generation"]["nltk_data"]]
        from cleartext import contextual
        contextual.NAME = self.plan["generation"]["encoder"]
        from cleartext.ensemble_pipeline import EnsembleClearText
        pipe = EnsembleClearText.load(Path(self.plan["generation"]["runs"][level]))
        if not getattr(contextual.model, "_study_wrapped", False):
            load = contextual.model

            def model():
                result = load()
                self._limit_threads()  # The production loader sets four threads.
                return result

            model._study_wrapped = True
            contextual.model = model
        self._limit_threads()
        # Decision rows depend on the word-stage text, target, the original text
        # and the threshold only; term protection and guardrails run live after.
        current = {}
        rank_word, decide = pipe.rank_word, pipe.decide

        def cached_rank(doc, token, original_doc=None, guard=True):
            context = original_doc if original_doc is not None else doc
            current["key"] = (doc.text, token.idx, context.text, pipe.decision_threshold)
            try:
                return rank_word(doc, token, original_doc, guard)
            finally:
                current.pop("key", None)

        def cached_decide(doc, token, slot, candidates):
            key = current.get("key")
            if key is None:
                return decide(doc, token, slot, candidates)
            cache = self.decisions.setdefault(level, {})
            if key not in cache:
                rows = decide(doc, token, slot, candidates)
                cache[key] = (copy.deepcopy(rows), copy.deepcopy(slot.cache.get("senses")))
                self.dirty.add(level)
            rows, senses = cache[key]
            if senses is not None:
                slot.cache["senses"] = copy.deepcopy(senses)
            return copy.deepcopy(rows)

        pipe.rank_word, pipe.decide = cached_rank, cached_decide
        self.pipes[level] = pipe
        return pipe

    def _decision_path(self, level, passage):
        return (self.state / "memo" / self.signature[:16] / "decisions" / level
                / (digest([passage["id"], passage["original"]]) + ".pkl"))

    def begin(self, passage):
        """Bound memory to one passage and reload its persisted decision rows."""
        self.decisions = {}
        self.dirty = set()
        for level in self.plan["generation"]["runs"]:
            path = self._decision_path(level, passage)
            if path.exists():
                value = pickle.loads(path.read_bytes())
                if value["signature"] != self.signature or value["id"] != passage["id"]:
                    raise ValueError("Decision cache identity changed")
                self.decisions[level] = value["cache"]

    def end(self, passage):
        for level in self.dirty:
            path = self._decision_path(level, passage)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(pickle.dumps({"signature": self.signature, "id": passage["id"],
                                                "cache": self.decisions[level]}))
            temporary.replace(path)
        self.dirty = set()

    def structure_rewriter(self, level):
        if level == "t5":
            if self._t5 is None:
                from cleartext.structure.seq2seq import SentenceGenerator
                from evals.pipeline_benchmark import SentencewiseGenerator
                generator = SentenceGenerator(self.plan["generation"]["t5"], device="cpu")
                self._limit_threads()
                memo = DiskMemo(self.state, "t5", self.signature)

                class Remembered:
                    def predict(_, sentence):
                        if sentence not in memo:
                            result = generator.predict(sentence)
                            memo[sentence] = {"output": result["output"], "raw": result["raw"]}
                        return memo[sentence]

                self._t5 = SentencewiseGenerator(Remembered(), cache={})
            return self._t5
        if level == "compound_tagger":
            if self._compound is None:
                from cleartext.structure.compound import DependencyCompoundSplitter
                self._compound = DependencyCompoundSplitter(guards="scoped",
                                                            tagger=self.plan["generation"]["tagger"])
            return self._compound
        return None

    def grammar(self):
        if self._grammar is None:
            from cleartext.grammar_preprocessing import GrammarCorrector
            corrector = GrammarCorrector(server=f"http://127.0.0.1:{self.plan['generation']['languagetool']['port']}")
            memo = DiskMemo(self.state, "grammar", self.signature)

            class Remembered:
                def correct(_, text):
                    if text not in memo:
                        result = corrector.correct(text)
                        if not result.get("available"):
                            raise RuntimeError("Grammar service unavailable: " + str(result.get("reason")))
                        memo[text] = result
                    return memo[text]

            self._grammar = Remembered()
        return self._grammar

    def predict(self, name, text):
        if name == "original":
            return {"output": text, "stages": []}
        if name == "rule_based":
            if self._dictionary is None:
                from evals.baselines import RuleBasedSimplifier
                self._dictionary = RuleBasedSimplifier()
            return {"output": self._dictionary(text), "stages": ["dictionary"]}
        config = parse_id(name)
        pipe = self.pipe(config["words"])
        previous = pipe.technical_terms
        pipe.technical_terms = config["terms"] == "on"
        try:
            result = pipe.analyze(
                text,
                structure=config["structure"] != "off",
                phrases=config["phrases"] != "off",
                words=config["words"] != "off",
                structure_rewriter=self.structure_rewriter(config["structure"]),
                grammar_corrector=self.grammar() if config["grammar"] == "on" else None,
                guardrails=True,
                negation=config["negation"] == "on",
            )
        finally:
            pipe.technical_terms = previous
        stages = [e.get("stage", "structure" if "rule" in e else "unknown") for e in result["edits"]]
        return {"output": result["output"], "stages": stages, "edits": result["edits"],
                "blocked": {k: result[k] for k in ("term_preservation", "literal_preservation") if k in result}}


# --- scoring ------------------------------------------------------------------

class StudyScorer:
    """Week 7 readability/preservation metrics plus sentence-level aligned NLI."""

    def __init__(self, plan, state, signature, threads=1):
        from . import preservation, runner
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline
        import torch
        torch.set_num_threads(threads)
        self.plan, self.runner, self.preservation = plan, runner, preservation
        embedding, nli = plan["evaluation"]["embedding"], plan["evaluation"]["nli"]
        preservation._EMB_MODELS[embedding] = SentenceTransformer(embedding, device="cpu")
        preservation._NLI_PIPES[nli] = pipeline("text-classification", model=nli, top_k=None,
                                                padding=True, device=-1)
        self.memo = DiskMemo(state, "nli-pairs", signature)

    def entail(self, premise, hypothesis):
        key = [premise, hypothesis]
        if key not in self.memo:
            self.memo[key] = self.preservation.entailment_probability(
                premise, hypothesis, self.plan["evaluation"]["nli"])
        return self.memo[key]

    def __call__(self, source, output):
        embedding, nli = self.plan["evaluation"]["embedding"], self.plan["evaluation"]["nli"]
        return {
            "readability": self.runner.readability_report(output),
            "preservation": self.runner.preservation_report(source, output, embedding_model=embedding,
                                                            nli_model=nli),
            "sentence_nli": self.preservation.sentence_nli(source, output, nli, entail=self.entail),
        }


# --- batches ------------------------------------------------------------------

def load_panel(plan, panel):
    rows = json.loads(Path(plan["panels"][panel]).read_text())
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Panel IDs must be unique")
    return rows


def design_names(path):
    rows = json.loads(Path(path).read_text())
    configs = rows["configs"] if isinstance(rows, dict) else rows
    names = [c if isinstance(c, str) else config_id(c) for c in configs]
    for name in names:
        parse_id(name)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate configs in design")
    return names


def run(args):
    plan = json.loads(args.plan.read_text())
    preflight(plan, check_artifacts=not args.preflight)
    if args.preflight:
        print("Preflight passed")
        return
    gen_sig, score_sig = signatures(plan)
    names = [*REFERENCES, *[n for n in design_names(args.design) if n not in REFERENCES]]
    passages = load_panel(plan, args.panel)
    shard, shards = map(int, args.shard.split("/"))
    selected = [p for k, p in enumerate(passages) if k % shards == shard]
    state = args.state
    store = StudyStore(state, gen_sig, score_sig)
    import torch
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:  # Already fixed for this process.
        pass
    started, deadline = time.monotonic(), time.monotonic() + args.max_seconds
    (state / "locks").mkdir(parents=True, exist_ok=True)
    with (state / "locks" / f"{args.panel}-{shard}-of-{shards}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generator, scorer = None, None

        def log(message):
            print(f"{time.strftime('%H:%M:%S')} +{time.monotonic() - started:.0f}s {message}", flush=True)

        done = True
        for passage in selected:
            missing = [n for n in names if store.output(passage, n) is None]
            pairs = []
            if missing:
                generator = generator or StudyGenerator(plan, state, gen_sig, args.threads)
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
                        log(f"output {passage['id']} {name} {result['seconds']:.1f}s")
                finally:
                    generator.end(passage)
            if not done:
                break
            if args.score:
                for name in names:
                    entry = store.output(passage, name)
                    if store.metrics(passage["original"], entry["output"]) is None:
                        pairs.append(entry["output"])
                for output in dict.fromkeys(pairs):
                    if time.monotonic() >= deadline:
                        done = False
                        break
                    scorer = scorer or StudyScorer(plan, state, score_sig, args.threads)
                    store.save_metrics(passage["original"], output, scorer(passage["original"], output))
                    log(f"scored {passage['id']} ({len(pairs)} new pairs)")
            if not done:
                break
        complete = all(store.output(p, n) is not None and (not args.score or store.metrics(
            p["original"], store.output(p, n)["output"]) is not None) for p in selected for n in names)
    preflight(plan, check_artifacts=True)
    receipt = {"host": socket.gethostname(), "panel": args.panel, "shard": args.shard,
               "design": str(args.design), "design_sha256": sha(args.design), "configs": len(names),
               "seconds": time.monotonic() - started, "generation_signature": gen_sig,
               "scoring_signature": score_sig, "shard_complete": complete,
               "frozen_inputs_rechecked": True}
    write_json(state / "batches" / f"{time.time_ns()}.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def grammar_memo(args):
    """Fill the grammar memo for every panel input, so shards need no server."""
    plan = json.loads(args.plan.read_text())
    preflight(plan)
    generator = StudyGenerator(plan, args.state, signatures(plan)[0])
    corrector, changed, total = generator.grammar(), 0, 0
    for panel in plan["panels"]:
        for passage in load_panel(plan, panel):
            changed += corrector.correct(passage["original"])["output"] != passage["original"]
            total += 1
    receipt = {"inputs": total, "changed": changed, "generation_signature": signatures(plan)[0]}
    write_json(args.state / "grammar-memo-receipt.json", receipt)
    print(json.dumps(receipt))


# --- report -------------------------------------------------------------------

def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def noninitial_entities_preserved(original, output):
    """Alternative (post hoc) entity check: capitalized spans that do not start
    a sentence, so discourse words like "However" are not counted as names."""
    import re
    from .preservation import _ENTITY_RE, split_sentences
    starts = {a + len(original[a:b]) - len(original[a:b].lstrip("\"'([“‘ \n\t")) for a, b in split_sentences(original)}
    items = list(dict.fromkeys(m.group() for m in _ENTITY_RE.finditer(original) if m.start() not in starts))
    if not items:
        return 1.0
    low = output.lower()
    return sum(bool(re.search(r"(?<!\w)" + re.escape(e.lower()) + r"(?!\w)", low)) for e in items) / len(items)


def summarize(passages, outputs, metrics):
    """Aggregate one config: outputs/metrics are {passage_id: ...}."""
    rows = [(p, outputs[p["id"]], metrics.get(p["id"])) for p in passages]
    pres = lambda m, k: m["preservation"][k] if m else None
    nli = [pres(m, "nli_mean") for _, _, m in rows]
    sent = [m["sentence_nli"]["sent_nli_mean"] if m else None for _, _, m in rows]
    changed = [o["output"] != p["original"] for p, o, _ in rows]
    stage_counts = {}
    for _, o, _ in rows:
        for stage in set(o.get("stages", [])):
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
    out = {
        "n": len(rows), "changed": sum(changed), "stage_passages": stage_counts,
        "fk_grade": _mean(m["readability"]["flesch_kincaid_grade"] for _, _, m in rows if m),
        "flesch_ease": _mean(m["readability"]["flesch_reading_ease"] for _, _, m in rows if m),
        "sentence_length": _mean(m["readability"]["avg_sentence_length"] for _, _, m in rows if m),
        "embedding": _mean(pres(m, "embedding_similarity") for _, _, m in rows),
        "nli_mean": _mean(nli), "nli_n": sum(v is not None for v in nli),
        "nli_abstained": sum(v is None for v in nli),
        "nli_below_080": sum(v is None or v < 0.80 for v in nli),
        "sent_nli_mean": _mean(sent), "sent_nli_below_080": sum(v is None or v < 0.80 for v in sent),
        "sent_nli_changed_min_mean": _mean(m["sentence_nli"]["sent_nli_changed_min"] for _, _, m in rows if m),
        "entities": _mean(pres(m, "entities_preserved") for _, _, m in rows),
        "entities_noninitial": _mean(noninitial_entities_preserved(p["original"], o["output"]) for p, o, _ in rows),
        "numbers_all": all(pres(m, "numbers_preserved") == 1.0 for _, _, m in rows),
        "numbers_failed": sum(pres(m, "numbers_preserved") != 1.0 for _, _, m in rows),
        "negations_all": all(pres(m, "negations_preserved") == 1.0 for _, _, m in rows),
        "negations_failed": sum(pres(m, "negations_preserved") != 1.0 for _, _, m in rows),
        "tenses": _mean(pres(m, "tenses_preserved") for _, _, m in rows),
        "tfidf": _mean(pres(m, "tfidf_cosine_similarity") for _, _, m in rows),
    }
    if out["changed"]:
        idx = [k for k, c in enumerate(changed) if c]
        out["changed_only"] = {"fk_grade": _mean(rows[k][2]["readability"]["flesch_kincaid_grade"] for k in idx),
                               "nli_mean": _mean(nli[k] for k in idx), "sent_nli_mean": _mean(sent[k] for k in idx)}
    return out


def gates(summary, rule_based_entities, nli_key="nli_mean", below_key="nli_below_080", entity_key="entities"):
    """Frozen SELECTION_RULE.md gates. Keys are swappable only for labeled alternatives."""
    limit = 0.05 * summary["n"]
    return {
        "nli_mean_ge_095": summary[nli_key] is not None and summary[nli_key] >= 0.95,
        "nli_below_080_le_5pct": summary[below_key] <= limit,
        "numbers_negations_all": summary["numbers_all"] and summary["negations_all"],
        "entities_ge_rule_based": summary[entity_key] is not None and summary[entity_key] >= rule_based_entities,
    }


def select(summaries, rule_based_entities, sari=None, **gate_keys):
    """Lowest FK among configs passing all gates; ties within 0.1 by higher SARI."""
    passing = {n: s for n, s in summaries.items() if n not in REFERENCES
               and all(gates(s, rule_based_entities, **gate_keys).values())}
    if not passing:
        return {"winner": None, "passing": [], "tied": []}
    best = min(s["fk_grade"] for s in passing.values())
    tied = sorted(n for n, s in passing.items() if s["fk_grade"] <= best + 0.1)
    winner = min(passing, key=lambda n: passing[n]["fk_grade"])
    unresolved = False
    if len(tied) > 1:
        if sari and all(n in sari for n in tied):
            winner = max(tied, key=lambda n: (sari[n], -passing[n]["fk_grade"]))
        else:
            unresolved = True
    return {"winner": winner, "passing": sorted(passing, key=lambda n: passing[n]["fk_grade"]),
            "tied": tied, "tie_unresolved_missing_sari": unresolved}


def pareto(summaries, x="fk_grade", y="nli_mean"):
    """Configs not dominated on (lower x, higher y)."""
    items = {n: s for n, s in summaries.items() if s[x] is not None and s[y] is not None}
    front = []
    for n, s in items.items():
        if not any((t[x] <= s[x] and t[y] >= s[y]) and (t[x] < s[x] or t[y] > s[y])
                   for m, t in items.items() if m != n):
            front.append(n)
    return sorted(front, key=lambda n: items[n][x])


def collect(states, plan, panel, names, need_metrics=True):
    """{name: (outputs, metrics)} for complete configs; pending names listed separately."""
    passages = load_panel(plan, panel)
    stores = []
    for state in states:
        state_plan = json.loads((Path(state) / "plan.json").read_text())
        stores.append(StudyStore(state, *signatures(state_plan)))
    complete, pending = {}, []
    for name in names:
        outputs, metrics, ok = {}, {}, True
        for p in passages:
            found = [s.output(p, name) for s in stores]
            found = [f for f in found if f is not None]
            if not found:
                ok = False
                break
            if len({f["output"] for f in found}) != 1:
                raise ValueError(f"Conflicting outputs across states: {p['id']} / {name}")
            outputs[p["id"]] = found[0]
            if need_metrics:
                m = next((s.metrics(p["original"], found[0]["output"]) for s in stores
                          if s.metrics(p["original"], found[0]["output"]) is not None), None)
                if m is None:
                    ok = False
                    break
                metrics[p["id"]] = m
        if ok:
            complete[name] = (outputs, metrics)
        else:
            pending.append(name)
    return passages, complete, pending


def versus_baseline(passages, complete, summaries, base=None):
    """Paired differences from the v7 baseline and the pre-registered activity test."""
    base = base or config_id(BASELINE)
    if base not in complete:
        return {}
    b, out = summaries[base], {}
    for name, (outputs, _) in complete.items():
        s = summaries[name]
        delta = {k: (s[k] - b[k]) if s[k] is not None and b[k] is not None else None
                 for k in ("fk_grade", "nli_mean", "sent_nli_mean")}
        differs = sum(outputs[p["id"]]["output"] != complete[base][0][p["id"]]["output"] for p in passages)
        active = (abs(delta["fk_grade"] or 0) >= 0.02 or abs(delta["nli_mean"] or 0) >= 0.005
                  or abs(delta["sent_nli_mean"] or 0) >= 0.005 or differs >= 3)
        out[name] = {**delta, "outputs_differ": differs, "active": active}
    return out


def report(args):
    from .readability import flesch_kincaid_grade
    from .sari import corpus_sari
    plan = json.loads((args.state[0] / "plan.json").read_text())
    names = [*REFERENCES]
    for design in args.design:
        names += [n for n in design_names(design) if n not in names]
    passages, complete, pending = collect(args.state, plan, "sourced", names)
    summaries = {n: summarize(passages, *complete[n]) for n in complete}
    sari, sari_hf, asset_fk = {}, {}, {}
    if "asset_test" in plan["panels"]:
        asset, asset_complete, _ = collect(args.state, plan, "asset_test", names, need_metrics=False)
        for name, (outputs, _) in asset_complete.items():
            preds = [outputs[p["id"]]["output"] for p in asset]
            args3 = ([p["original"] for p in asset], preds, [p["references"] for p in asset])
            sari[name] = corpus_sari(*args3, zero_division=0)
            sari_hf[name] = corpus_sari(*args3, zero_division=1)
            asset_fk[name] = _mean(flesch_kincaid_grade(t) for t in preds)
    rb = summaries["rule_based"]["entities"]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "panel": {"name": "sourced", "n": len(passages), "sha256": plan["panel_sha256"]["sourced"]},
        "asset_test_n": len(load_panel(plan, "asset_test")) if "asset_test" in plan["panels"] else None,
        "configs": summaries, "pending": pending, "asset_sari": sari, "asset_sari_evaluate_convention": sari_hf,
        "asset_sari_note": "asset_sari uses the original Xu et al. 0/0=0 convention (ranking/tie-break); "
                           "asset_sari_evaluate_convention uses Hugging Face evaluate's 0/0=1, which rewards copying.",
        "asset_fk_grade": asset_fk,
        "gates": {n: gates(s, rb) for n, s in summaries.items() if n not in REFERENCES},
        "frozen_selection": select(summaries, rb, sari),
        # Post hoc, labeled alternatives (SELECTION_RULE.md stays frozen).
        "alternative_noninitial_entities": {
            "rule": "Frozen gates 1-3; gate 4 uses non-sentence-initial capitalized spans vs rule_based",
            "rule_based_entities": summaries["rule_based"]["entities_noninitial"],
            "gates": {n: gates(s, summaries["rule_based"]["entities_noninitial"], entity_key="entities_noninitial")
                      for n, s in summaries.items() if n not in REFERENCES},
            "selection": select(summaries, summaries["rule_based"]["entities_noninitial"], sari,
                                entity_key="entities_noninitial")},
        "alternative_sentence_nli": {
            "rule": "Gates 1-2 on sentence-level aligned NLI; gate 4 on non-sentence-initial entities",
            "selection": select(summaries, summaries["rule_based"]["entities_noninitial"], sari,
                                nli_key="sent_nli_mean", below_key="sent_nli_below_080",
                                entity_key="entities_noninitial")},
        "pareto_fk_vs_nli": pareto({n: s for n, s in summaries.items()}),
        "pareto_fk_vs_sentence_nli": pareto(summaries, y="sent_nli_mean"),
        "rule_based_entities": rb,
        "versus_baseline": versus_baseline(passages, complete, summaries),
        "signatures": dict(zip(("generation", "scoring"), signatures(plan))),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "aggregate-results.json", result)
    (args.out / "results-table.md").write_text(markdown(result))
    print(f"{len(summaries)} complete configs, {len(pending)} pending -> {args.out}")


def markdown(result):
    head = ("| Config | Changed | FK ↓ | Flesch ↑ | Words/sent | NLI passage | NLI<0.8 | "
            "NLI sentence | Sent<0.8 | Entities | Num/neg fails | Gates | ASSET SARI |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|\n")
    lines = []
    front = set(result["pareto_fk_vs_nli"])
    for name, s in sorted(result["configs"].items(), key=lambda kv: kv[1]["fk_grade"]):
        g = result["gates"].get(name)
        gate = "ref" if g is None else ("pass" if all(g.values()) else
                                        "fail " + ",".join(str(k + 1) for k, v in enumerate(g.values()) if not v))
        label = name.replace("|", " ") + (" ◆" if name in front else "")
        sari = result["asset_sari"].get(name)
        lines.append(
            f"| {label} | {s['changed']}/{s['n']} | {s['fk_grade']:.3f} | {s['flesch_ease']:.2f} | "
            f"{s['sentence_length']:.2f} | {s['nli_mean']:.4f} | {s['nli_below_080']} | "
            f"{s['sent_nli_mean']:.4f} | {s['sent_nli_below_080']} | {s['entities']:.4f} | "
            f"{s['numbers_failed']}/{s['negations_failed']} | {gate} | "
            f"{'' if sari is None else f'{sari:.2f}'} |")
    return head + "\n".join(lines) + "\n\n◆ = on the FK/passage-NLI Pareto frontier.\n"


def settings(path):
    """analyze() settings for a study config file such as recommended-config.json."""
    raw = json.loads(Path(path).read_text())
    config = normalize(raw["factors"])
    if config["grammar"] == "on" or config["structure"] == "compound_tagger":
        raise ValueError("This loader covers the recommended stages; use StudyGenerator for others")
    checkpoint = raw.get("structure_checkpoint", {}).get("path")
    if config["structure"] == "t5" and not checkpoint:
        raise ValueError("structure=t5 needs structure_checkpoint.path")
    run = raw.get("pipeline_run", "models/runs/ensemble-v7-20260928")
    if config["words"] == "contextual":
        run = raw.get("contextual_run", "models/runs/ensemble-context-similarity-20261002")
    return {"run": str(ROOT / run), "technical_terms": config["terms"] == "on",
            "t5": str(Path(checkpoint).expanduser()) if config["structure"] == "t5" else None,
            "analyze": {"structure": config["structure"] != "off", "phrases": config["phrases"] != "off",
                        "words": config["words"] != "off", "negation": config["negation"] == "on",
                        "guardrails": raw.get("guardrails", True)}}


def simplifier(path):
    """Opt-in text -> output callable for a study config file; loads local models."""
    from cleartext.ensemble_pipeline import EnsembleClearText
    chosen = settings(path)
    pipe = EnsembleClearText.load(Path(chosen["run"]))
    pipe.technical_terms = chosen["technical_terms"]
    rewriter = None
    if chosen["t5"]:
        from cleartext.structure.seq2seq import SentenceGenerator
        from .pipeline_benchmark import SentencewiseGenerator
        rewriter = SentencewiseGenerator(SentenceGenerator(chosen["t5"], device="cpu"))
    return lambda text: pipe.analyze(text, structure_rewriter=rewriter, **chosen["analyze"])["output"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--state", type=Path, required=True)
    for name in ("t5", "tagger", "encoder", "nltk-data", "languagetool", "java", "embedding", "nli"):
        f.add_argument("--" + name, type=Path, required=True)
    f.add_argument("--asset", type=Path)
    f.add_argument("--languagetool-port", type=int, default=8797)
    r = sub.add_parser("run")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--state", type=Path, required=True)
    r.add_argument("--design", type=Path, required=True)
    r.add_argument("--panel", default="sourced")
    r.add_argument("--shard", default="0/1")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--max-seconds", type=int, default=1500)
    r.add_argument("--no-score", dest="score", action="store_false")
    r.add_argument("--preflight", action="store_true")
    g = sub.add_parser("grammar-memo")
    g.add_argument("--plan", type=Path, required=True)
    g.add_argument("--state", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--state", type=Path, action="append", required=True)
    p.add_argument("--design", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"freeze": freeze, "run": run, "grammar-memo": grammar_memo, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
