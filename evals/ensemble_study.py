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

`freeze --reuse-state OLD` lets a new plan read OLD's T5, grammar and lexical
decision memos and its metrics, only after checking that everything those
caches depend on is unchanged (see reuse_compatibility). Outputs are never
reused: old cells come in through `report --state OLD --state NEW`.
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

# Level order matters only for display. Levels after the first four rows' old
# ones were added on 2026-11-10 (ensemble v2): Chhoeng's phrase finder as
# "simpleppdb" (his SimplePPDB list) and "crf" (his default sources: manual,
# verified and the CRF detector with phrase_replacer.py), alone or after the
# authored rules; stacked/ablated T5; and a vetoed contextual word stage.
FACTORS = {
    "words": ("v7", "contextual", "off", "contextual_veto"),
    "phrases": ("authored", "off", "simpleppdb", "authored+simpleppdb", "crf", "authored+crf"),
    "structure": ("rules", "t5", "compound_tagger", "off", "t5+compound_tagger", "t5_raw",
                  "t5_unchecked"),
    "negation": ("off", "on"),
    "grammar": ("off", "on"),
    "terms": ("off", "on"),
    "passive": ("off", "on", "agentive"),
}
# Factors added after the first run, with the level that reproduces old behaviour.
LEGACY = {"passive": "off"}
FINDER_SOURCES = {"simpleppdb": ("simpleppdb",), "crf": ("manual", "verified", "model")}
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


def teammate_sources():
    """Chhoeng's stage code and data (passive/, pos_tag/), loaded by teammate_stages."""
    names = ["pos_tag.py", "phrase_finder.py", "phrase_model.py", "phrase_replacer.py",
             "manual_phrases.csv", "verified_phrases.csv", "simpleppdb_phrases.csv",
             "learned_replacements.csv", "phrase_crf.crfsuite", "phrase_crf.json",
             "replacer_settings.json"]
    return [ROOT / "passive/passive_to_active.py", *(ROOT / "pos_tag" / n for n in names)]


def generation_sources():
    return sorted([*(ROOT / "models/src/cleartext").rglob("*.py"), ROOT / "evals/baselines.py",
                   ROOT / "evals/pipeline_benchmark.py", ROOT / "evals/data/simplification_dict.json",
                   *teammate_sources()])


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
    if args.teammate_nli:
        # phrase_replacer.py's BERT, similarity and NLI models, pinned to local snapshots.
        generation["teammate_models"] = {"bert": str(args.encoder.resolve()),
                                         "similarity": str(args.embedding.resolve()),
                                         "nli": str(args.teammate_nli.resolve())}
        generation["teammate_packages"] = {n: version(n) for n in ("python-crfsuite",)}
    artifacts = []
    for run in generation["runs"].values():
        artifacts += _run_dependencies(run)
    for key in ("t5", "tagger", "encoder"):
        artifacts += _files(generation[key])
    for path in generation.get("teammate_models", {}).values():
        artifacts += _files(path)
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
    if args.reuse_state:
        plan["reuse"] = reuse_compatibility(args.reuse_state.resolve(), plan)
    state.mkdir(parents=True, exist_ok=True)
    path = state / "plan.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a frozen plan is never overwritten")
    write_json(path, plan)
    print(f"Froze {path}: {len(plan['generation_artifacts_sha256'])} generation and "
          f"{len(plan['evaluation_artifacts_sha256'])} evaluation artifacts")


def _source_at(head, path):
    rel = Path(path).relative_to(ROOT)
    return subprocess.run(["git", "show", f"{head}:{rel}"], cwd=ROOT, capture_output=True,
                          check=True).stdout


def _without_analyze(source):
    """ensemble_pipeline.py's AST minus EnsembleClearText.analyze (the only
    function the memoized stages do not call)."""
    import ast
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "EnsembleClearText":
            node.body = [n for n in node.body if not (isinstance(n, ast.FunctionDef) and n.name == "analyze")]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def reuse_compatibility(old_state, plan):
    """Allow memo/metric reuse from old_state only if their inputs are unchanged.

    T5 sentence outputs, grammar corrections and lexical decision rows depend
    on the frozen artifacts, the runtime, every source file except
    ensemble_pipeline.analyze (new opt-in stages live there and in new files)
    and the memo code in StudyGenerator, which keeps the same keys and values.
    Metrics are reused only under an identical scoring signature.
    """
    old = json.loads((old_state / "plan.json").read_text())
    problems = []
    if old["runtime"] != plan["runtime"]:
        problems.append("runtime")
    for key in ("runs", "t5", "tagger", "encoder", "nltk_data", "languagetool"):
        if old["generation"][key] != plan["generation"][key]:
            problems.append(f"generation.{key}")
    for path, digest_ in old["generation_artifacts_sha256"].items():
        if plan["generation_artifacts_sha256"].get(path) != digest_:
            problems.append(f"artifact {path}")
    pipeline = str(ROOT / "models/src/cleartext/ensemble_pipeline.py")
    for path, digest_ in old["generation_sources_sha256"].items():
        if path == pipeline:
            continue
        if plan["generation_sources_sha256"].get(path) != digest_:
            problems.append(f"source {path}")
    old_pipeline = _source_at(old["head"], pipeline)
    if hashlib.sha256(old_pipeline).hexdigest() != old["generation_sources_sha256"][pipeline]:
        problems.append("old ensemble_pipeline.py is not the frozen one")
    same_pipeline = _without_analyze(old_pipeline) == _without_analyze(Path(pipeline).read_bytes())
    if not same_pipeline:
        problems.append("ensemble_pipeline.py changed outside analyze()")
    added = sorted(set(plan["generation_sources_sha256"]) - set(old["generation_sources_sha256"]))
    if problems:
        raise SystemExit("Cannot reuse " + str(old_state) + ": " + "; ".join(problems))
    old_gen, old_score = state_signatures(old_state)
    return {"state": str(old_state), "generation_signature": old_gen, "scoring_signature": old_score,
            "memos": ["t5", "grammar", "decisions"],
            "metrics": old_score == signatures(plan)[1],
            "ensemble_pipeline_without_analyze_sha256": _without_analyze(old_pipeline),
            "sources_added": added, "old_head": old["head"]}


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


def state_signatures(state):
    """The (generation, scoring) signatures a state's batches ran under.

    Read from batch receipts, because signatures() hashes the current harness
    source and so changes for older states once StudyGenerator is edited.
    """
    found = {(r["generation_signature"], r["scoring_signature"])
             for r in (json.loads(p.read_text()) for p in (Path(state) / "batches").glob("*.json"))}
    if not found:
        return signatures(json.loads((Path(state) / "plan.json").read_text()))
    if len(found) != 1:
        raise ValueError(f"{state} has batches under {len(found)} signature pairs")
    return found.pop()


# --- storage ------------------------------------------------------------------

def _plain(value):
    return json.loads(json.dumps(value, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))


class StudyStore:
    """Outputs keyed by generation signature; metrics by scoring signature."""

    def __init__(self, root, generation_signature, scoring_signature, metric_fallback=None):
        self.root = Path(root)
        self.gen, self.score = generation_signature, scoring_signature
        # Another state's root with the same scoring signature (plan["reuse"]).
        self.metric_fallback = Path(metric_fallback) if metric_fallback else None

    def _output_path(self, passage, name):
        return self.root / "outputs" / self.gen[:16] / (digest([passage["id"], name]) + ".json")

    def _metric_path(self, source, output, root=None):
        return (root or self.root) / "metrics" / self.score[:16] / (digest([source, output]) + ".json")

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
        if not path.exists() and self.metric_fallback:
            path = self._metric_path(source, output, self.metric_fallback)
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

    def __init__(self, root, namespace, signature, fallback=None):
        self.dir = Path(root) / "memo" / signature[:16] / namespace
        self.signature = signature
        # Read-only (root, signature) of a compatible older state (plan["reuse"]).
        self.fallback = DiskMemo(fallback[0], namespace, fallback[1]) if fallback else None

    def _path(self, key):
        return self.dir / (digest(key) + ".json")

    def __contains__(self, key):
        return self._path(key).exists() or (self.fallback is not None and key in self.fallback)

    def __getitem__(self, key):
        if not self._path(key).exists() and self.fallback is not None:
            return self.fallback[key]
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
        self._remembered_t5, self._structure, self._passive, self._finders = None, {}, {}, {}
        self._veto = None
        reuse = plan.get("reuse")
        self.fallback = (reuse["state"], reuse["generation_signature"]) if reuse else None

    def _limit_threads(self):
        import torch
        torch.set_num_threads(self.threads)

    def pipe(self, level):
        level = {"off": "v7", "contextual_veto": "contextual"}.get(level, level)
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

    def _decision_path(self, level, passage, root=None, signature=None):
        return (Path(root or self.state) / "memo" / (signature or self.signature)[:16] / "decisions"
                / level / (digest([passage["id"], passage["original"]]) + ".pkl"))

    def begin(self, passage):
        """Bound memory to one passage and reload its persisted decision rows."""
        self.decisions = {}
        self.dirty = set()
        for level in self.plan["generation"]["runs"]:
            path, signature = self._decision_path(level, passage), self.signature
            if not path.exists() and self.fallback:
                path, signature = self._decision_path(level, passage, *self.fallback), self.fallback[1]
            if path.exists():
                value = pickle.loads(path.read_bytes())
                if value["signature"] != signature or value["id"] != passage["id"]:
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

    def _t5_generator(self):
        if self._remembered_t5 is None:
            generator = []  # Loaded on the first memo miss only.
            memo = DiskMemo(self.state, "t5", self.signature, self.fallback)
            study = self

            class Remembered:
                def predict(_, sentence):
                    if sentence not in memo:
                        if not generator:
                            from cleartext.structure.seq2seq import SentenceGenerator
                            generator.append(SentenceGenerator(study.plan["generation"]["t5"], device="cpu"))
                            study._limit_threads()
                        result = generator[0].predict(sentence)
                        memo[sentence] = {"output": result["output"], "raw": result["raw"]}
                    return memo[sentence]

            self._remembered_t5 = Remembered()
        return self._remembered_t5

    def structure_rewriter(self, level):
        """t5: per-sentence generation checks, then the passage structure checks.
        t5_raw drops the generation checks; t5_unchecked drops both."""
        if level in ("t5", "t5_raw", "t5_unchecked"):
            if level not in self._structure:
                from evals.pipeline_benchmark import SentencewiseGenerator
                model = SentencewiseGenerator(self._t5_generator(), guarded=level == "t5", cache={})
                if level == "t5_unchecked":
                    model.validate = lambda source, output: {"pass": True, "checks": {}, "failed": []}
                self._structure[level] = model
            return self._structure[level]
        if level == "t5+compound_tagger":
            return [self.structure_rewriter("t5"), self.structure_rewriter("compound_tagger")]
        if level == "compound_tagger":
            if self._compound is None:
                from cleartext.structure.compound import DependencyCompoundSplitter
                self._compound = DependencyCompoundSplitter(guards="scoped",
                                                            tagger=self.plan["generation"]["tagger"])
            return self._compound
        return None

    def passive(self, level):
        if level == "off":
            return None
        if level not in self._passive:
            from cleartext.teammate_stages import PassiveStage
            self._passive[level] = PassiveStage(placeholder="auto" if level == "on" else None)
        return self._passive[level]

    def phrase_finder(self, level):
        """Chhoeng's finder for a phrases level, memoized per input text."""
        name = level.split("+")[-1]
        if name not in FINDER_SOURCES:
            return None
        if name not in self._finders:
            memo = DiskMemo(self.state, "finder-" + name, self.signature)
            study, stage = self, []

            class Remembered:
                def rewrite(_, text):
                    if text not in memo:
                        if not stage:
                            from cleartext.teammate_stages import PhraseFinderStage
                            stage.append(PhraseFinderStage(FINDER_SOURCES[name],
                                                           study.plan["generation"]["teammate_models"]))
                            study._limit_threads()
                        output, edits = stage[0].rewrite(text)
                        memo[text] = {"output": output, "edits": edits}
                    value = memo[text]
                    return value["output"], copy.deepcopy(value["edits"])

            self._finders[name] = Remembered()
        return self._finders[name]

    def word_veto(self, level):
        if level != "contextual_veto":
            return None
        if self._veto is None:
            from cleartext.teammate_stages import EntailmentVeto
            self._veto = EntailmentVeto(model=self.plan["generation"]["teammate_models"]["nli"])
            self._limit_threads()
        return self._veto

    def grammar(self):
        if self._grammar is None:
            from cleartext.grammar_preprocessing import GrammarCorrector
            corrector = GrammarCorrector(server=f"http://127.0.0.1:{self.plan['generation']['languagetool']['port']}")
            memo = DiskMemo(self.state, "grammar", self.signature, self.fallback)

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
                phrases=config["phrases"] in ("authored", "authored+simpleppdb", "authored+crf"),
                words=config["words"] != "off",
                structure_rewriter=self.structure_rewriter(config["structure"]),
                grammar_corrector=self.grammar() if config["grammar"] == "on" else None,
                guardrails=True,
                negation=config["negation"] == "on",
                passive_converter=self.passive(config["passive"]),
                phrase_finder=self.phrase_finder(config["phrases"]),
                word_veto=self.word_veto(config["words"]),
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
    reuse = plan.get("reuse") or {}
    store = StudyStore(state, gen_sig, score_sig, reuse.get("state") if reuse.get("metrics") else None)
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
        stores.append(StudyStore(state, *state_signatures(state)))
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


def paired_delta(values, base, resamples=10000, seed=20261110):
    """Mean per-passage difference (values - base) with a percentile bootstrap 95% CI."""
    import numpy as np
    pairs = [(v, b) for v, b in zip(values, base) if v is not None and b is not None]
    d = np.array([v - b for v, b in pairs], dtype=float)
    means = d[np.random.default_rng(seed).integers(0, len(d), (resamples, len(d)))].mean(axis=1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n": len(d), "passages_nonzero": int((np.abs(d) > 1e-12).sum())}


def per_passage(passages, complete):
    """Per-passage FK and passage NLI (numbers only; no text) for each complete config."""
    out = {}
    for name, (_, metrics) in complete.items():
        out[name] = {"fk_grade": [metrics[p["id"]]["readability"]["flesch_kincaid_grade"] for p in passages],
                     "nli_mean": [metrics[p["id"]]["preservation"]["nli_mean"] for p in passages]}
    return out


def versus_best(passages, complete, best, per):
    """Paired deltas from the best config and the passages each config changes relative to it."""
    if best not in complete:
        return {}
    out = {}
    for name, (outputs, _) in complete.items():
        changed = [p["id"] for p in passages if outputs[p["id"]]["output"] != complete[best][0][p["id"]]["output"]]
        out[name] = {"fk_grade": paired_delta(per[name]["fk_grade"], per[best]["fk_grade"]),
                     "nli_mean": paired_delta(per[name]["nli_mean"], per[best]["nli_mean"]),
                     "outputs_differ": len(changed), "passages_differ": changed}
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
    per = per_passage(passages, complete)
    # The tie-break convention pre-registered with the harness (31f2334) is
    # evaluate's 0/0=1; the Xu 0/0=0 ranking is reported alongside (PR #13 disclosure).
    tiebreak = sari_hf if args.tiebreak == "evaluate" else sari
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "panel": {"name": "sourced", "n": len(passages), "sha256": plan["panel_sha256"]["sourced"]},
        "asset_test_n": len(load_panel(plan, "asset_test")) if "asset_test" in plan["panels"] else None,
        "configs": summaries, "pending": pending, "asset_sari": sari, "asset_sari_evaluate_convention": sari_hf,
        "asset_sari_note": "asset_sari uses the original Xu et al. 0/0=0 convention (ranking/tie-break); "
                           "asset_sari_evaluate_convention uses Hugging Face evaluate's 0/0=1, which rewards copying.",
        "asset_fk_grade": asset_fk,
        "gates": {n: gates(s, rb) for n, s in summaries.items() if n not in REFERENCES},
        "tiebreak_sari_convention": "0/0=1 (evaluate)" if args.tiebreak == "evaluate" else "0/0=0 (Xu)",
        "frozen_selection": select(summaries, rb, tiebreak),
        "frozen_selection_other_sari_convention": select(summaries, rb, sari if tiebreak is sari_hf else sari_hf),
        # Post hoc, labeled alternatives (SELECTION_RULE.md stays frozen).
        "alternative_noninitial_entities": {
            "rule": "Frozen gates 1-3; gate 4 uses non-sentence-initial capitalized spans vs rule_based",
            "rule_based_entities": summaries["rule_based"]["entities_noninitial"],
            "gates": {n: gates(s, summaries["rule_based"]["entities_noninitial"], entity_key="entities_noninitial")
                      for n, s in summaries.items() if n not in REFERENCES},
            "selection": select(summaries, summaries["rule_based"]["entities_noninitial"], tiebreak,
                                entity_key="entities_noninitial")},
        "alternative_sentence_nli": {
            "rule": "Gates 1-2 on sentence-level aligned NLI; gate 4 on non-sentence-initial entities",
            "selection": select(summaries, summaries["rule_based"]["entities_noninitial"], tiebreak,
                                nli_key="sent_nli_mean", below_key="sent_nli_below_080",
                                entity_key="entities_noninitial"),
            "selection_other_sari_convention": select(
                summaries, summaries["rule_based"]["entities_noninitial"],
                sari if tiebreak is sari_hf else sari_hf, nli_key="sent_nli_mean",
                below_key="sent_nli_below_080", entity_key="entities_noninitial")},
        "pareto_fk_vs_nli": pareto({n: s for n, s in summaries.items()}),
        "pareto_fk_vs_sentence_nli": pareto(summaries, y="sent_nli_mean"),
        "rule_based_entities": rb,
        "versus_baseline": versus_baseline(passages, complete, summaries),
        "versus_bases": {b: versus_baseline(passages, complete, summaries, base=b) for b in args.base},
        "passage_ids": [p["id"] for p in passages],
        "per_passage": per,
        "versus_best": {"best": args.best, "rows": versus_best(passages, complete, args.best, per)}
        if args.best else None,
        "signatures": {str(s): dict(zip(("generation", "scoring"), state_signatures(s))) for s in args.state},
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
    f.add_argument("--teammate-nli", type=Path, help="local cross-encoder/nli-deberta-v3-small for "
                   "Chhoeng's phrase replacer and the word veto")
    f.add_argument("--reuse-state", type=Path, help="older state whose memos/metrics may be reused")
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
    p.add_argument("--base", action="append", default=[], help="extra base config IDs for the activity test")
    p.add_argument("--best", help="config ID that paired ablation deltas are measured from")
    p.add_argument("--tiebreak", choices=("evaluate", "xu"), default="evaluate",
                   help="SARI convention for the frozen tie-break (pre-registered: evaluate, 0/0=1)")
    args = ap.parse_args()
    {"freeze": freeze, "run": run, "grammar-memo": grammar_memo, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
