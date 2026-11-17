"""Parameter sweeps over the ensemble's stage settings (November 16, 2026).

Extends evals/ensemble_study.py from stage on/off levels to stage settings. A
config is an ensemble_study factor config plus a dict of stage parameters
(PARAMS; an unset parameter keeps today's behavior). Storage, scoring, gates,
Pareto and paired bootstrap deltas are ensemble_study's; this module adds the
parameterized generator, panel guards and the sweep report.

Subcommands:
  freeze   frozen plan (paths, hashes, runtime) for named panels in a state dir
  run      one bounded batch of a design on one panel (shard), checkpointed per cell
  report   aggregate-only JSON/Markdown for complete configs (raw text stays local)

Config IDs are "<ensemble_study ID>#<canonical JSON of set params>", or the
plain ensemble_study ID when no parameter is set. Stage caches are keyed by
everything a stage reads: T5 beams per (sentence, beam width), word decision
rows without the decision threshold or meaning floor (both reapplied per
config), finder outputs per (sources, settings), NLI scores per pair.

No panel may contain a locked test passage: freeze, run and report refuse
files named *LOCKED*, any passage whose split is "test" and any passage that
ensemble_study.locked_ids matches (test-set v3 test IDs or text hashes).
The one exception is the pre-registered locked-test run: `freeze
--unlock-test FINALISTS --dev-plan DEV` accepts a finalists design only if it
is committed and unmodified in Git, and only with generation and scoring
sources and artifacts identical to the dev plan's; `run` on that plan accepts
only that design.
"""

import argparse
import copy
import fcntl
import hashlib
import inspect
import json
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from . import ensemble_study as S
from .checkpoints import digest, write_json

ROOT = S.ROOT
# Every tunable this study sweeps; unset = the current behavior (the run's
# config.json or artifact value where noted). See PARAMETERS.md for ranges.
PARAMS = {
    # T5 splitter (cleartext.structure.tuned.TunedSplitter, seq2seq.SentenceGenerator)
    "t5.beams": "beam width (2)",
    "t5.candidates": "beams tried per sentence, first passing one used (1)",
    "t5.waive": "checks ignored per sentence and per passage (none)",
    "t5.passage": "passage check: single | passage | off (single)",
    "t5.min_words": "sentences shorter than this are not split (0)",
    "t5.min_entailment": "small-NLI floor on each accepted split (none)",
    # Compound splitter (structure.compound.DependencyCompoundSplitter)
    "cmp.threshold": "boundary tagger threshold (artifact 0.4)",
    "cmp.min_sentence_words": "minimum sentence words (0)",
    "cmp.min_clause_words": "minimum clause words (0)",
    # Authored phrase rules (rewrites.phrase_rewrite)
    "phr.min_gain": "minimum predicted complexity gain (0.02)",
    "phr.max_edits": "phrase edits per passage (1)",
    # Chhoeng's phrase finder (teammate_stages.PhraseFinderStage)
    "finder.sources": "phrase sources (from the phrases level)",
    "finder.model_threshold": "CRF span threshold (0.3)",
    "finder.min_entailment": "replacer NLI floor (0.575)",
    "finder.min_similarity": "replacer sentence similarity floor (0.80)",
    "finder.max_fluency_drop": "replacer fluency drop ceiling (0.966)",
    "finder.ppdb_min": "minimum SimplePPDB ppdb score (none)",
    # Passive converter (teammate_stages.PassiveStage)
    "passive.placeholder": "agentless subject: someone | people (auto picks)",
    # Word stage (EnsembleClearText)
    "w.threshold": "decision acceptance threshold (0.45)",
    "w.detector": "CWI detector threshold (0.39)",
    "w.noun": "noun hardness threshold (0.8)",
    "w.floor": "meaning floor on sense fit (0.1)",
    "w.max_edits": "word edits per passage (1)",
    "w.alpha": "hardness exponent in cross-word ranking (0.5)",
    "w.min_gain": "minimum difficulty gain of a candidate (none)",
    "w.zipf_margin": "candidate zipf frequency minus target's, minimum (none)",
    "w.veto": "entailment veto threshold with words=contextual_veto (0.575)",
    "w.veto_refill": "a vetoed slot goes to the next-ranked word (true)",
}
# EnsembleClearText attribute each word/phrase parameter sets.
PIPE_ATTRS = {"w.threshold": "decision_threshold", "w.noun": "noun_threshold",
              "w.floor": "meaning_floor", "w.max_edits": "max_word_edits",
              "w.alpha": "hardness_alpha", "w.min_gain": "word_min_gain",
              "w.zipf_margin": "word_zipf_margin", "w.veto_refill": "veto_refill",
              "phr.min_gain": "phrase_min_gain", "phr.max_edits": "max_phrase_edits"}
FINDER_PARAMS = {"finder.model_threshold": "model_threshold", "finder.min_entailment": "min_entailment",
                 "finder.min_similarity": "min_similarity", "finder.max_fluency_drop": "max_fluency_drop",
                 "finder.ppdb_min": "ppdb_min"}
T5_LEVELS = ("t5", "t5_raw", "t5_unchecked", "t5+compound_tagger")
GENERATION_VERSION = 1


# --- config IDs ---------------------------------------------------------------

def _canonical(params):
    out = {}
    for key, value in params.items():
        if key not in PARAMS:
            raise ValueError(f"Unknown parameter {key}")
        if isinstance(value, (list, tuple)):
            value = sorted(value)
        out[key] = value
    return out


def make_id(base, params=None):
    """ID for a factor config plus parameters (unset parameters omitted)."""
    params = _canonical(params or {})
    check(S.normalize(base), params)
    head = S.config_id(base)
    return head + ("#" + json.dumps(params, sort_keys=True, separators=(",", ":")) if params else "")


def parse(name):
    """(factor config, params) for an ID; references return (name, {})."""
    if name in S.REFERENCES:
        return name, {}
    head, _, tail = name.partition("#")
    params = json.loads(tail) if tail else {}
    config = S.parse_id(head)
    check(config, params)
    return config, params


def check(config, params):
    """Reject parameters that the config's stages would silently ignore."""
    structure, words, phrases = config["structure"], config["words"], config["phrases"]
    t5 = [k for k in params if k.startswith("t5.")]
    if t5 and structure not in T5_LEVELS:
        raise ValueError(f"{t5} need a T5 structure level")
    if any(k.startswith("cmp.") for k in params) and "compound" not in structure:
        raise ValueError("cmp.* needs a compound structure level")
    if any(k.startswith("finder.") for k in params) and phrases.split("+")[-1] not in S.FINDER_SOURCES:
        raise ValueError("finder.* needs a phrase-finder level")
    if any(k.startswith("w.") for k in params) and words == "off":
        raise ValueError("w.* needs a word level")
    if ("w.veto" in params or "w.veto_refill" in params) and words != "contextual_veto":
        raise ValueError("w.veto* needs words=contextual_veto")
    if "phr.min_gain" in params or "phr.max_edits" in params:
        if not phrases.startswith("authored"):
            raise ValueError("phr.* needs authored phrases")
    if "passive.placeholder" in params and (config.get("passive") != "on"
                                           or params["passive.placeholder"] not in ("someone", "people")):
        raise ValueError("passive.placeholder needs passive=on and someone|people")
    if params.get("t5.candidates", 1) > params.get("t5.beams", 2):
        raise ValueError("t5.candidates cannot exceed t5.beams")
    if params.get("t5.passage", "single") not in ("single", "passage", "off"):
        raise ValueError("t5.passage must be single, passage or off")


def design_names(path):
    """IDs of a design file: {"configs": [ID | {"base": ..., "params": ...}, ...]}."""
    rows = json.loads(Path(path).read_text())
    configs = rows["configs"] if isinstance(rows, dict) else rows
    names = [c if isinstance(c, str) else make_id(c["base"], c.get("params")) for c in configs]
    for name in names:
        parse(name)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate configs in design")
    return names


def design_groups(path):
    """{ID: (group, value)} for configs that carry a sweep group label."""
    rows = json.loads(Path(path).read_text())
    out = {}
    for c in rows.get("configs", []) if isinstance(rows, dict) else []:
        if isinstance(c, dict) and c.get("group"):
            out[make_id(c["base"], c.get("params"))] = (c["group"], c.get("value"))
    return out


# --- frozen plan --------------------------------------------------------------

def guard_rows(rows, label):
    # ensemble_study.locked_ids matches test IDs and test text hashes and fails
    # closed without the committed lists. This harness has no unlock option.
    bad = [r["id"] for r in rows if r.get("split") == "test"] + S.locked_ids(rows)
    if bad:
        raise SystemExit(f"Refusing {label}: {len(bad)} locked test passages (e.g. {bad[0]})")


def guard_panel(path, unlocked=False):
    path = Path(path)
    if unlocked:
        return json.loads(path.read_text())
    if "locked" in path.name.lower():
        raise SystemExit(f"Refusing locked panel {path}")
    rows = json.loads(path.read_text())
    guard_rows(rows, path)
    if len({r["id"] for r in rows}) != len(rows) or any(not r.get("original", "").strip() for r in rows):
        raise SystemExit(f"{path}: IDs must be unique and every passage needs text")
    return rows


def freeze(args):
    state = args.state.resolve()
    if state == ROOT or ROOT in state.parents:
        raise SystemExit("State must be outside the checkout; raw text stays local.")
    unlock = committed_finalists(args.unlock_test, args.dev_plan) if args.unlock_test else None
    panels = {}
    for item in args.panel:
        name, _, path = item.partition("=")
        guard_panel(path, unlocked=unlock is not None)
        panels[name] = Path(path).resolve()
    hub = {"bert": args.encoder, "similarity": args.embedding, "nli": args.teammate_nli}
    generation = {
        "runs": {"v7": str(ROOT / "models/runs/ensemble-v7-20260928"),
                 "contextual": str(ROOT / "models/runs/ensemble-context-similarity-20261002")},
        "t5": str(args.t5.resolve()), "tagger": str(args.tagger.resolve()),
        "encoder": str(args.encoder.resolve()), "nltk_data": str(args.nltk_data.resolve()),
        "teammate_models": {k: str(Path(v).resolve()) for k, v in hub.items()},
        "teammate_packages": {"python-crfsuite": S.version("python-crfsuite")},
    }
    artifacts = []
    for run in generation["runs"].values():
        artifacts += S._run_dependencies(run)
    for key in ("t5", "tagger", "encoder"):
        artifacts += S._files(generation[key])
    for path in generation["teammate_models"].values():
        artifacts += S._files(path)
    eval_artifacts = S._files(args.embedding) + S._files(args.nli)
    plan = {
        "created": datetime.now(timezone.utc).isoformat(),
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip(),
        "panels": {k: str(v) for k, v in panels.items()},
        "panel_sha256": {k: S.sha(v) for k, v in panels.items()},
        "generation": generation,
        "evaluation": {"embedding": str(args.embedding.resolve()), "nli": str(args.nli.resolve())},
        "generation_artifacts_sha256": {str(p): S.sha(p) for p in sorted(set(artifacts))},
        "evaluation_artifacts_sha256": {str(p): S.sha(p) for p in sorted(set(eval_artifacts))},
        "generation_sources_sha256": {str(p): S.sha(p) for p in S.generation_sources()},
        "scoring_sources_sha256": {str(p): S.sha(p) for p in S.scoring_sources()},
        "runtime": S.runtime(),
    }
    if args.reuse_metrics:
        # Metrics depend only on (source, output) and the scoring signature.
        old = S.state_signatures(args.reuse_metrics.resolve())[1]
        if old != signatures(plan)[1]:
            raise SystemExit("Scoring signature differs; metrics cannot be reused")
        plan["reuse_metrics"] = {"state": str(args.reuse_metrics.resolve()), "scoring_signature": old}
    if unlock:
        dev = json.loads(Path(args.dev_plan).read_text())
        keys = ("generation", "evaluation", "generation_artifacts_sha256", "evaluation_artifacts_sha256",
                "generation_sources_sha256", "scoring_sources_sha256", "runtime")
        changed = [k for k in keys if plan[k] != dev[k]]
        if changed:
            raise SystemExit(f"Locked-test plan differs from the dev plan in {changed}; not unlocking")
        if signatures(plan) != signatures(dev):
            raise SystemExit("Locked-test signatures differ from the dev plan's; not unlocking")
        plan["unlock_test"] = True
        plan["finalists"] = unlock
    state.mkdir(parents=True, exist_ok=True)
    path = state / "plan.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a frozen plan is never overwritten")
    write_json(path, plan)
    print(f"Froze {path}: panels {sorted(panels)}")


def signatures(plan):
    gen = digest({"generation": plan["generation"], "artifacts": plan["generation_artifacts_sha256"],
                  "sources": S._relative(plan["generation_sources_sha256"]),
                  "harness": hashlib.sha256((inspect.getsource(S.StudyGenerator)
                                             + inspect.getsource(SweepGenerator)).encode()).hexdigest(),
                  "params": sorted(PARAMS), "version": GENERATION_VERSION, "runtime": plan["runtime"]})
    # Same formula as ensemble_study, so identical scoring setups share metrics.
    score = S.signatures(plan)[1]
    return gen, score


class Store(S.StudyStore):
    """StudyStore whose metric lookup falls back through several older states."""

    def __init__(self, root, generation_signature, scoring_signature, fallbacks=()):
        super().__init__(root, generation_signature, scoring_signature)
        self.fallbacks = [Path(f) for f in fallbacks]

    def metrics(self, source, output):
        for root in [None, *self.fallbacks]:
            path = self._metric_path(source, output, root)
            if path.exists():
                value = json.loads(path.read_text())
                if value.get("pair") != {"source": source, "output": output} or value.get("signature") != self.score:
                    raise ValueError("Metric cache identity changed")
                return value
        return None


def metric_fallbacks(plan):
    """plan["reuse_metrics"]'s state, then the states it reused from (their
    plan["reuse"]), while the scoring signature stays identical."""
    reuse = plan.get("reuse_metrics")
    if not reuse:
        return []
    found, state = [], Path(reuse["state"])
    while state and S.state_signatures(state)[1] == reuse["scoring_signature"]:
        found.append(state)
        older = json.loads((state / "plan.json").read_text()).get("reuse") or {}
        state = Path(older["state"]) if older.get("metrics") else None
    return found


def committed_finalists(path, dev_plan):
    """Identity of a finalists design that Git tracks with no uncommitted change."""
    if dev_plan is None:
        raise SystemExit("--unlock-test needs --dev-plan")
    path = Path(path).resolve()
    rel = str(path.relative_to(ROOT))
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    if git("ls-files", "--error-unmatch", rel).returncode or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        raise SystemExit(f"{rel} must be committed and unmodified before the test split is unlocked")
    return {"path": rel, "sha256": S.sha(path), "commit": git("log", "-1", "--format=%H", "--", rel).stdout.strip(),
            "configs": design_names(path)}


def preflight(plan, check_artifacts=False, design=None):
    for name, path in plan["panels"].items():
        guard_panel(path, unlocked=plan.get("unlock_test", False))
    if plan.get("unlock_test"):
        finalists = plan["finalists"]
        if design is not None and (S.sha(design) != finalists["sha256"]
                                   or design_names(design) != finalists["configs"]):
            raise SystemExit("An unlocked plan runs only its committed finalists design")
    S.preflight(plan, check_artifacts)


# --- generation ---------------------------------------------------------------

class SweepGenerator(S.StudyGenerator):
    """StudyGenerator whose stages take PARAMS settings."""

    def __init__(self, plan, state, signature, threads=1):
        super().__init__(plan, state, signature, threads)
        self._beams, self._splitters, self._compounds = {}, {}, {}
        self._small_nli, self._checks = None, {}

    def pipe(self, level):
        """Load a word run once; decision rows are cached without the threshold
        and meaning floor, which are reapplied for each config's settings."""
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
                self._limit_threads()
                return result

            model._study_wrapped = True
            contextual.model = model
        self._limit_threads()
        current = {}
        rank_word, decide = pipe.rank_word, pipe.decide

        def cached_rank(doc, token, original_doc=None, guard=True):
            context = original_doc if original_doc is not None else doc
            current["key"] = (doc.text, token.idx, context.text)
            try:
                return rank_word(doc, token, original_doc, guard)
            finally:
                current.pop("key", None)

        def cached_decide(doc, token, slot, candidates):
            key = current.get("key")
            threshold, floor = pipe.decision_threshold, pipe.meaning_floor
            cache = self.decisions.setdefault(level, {})
            if key is None or key not in cache:
                pipe.decision_threshold, pipe.meaning_floor = float("-inf"), 0.0
                try:
                    rows = decide(doc, token, slot, candidates)
                finally:
                    pipe.decision_threshold, pipe.meaning_floor = threshold, floor
                entry = (copy.deepcopy(rows), copy.deepcopy(slot.cache.get("senses")))
                if key is not None:
                    cache[key] = entry
                    self.dirty.add(level)
            else:
                entry = cache[key]
            rows, senses = entry
            if senses is not None:
                slot.cache["senses"] = copy.deepcopy(senses)
            rows = copy.deepcopy(rows)
            # Same rejections, in the same order, as EnsembleClearText.decide.
            for r in rows:
                if r["accept"] < threshold:
                    r["rejections"].insert(0, "low acceptance probability")
                if floor and r["members"].get("sense_fit", 1.0) < floor:
                    r["rejections"].append("meaning too unlikely here")
            return rows

        pipe.rank_word, pipe.decide = cached_rank, cached_decide
        pipe._defaults = {attr: getattr(pipe, attr) for attr in PIPE_ATTRS.values()}
        pipe._defaults["detector"] = pipe.detector
        self.pipes[level] = pipe
        return pipe

    def beams(self, width):
        """{"candidates": [...]} per sentence for one beam width, memoized on disk."""
        if width not in self._beams:
            memo = S.DiskMemo(self.state, f"t5-beams{width}", self.signature)
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
        """Chhoeng's min(forward, backward) entailment with the small cross-encoder."""
        memo = S.DiskMemo(self.state, "small-nli", self.signature)
        key = [premise, hypothesis]
        if key not in memo:
            if self._small_nli is None:
                from cleartext.teammate_stages import EntailmentVeto
                self._small_nli = EntailmentVeto(model=self.plan["generation"]["teammate_models"]["nli"])
                self._limit_threads()
            memo[key] = self._small_nli.score(premise, hypothesis)
        return memo[key]

    def t5_splitter(self, level, params):
        from cleartext.structure.tuned import HARD_CHECKS, SOFT_CHECKS, TunedSplitter
        every = (*HARD_CHECKS, *SOFT_CHECKS)
        base = {"t5": ((), None, "single"), "t5_raw": (every, (), "single"),
                "t5_unchecked": (every, (), "off")}[level]
        waive = tuple(sorted(set(base[0]) | set(params.get("t5.waive", ()))))
        passage_waive = base[1] if base[1] is not None and "t5.waive" not in params else None
        key = (level, json.dumps({k: v for k, v in params.items() if k.startswith("t5.")}, sort_keys=True))
        if key not in self._splitters:
            width = params.get("t5.beams", 2)
            entail = self.small_entailment if params.get("t5.min_entailment") is not None else None
            self._splitters[key] = TunedSplitter(
                self.beams(width), waive=waive, passage=params.get("t5.passage", base[2]),
                min_words=params.get("t5.min_words", 0), candidates=params.get("t5.candidates", 1),
                entailment=entail, min_entailment=params.get("t5.min_entailment"),
                check_cache=self._checks, passage_waive=passage_waive)
        return self._splitters[key]

    def compound(self, params):
        key = tuple(params.get(k) for k in ("cmp.threshold", "cmp.min_sentence_words", "cmp.min_clause_words"))
        if key not in self._compounds:
            from cleartext.structure.compound import DependencyCompoundSplitter
            self._compounds[key] = DependencyCompoundSplitter(
                min_sentence_words=params.get("cmp.min_sentence_words", 0),
                min_clause_words=params.get("cmp.min_clause_words", 0), guards="scoped",
                tagger=self.plan["generation"]["tagger"], tagger_threshold=params.get("cmp.threshold"))
        return self._compounds[key]

    def tuned_structure(self, level, params):
        if level == "t5+compound_tagger":
            return [self.t5_splitter("t5", params), self.compound(params)]
        if level in T5_LEVELS:
            return self.t5_splitter(level, params)
        if level == "compound_tagger":
            return self.compound(params)
        return None

    def tuned_passive(self, level, params):
        if level != "on" or "passive.placeholder" not in params:
            return self.passive(level)
        key = "placeholder-" + params["passive.placeholder"]
        if key not in self._passive:
            from cleartext.teammate_stages import PassiveStage
            self._passive[key] = PassiveStage(placeholder=params["passive.placeholder"])
        return self._passive[key]

    def tuned_finder(self, level, params):
        name = level.split("+")[-1]
        if name not in S.FINDER_SOURCES:
            return None
        sources = tuple(params.get("finder.sources") or S.FINDER_SOURCES[name])
        settings = {FINDER_PARAMS[k]: v for k, v in params.items() if k in FINDER_PARAMS}
        key = json.dumps([sources, settings], sort_keys=True)
        if key not in self._finders:
            memo = S.DiskMemo(self.state, "finder-" + digest(key)[:16], self.signature)
            study, stage = self, []

            class Remembered:
                def rewrite(_, text):
                    if text not in memo:
                        if not stage:
                            from cleartext.teammate_stages import PhraseFinderStage
                            stage.append(PhraseFinderStage(sources, study.plan["generation"]["teammate_models"],
                                                           settings=settings))
                            study._limit_threads()
                        output, edits = stage[0].rewrite(text)
                        memo[text] = {"output": output, "edits": edits}
                    value = memo[text]
                    return value["output"], copy.deepcopy(value["edits"])

            self._finders[key] = Remembered()
        return self._finders[key]

    def tuned_veto(self, level, params):
        if level != "contextual_veto":
            return None
        threshold = params.get("w.veto")
        if threshold is None:
            from cleartext.teammate_stages import EntailmentVeto
            if self._veto is None:
                self._veto = EntailmentVeto(model=self.plan["generation"]["teammate_models"]["nli"])
            threshold = self._veto.threshold
        return lambda sentence, edited: self.small_entailment(sentence, edited) < threshold

    def predict(self, name, text):
        if name in S.REFERENCES:
            return super().predict(name, text)
        config, params = parse(name)
        pipe = self.pipe(config["words"])
        pipe.technical_terms = config["terms"] == "on"
        for key, attr in PIPE_ATTRS.items():
            setattr(pipe, attr, params.get(key, pipe._defaults[attr]))
        pipe.detector = pipe._defaults["detector"]
        if "w.detector" in params:
            pipe.detector = {**pipe.detector, "threshold": params["w.detector"]}
        try:
            result = pipe.analyze(
                text,
                structure=config["structure"] != "off",
                phrases=config["phrases"] in ("authored", "authored+simpleppdb", "authored+crf"),
                words=config["words"] != "off",
                structure_rewriter=self.tuned_structure(config["structure"], params),
                grammar_corrector=self.grammar() if config["grammar"] == "on" else None,
                guardrails=True,
                negation=config["negation"] == "on",
                passive_converter=self.tuned_passive(config["passive"], params),
                phrase_finder=self.tuned_finder(config["phrases"], params),
                word_veto=self.tuned_veto(config["words"], params),
            )
        finally:
            pipe.technical_terms = False
            for attr, value in pipe._defaults.items():
                setattr(pipe, attr, value)
        stages = [e.get("stage", "structure" if "rule" in e else "unknown") for e in result["edits"]]
        return {"output": result["output"], "stages": stages, "edits": result["edits"],
                "blocked": {k: result[k] for k in ("term_preservation", "literal_preservation") if k in result}}


# --- batches ------------------------------------------------------------------

def run(args):
    plan = json.loads(args.plan.read_text())
    preflight(plan, check_artifacts=not args.preflight, design=args.design)
    if args.preflight:
        print("Preflight passed")
        return
    gen_sig, score_sig = signatures(plan)
    names = [*S.REFERENCES, *[n for n in design_names(args.design) if n not in S.REFERENCES]]
    passages = S.load_panel(plan, args.panel)
    shard, shards = map(int, args.shard.split("/"))
    selected = [p for k, p in enumerate(passages) if k % shards == shard]
    state = args.state
    store = Store(state, gen_sig, score_sig, metric_fallbacks(plan))
    import torch
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        pass
    started, deadline = time.monotonic(), time.monotonic() + args.max_seconds
    (state / "locks").mkdir(parents=True, exist_ok=True)
    counts = {"outputs": 0, "scored": 0}
    with (state / "locks" / f"{args.panel}-{shard}-of-{shards}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generator, scorer = None, None

        def log(message):
            print(f"{time.strftime('%H:%M:%S')} +{time.monotonic() - started:.0f}s {message}", flush=True)

        done = True
        # Generate every config first (cheap stages, shared caches), then score.
        for passage in selected:
            missing = [n for n in names if store.output(passage, n) is None]
            if not missing:
                continue
            generator = generator or SweepGenerator(plan, state, gen_sig, args.threads)
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
                        log(f"output {passage['id']} {name} {result['seconds']:.1f}s")
            finally:
                generator.end(passage)
            if not done:
                break
        if done and args.score:
            for passage in selected:
                pairs = [store.output(passage, n)["output"] for n in names]
                pairs = [o for o in dict.fromkeys(pairs) if store.metrics(passage["original"], o) is None]
                for output in pairs:
                    if time.monotonic() >= deadline:
                        done = False
                        break
                    scorer = scorer or S.StudyScorer(plan, state, score_sig, args.threads)
                    store.save_metrics(passage["original"], output, scorer(passage["original"], output))
                    counts["scored"] += 1
                if not done:
                    break
            log(f"scored {counts['scored']} pairs")
        complete = all(store.output(p, n) is not None and (not args.score or store.metrics(
            p["original"], store.output(p, n)["output"]) is not None) for p in selected for n in names)
    preflight(plan, check_artifacts=True)
    receipt = {"host": socket.gethostname(), "panel": args.panel, "shard": args.shard,
               "design": str(args.design), "design_sha256": S.sha(args.design), "configs": len(names),
               "seconds": time.monotonic() - started, "generation_signature": gen_sig,
               "scoring_signature": score_sig, "shard_complete": complete, **counts,
               "frozen_inputs_rechecked": True}
    write_json(state / "batches" / f"{time.time_ns()}.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


# --- report -------------------------------------------------------------------

def collect(states, plan, panel, names, need_metrics=True):
    """ensemble_study.collect, also reading metrics from plan["reuse_metrics"]."""
    fallbacks = metric_fallbacks(plan)
    passages = S.load_panel(plan, panel)
    stores = [Store(state, *S.state_signatures(state), fallbacks) for state in states]
    complete, pending = {}, []
    for name in names:
        outputs, metrics = {}, {}
        for p in passages:
            found = [f for f in (s.output(p, name) for s in stores) if f is not None]
            if not found:
                break
            if len({f["output"] for f in found}) != 1:
                raise ValueError(f"Conflicting outputs across states: {p['id']} / {name}")
            outputs[p["id"]] = found[0]
            if need_metrics:
                m = next((m for m in (s.metrics(p["original"], found[0]["output"]) for s in stores)
                          if m is not None), None)
                if m is None:
                    break
                metrics[p["id"]] = m
        else:
            complete[name] = (outputs, metrics)
            continue
        pending.append(name)
    return passages, complete, pending


def config_rows(passages, complete):
    """{name: [testset_v3.passage_row per passage]}: Week 7 metrics plus Kea-Roy's S,
    the fixed and capitalized entity checks and the changed-token rate."""
    from . import testset_v3 as T
    for p in passages:
        p.setdefault("kind", "standard")
        p.setdefault("domain", "all")
    return {n: [T.passage_row(p, outputs[p["id"]]["output"], metrics[p["id"]]) for p in passages]
            for n, (outputs, metrics) in complete.items()}


def select(summaries, rule_based_entities, gate_rows=None):
    """Frozen gates; lowest FK; ties within 0.1 grade broken by the higher Kea-Roy S."""
    from . import testset_v3 as T
    gate_rows = gate_rows or {n: T.gates(s, rule_based_entities) for n, s in summaries.items()}
    passing = {n: s for n, s in summaries.items() if n not in S.REFERENCES and all(gate_rows[n].values())}
    if not passing:
        return {"winner": None, "passing": [], "tied": []}
    best = min(s["fk_grade"] for s in passing.values())
    tied = sorted(n for n, s in passing.items() if s["fk_grade"] <= best + 0.1)
    winner = max(tied, key=lambda n: (passing[n]["kea_S"], -passing[n]["fk_grade"]))
    return {"winner": winner, "passing": sorted(passing, key=lambda n: passing[n]["fk_grade"]),
            "tied": tied}


def adoption(row, gates_pass, over_edit_pass):
    """Pre-registered adoption test against the incumbent (SWEEP_PROTOCOL.md, amendment 1)."""
    fk, nli = row["fk_grade"], row["nli_mean"]
    out = {"gates": gates_pass, "over_edit": over_edit_pass, "fk_beyond_tie": fk["mean"] <= -0.1,
           "fk_ci_below_zero": fk["ci95"][1] < 0, "nli_mean_delta_ge_minus_001": nli["mean"] >= -0.01}
    out["adopt"] = all(out.values())
    return out


def report(args):
    from . import testset_v3 as T
    from .readability import flesch_kincaid_grade
    from .sari import corpus_sari
    plan = json.loads((args.state[0] / "plan.json").read_text())
    names, groups = [*S.REFERENCES], {}
    for design in args.design:
        names += [n for n in design_names(design) if n not in names]
        # A config keeps the label of the first design that names it (phase A before B).
        groups = {**design_groups(design), **groups}
    passages, complete, pending = collect(args.state, plan, args.panel, names)
    if not plan.get("unlock_test"):
        guard_rows(passages, args.panel)
    rows = config_rows(passages, complete)
    summaries = {n: {**T.summarize(r), "sentence_length": S._mean(
        complete[n][1][p["id"]]["readability"]["avg_sentence_length"] for p in passages)} for n, r in rows.items()}
    rb, rb_cap = summaries["rule_based"]["entities"], summaries["rule_based"]["entities_capitalized"]
    gate_rows = {n: T.gates(s, rb) for n, s in summaries.items() if n not in S.REFERENCES}
    # Labeled alternative: gate 4 with the pre-November 16 capitalized-span entity check.
    old_gates = {n: T.gates({**s, "entities": s["entities_capitalized"]}, rb_cap)
                 for n, s in summaries.items() if n not in S.REFERENCES}
    controls = {n: T.summarize([x for x in r if x["kind"] == "simple_control"]) for n, r in rows.items()
                if any(x["kind"] == "simple_control" for x in r)}
    over_edit = {n: T.over_edit_gate(c) for n, c in controls.items()}
    inc = args.incumbent
    versus = {}
    if inc in rows:
        for n, r in rows.items():
            differ = sum(a["changed"] != b["changed"] or complete[n][0][p["id"]]["output"] != complete[inc][0][p["id"]]["output"]
                         for a, b, p in zip(r, rows[inc], passages))
            v = {k: T.paired(r, rows[inc], k) for k in ("fk_grade", "kea_S", "nli_mean")}
            v["outputs_differ"] = differ
            if n not in S.REFERENCES:
                v["adoption"] = adoption(v, all(gate_rows[n].values()),
                                         all(over_edit.get(n, {"none": True}).values()))
            versus[n] = v
    domains = sorted({p["domain"] for p in passages})
    by_domain = {d: {n: {k: v for k, v in T.summarize([x for x in r if x["domain"] == d]).items()
                         if k in ("n", "changed", "fk_grade", "kea_S", "nli_mean", "nli_below_080")}
                     for n, r in rows.items()} for d in domains}
    sari = {}
    for panel in [k for k in plan["panels"] if k in ("asset_test", "onestop_dev")]:
        refs_panel, done, _ = collect(args.state, plan, panel, names, need_metrics=False)
        sari[panel] = {}
        for name, (outputs, _) in done.items():
            preds = [outputs[p["id"]]["output"] for p in refs_panel]
            data = ([p["original"] for p in refs_panel], preds, [p["references"] for p in refs_panel])
            sari[panel][name] = {"xu": corpus_sari(*data, zero_division=0),
                                 "evaluate": corpus_sari(*data, zero_division=1),
                                 "fk_grade": S._mean(flesch_kincaid_grade(t) for t in preds), "n": len(preds)}
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "panel": {"name": args.panel, "n": len(passages), "sha256": plan["panel_sha256"][args.panel]},
        "configs": summaries, "pending": pending, "groups": {n: list(g) for n, g in groups.items()},
        "gates": gate_rows, "rule_based_entities": rb,
        "selection": select(summaries, rb, gate_rows),
        "selection_note": "Frozen gates (fixed entity check, as test-set v3), lowest FK, 0.1-grade ties broken "
                          "by Kea-Roy's S (SWEEP_PROTOCOL.md amendment 1)",
        "alternative_capitalized_entities": {"rule_based_entities": rb_cap, "gates": old_gates,
                                             "selection": select(summaries, rb_cap, old_gates)},
        "simple_controls": controls, "over_edit_gate": over_edit,
        "pareto_fk_vs_nli": S.pareto(summaries),
        "incumbent": inc, "versus_incumbent": versus, "by_domain": by_domain, "sari": sari,
        "passage_ids": [p["id"] for p in passages],
        "per_passage": {n: {k: [x[k] for x in r] for k in ("fk_grade", "kea_S", "nli_mean", "entities")}
                        for n, r in rows.items()},
        "signatures": {str(s): dict(zip(("generation", "scoring"), S.state_signatures(s))) for s in args.state},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f"aggregate-{args.panel}.json", result)
    (args.out / f"results-{args.panel}.md").write_text(markdown(result))
    print(f"{len(summaries)} complete configs, {len(pending)} pending -> {args.out}")


def short(name, group=None):
    """Readable label: factor levels that differ from the incumbent, then parameters."""
    if name in S.REFERENCES:
        return name
    if group and group[0] not in ("levels", "combined"):
        value = "none" if group[1] is None else group[1]
        return f"{group[0]} = {value}"
    config, params = parse(name)
    incumbent = S.normalize({"words": "off", "phrases": "authored", "structure": "t5+compound_tagger"})
    parts = [f"{k}={v}" for k, v in config.items() if v != incumbent[k]]
    for k, v in sorted(params.items()):
        parts.append(f"{k}={'all soft' if k == 't5.waive' and len(v) > 8 else v}")
    return ", ".join(parts) or "incumbent"


def _ci(v, digits):
    return f"{v['mean']:+.{digits}f} [{v['ci95'][0]:+.{digits}f}, {v['ci95'][1]:+.{digits}f}]"


def markdown(result):
    head = ("| Config | Changed | FK ↓ | ΔFK vs incumbent [95% CI] | Kea S ↑ | ΔS | NLI passage | ΔNLI [95% CI] | "
            "NLI<0.8 | Entities (fixed / capitalized) | Num/neg fails | Controls changed-token rate | Gates |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    lines, front = [], set(result["pareto_fk_vs_nli"])
    versus, controls = result["versus_incumbent"], result["simple_controls"]
    for name, s in sorted(result["configs"].items(), key=lambda kv: kv[1]["fk_grade"]):
        g = result["gates"].get(name)
        gate = "ref" if g is None else ("pass" if all(g.values()) else
                                        "fail " + ",".join(str(k + 1) for k, v in enumerate(g.values()) if not v))
        if g is not None and not all(result["over_edit_gate"].get(name, {"none": True}).values()):
            gate += " (over-edit fail)"
        v = versus.get(name)
        label = short(name, result["groups"].get(name)) + (" ◆" if name in front else "")
        rate = controls.get(name, {}).get("changed_token_rate")
        lines.append(
            f"| `{label}` | {s['changed']}/{s['n']} | {s['fk_grade']:.3f} | {_ci(v['fk_grade'], 3) if v else ''} | "
            f"{s['kea_S']:.4f} | {_ci(v['kea_S'], 4) if v else ''} | {s['nli_mean']:.4f} | "
            f"{_ci(v['nli_mean'], 4) if v else ''} | {s['nli_below_080']} | "
            f"{s['entities']:.4f} / {s['entities_capitalized']:.4f} | {s['numbers_failed']}/{s['negations_failed']} | "
            f"{'' if rate is None else f'{rate:.3f}'} | {gate} |")
    return head + "\n".join(lines) + "\n\n◆ = on the FK/passage-NLI Pareto frontier.\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--state", type=Path, required=True)
    f.add_argument("--panel", action="append", required=True, help="NAME=PATH (repeatable)")
    for name in ("t5", "tagger", "encoder", "nltk-data", "embedding", "nli", "teammate-nli"):
        f.add_argument("--" + name, type=Path, required=True)
    f.add_argument("--reuse-metrics", type=Path, help="older state with an identical scoring signature")
    f.add_argument("--unlock-test", type=Path, metavar="FINALISTS",
                   help="locked-test run only: the committed finalists design")
    f.add_argument("--dev-plan", type=Path, help="with --unlock-test: the dev plan whose sources must match")
    r = sub.add_parser("run")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--state", type=Path, required=True)
    r.add_argument("--design", type=Path, required=True)
    r.add_argument("--panel", required=True)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--max-seconds", type=int, default=1500)
    r.add_argument("--no-score", dest="score", action="store_false")
    r.add_argument("--preflight", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--state", type=Path, action="append", required=True)
    p.add_argument("--design", type=Path, action="append", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--incumbent", help="config ID that paired deltas and adoption are measured from")
    args = ap.parse_args()
    {"freeze": freeze, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
