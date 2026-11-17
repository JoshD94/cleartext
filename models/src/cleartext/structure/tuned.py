"""Opt-in tunable acceptance for sentence-wise T5 splits (param-sweeps study).

With its defaults, TunedSplitter reproduces the guarded T5 stage:
evals.pipeline_benchmark.SentencewiseGenerator(guarded=True) followed by the
adapter's whole-passage structure_checks. Each knob loosens or tightens one
part of that contract:

- waive: generation/structure checks whose failure is ignored, at sentence and
  (unless passage_waive is given) passage level. HARD_CHECKS guard the frozen
  gate-3/4 facts; the study never waives them except to replay the t5_raw and
  t5_unchecked references (t5_raw: every check waived per sentence, none per passage).
- passage: "single" (whole-passage structure_checks, the default), "passage"
  (compound.passage_checks: relative sentence count, fragments compared as a
  multiset) or "off".
- min_words: sentences with fewer words are left unsplit.
- candidates: beams tried per sentence; the first that differs from the source
  and passes is used (1 = top beam only, the default).
- entailment/min_entailment: an independent NLI score (a, b) -> [0, 1] that each
  accepted split must reach. No language model writes text here.
"""

from .common import structure_checks
from .compound import passage_checks
from .seq2seq import generation_checks

HARD_CHECKS = ("numbers", "negation", "entities", "code_literals", "technical_phrases")
SOFT_CHECKS = (
    "modality", "content_retained", "no_new_content", "argument_roles", "operator_scope",
    "finite_predicates", "logical_connectives", "sentence_count", "complete_clauses",
    "discourse_markers", "uncertainty_scope", "predicate_tense", "complement_scope",
    "generation_terminated", "subject_groups", "relative_attachments",
)
PASSAGE_MODES = ("single", "passage", "off")


class GeneratorCandidates:
    """candidates(sentence) for a SentenceGenerator: its beams, top first."""

    def __init__(self, generator):
        self.generator = generator

    def candidates(self, sentence):
        result = self.generator.predict(sentence)
        if "candidates" in result:
            return result["candidates"]
        checks = result.get("raw_checks", {}).get("checks", {})
        return [{"raw": result["raw"], "terminated": checks.get("generation_terminated", True)}]


class TunedSplitter:
    backend = "seq2seq"
    supports_passages = True

    def __init__(self, generator, waive=(), passage="single", min_words=0, candidates=1,
                 entailment=None, min_entailment=None, check_cache=None, passage_waive=None):
        passage_waive = waive if passage_waive is None else passage_waive
        unknown = (set(waive) | set(passage_waive)) - set(HARD_CHECKS) - set(SOFT_CHECKS)
        if unknown:
            raise ValueError(f"Unknown checks: {sorted(unknown)}")
        if passage not in PASSAGE_MODES:
            raise ValueError(f"passage must be one of {PASSAGE_MODES}")
        if min_entailment is not None and entailment is None:
            raise ValueError("min_entailment needs an entailment scorer")
        if type(candidates) is not int or candidates < 1:
            raise ValueError("candidates must be a positive integer")
        self.generator, self.waive, self.passage = generator, frozenset(waive), passage
        self.passage_waive = frozenset(passage_waive)
        self.min_words, self.candidates = min_words, candidates
        self.entailment, self.min_entailment = entailment, min_entailment
        self.check_cache = check_cache if check_cache is not None else {}

    def _failed(self, source, raw, terminated):
        key = (source, raw, terminated)
        if key not in self.check_cache:
            self.check_cache[key] = generation_checks(source, raw, terminated=terminated)["failed"]
        return [k for k in self.check_cache[key] if k not in self.waive]

    def sentence(self, source):
        if self.min_words and len(source.split()) < self.min_words:
            return source
        beams = self.generator.candidates(source)[: self.candidates]
        for k, beam in enumerate(beams):
            raw = beam["raw"]
            # The top beam keeps the default contract (an unchanged top beam is
            # the answer); lower beams are only alternatives to an unsplit source.
            if k and raw == source:
                continue
            if self._failed(source, raw, beam["terminated"]):
                continue
            if (self.min_entailment is not None and raw != source
                    and self.entailment(source, raw) < self.min_entailment):
                continue
            return raw
        return source

    def predict(self, text):
        from ..features import nlp

        parts, offset = [], 0
        for sentence in nlp()(text).sents:
            start, end = sentence.start_char, sentence.end_char
            parts.extend([text[offset:start], self.sentence(text[start:end])])
            offset = end
        parts.append(text[offset:])
        return {"output": "".join(parts)}

    def validate(self, source, output):
        if self.passage == "off":
            return {"pass": True, "checks": {}, "failed": []}
        checks = (structure_checks if self.passage == "single" else passage_checks)(source, output)
        failed = [k for k in checks["failed"] if k not in self.passage_waive]
        return {**checks, "failed": failed, "pass": not failed}
