"""Explicitly loaded BERT fallback prototype. Existing default loaders are unchanged."""

import json
import pickle

from . import ensemble as E
from .bert_candidates import MaskedBertGenerator
from .building_blocks import BuildingBlockClearText
from .candidate_diagnostics import review_shortlist
from .candidate_validation import validate_proposal
from .data import ROOT
from .ensemble_pipeline import after_article, mass_noun, substitute
from .grammar_validation import GrammarValidator
from .lexical import guardrails
from .novel_context import (
    FEATURE_NAMES,
    choose_novel,
    context_features,
    validate_feature_names,
)
from .proposal_evidence import wordnet_evidence
from .technical_terms import detect_terms


class NovelCandidateClearText(BuildingBlockClearText):
    @classmethod
    def load(cls, run, allow_shadow=False):
        config = json.loads((run / "config.json").read_text())
        if config.get("shadow_only") and not allow_shadow:
            raise ValueError("shadow prototype requires explicit allow_shadow=True")
        names = validate_feature_names(config["feature_names"])
        validator = pickle.load((run / "validator.pkl").open("rb"))
        if validator.n_features_in_ != len(names):
            raise ValueError("validator width does not match configured features")
        pipe = super().load(ROOT / config["base_run"])
        pipe.novel_config = config
        pipe.novel_validator = validator
        pipe.novel_generator = MaskedBertGenerator(top_k=20)
        return pipe

    def analyze(self, text, **kwargs):
        # Preserve the sentence-level choice, including edits at other targets.
        # The fallback is considered only when the entire base pass abstains.
        previous = getattr(self, "_wordnet_only", False)
        self._wordnet_only = True
        try:
            base = super().analyze(text, **kwargs)
        finally:
            self._wordnet_only = previous
        if (
            base["edits"]
            or base["output"] != text
            or previous
            or self.novel_config["context_threshold"] is None
        ):
            return base
        return super().analyze(text, **kwargs)

    def rank_word(self, doc, token, original_doc=None, guard=True):
        result = super().rank_word(doc, token, original_doc, guard)
        if result["selected"] is not None:
            result["novel_status"] = "existing_wordnet_edit_retained"
            return result
        if getattr(self, "_wordnet_only", False):
            result["novel_status"] = "base_pass_abstained"
            return result
        threshold = self.novel_config["context_threshold"]
        if threshold is None:
            result["novel_status"] = "disabled_insufficient_context_validation"
            return result
        target, candidates = self.novel_generator.generate(token)
        existing = {row["word"].casefold() for row in result["candidates"]}
        # Cap the raw novel shortlist before validation, matching the paired cache.
        # Rejected proposals do not refill the shortlist from lower BERT ranks.
        candidates = review_shortlist(
            [row for row in candidates if row["word"].casefold() not in existing],
            limit=self.novel_config["candidate_limit"],
        )
        candidates = [
            row
            for row in candidates
            if validate_proposal(row, target[1])["dictionary_valid"]
        ]
        if not target or not candidates:
            result["novel_status"] = "no_dictionary_valid_novel_proposals"
            return result
        slot = E.Slot(doc, token, target=target)
        features = context_features(
            slot, candidates, self.novel_config.get("feature_names", FEATURE_NAMES)
        )
        probability = self.novel_validator.predict_proba(features)[:, 1]
        distribution = E.target_senses(slot, self.fit.members[0].sense_model)
        difficulty = self.difficulty.score(
            [token.text] + [row["word"] for row in candidates]
        )
        protected = any(
            span.protected
            and span.start < token.idx + len(token)
            and token.idx < span.end
            for span in detect_terms(doc)
        )
        rows = []
        for index, (candidate, fit) in enumerate(zip(candidates, probability)):
            output = substitute(doc, token, candidate["word"])
            checks = guardrails(doc.text, output, doc)
            grammar = GrammarValidator().validate(doc, output)
            mass = (
                target[1] == "n"
                and after_article(token)
                and mass_noun(candidate["word"])
            )
            gain = float(difficulty[0] - difficulty[index + 1])
            row = {
                **candidate,
                "output": output,
                "validation": validate_proposal(candidate, target[1]),
                "context_fit": float(fit),
                "fit": float(fit),
                "accept": float(fit),
                "complexity_delta": gain,
                "gain": gain,
                "utility": float(fit),
                "mechanical_checks_pass": checks["pass"]
                and grammar["pass"]
                and not protected
                and not mass,
                "guardrails": checks,
                "grammar_validation": grammar,
                "wordnet_evidence": wordnet_evidence(
                    distribution, target[1], candidate["lemma"]
                ),
                "rejections": [],
                "semantic_certified": False,
            }
            if choose_novel([row], threshold) is None:
                row["rejections"].append("novel context or preservation policy")
            rows.append(row)
        result["candidates"].extend(rows)
        result["selected"] = choose_novel(rows, threshold)
        result["novel_status"] = (
            "fallback_selected" if result["selected"] else "novel_proposals_rejected"
        )
        return result
