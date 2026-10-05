"""Opt-in orchestration of all six ClearText inspection and protection blocks."""

import json

from . import ensemble as E
from .data import ROOT
from .detail_preservation import DetailPreservationScorer
from .document_consistency import DocumentConsistencyChecker
from .edit_quality import EditQualityEvaluator
from .ensemble_pipeline import EnsembleClearText
from .features import nlp
from .grammar_validation import GrammarValidator
from .jargon_explanations import JargonExplainer


class BuildingBlockClearText(EnsembleClearText):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = {}

    @classmethod
    def load(cls, run=ROOT / "runs/ensemble-building-blocks-20261002"):
        pipe = super().load(run)
        config = json.loads((run / "config.json").read_text())
        pipe.blocks = config.get("building_blocks", {})
        allowed = {
            "detail_evidence",
            "grammar_validation",
            "jargon_explanations",
            "document_consistency",
            "edit_quality",
        }
        if set(pipe.blocks) - allowed or any(
            not isinstance(value, bool) for value in pipe.blocks.values()
        ):
            raise ValueError("building_blocks must contain supported boolean flags")
        return pipe

    def decide(self, doc, token, slot, candidates):
        rows = super().decide(doc, token, slot, candidates)
        if self.blocks.get("detail_evidence") and rows:
            distribution = E.target_senses(slot, self.fit.members[0].sense_model)
            scorer = DetailPreservationScorer()
            for row in rows:
                row["detail_evidence"] = scorer.evidence(distribution, row)
        return rows

    def first_guarded(self, doc, rows, guard=True):
        while True:
            selected = super().first_guarded(doc, rows, guard)
            if (
                selected is None
                or not guard
                or not self.blocks.get("grammar_validation")
            ):
                return selected
            report = GrammarValidator().validate(doc, selected["output"])
            selected["grammar_validation"] = report
            if report["pass"]:
                return selected
            selected["rejections"] += sorted(
                {"grammar:" + issue["code"] for issue in report["new_issues"]}
            )

    def analyze(self, text, **kwargs):
        result = super().analyze(text, **kwargs)
        rollback = []
        if self.blocks.get("grammar_validation"):
            report = GrammarValidator().validate(text, result["output"])
            result["grammar_validation"] = report
            if not report["pass"]:
                rollback.append("grammar_validation")
        if self.blocks.get("document_consistency"):
            report = DocumentConsistencyChecker().check(
                text, result["output"], result["edits"]
            )
            result["document_consistency"] = report
            if not report["pass"]:
                rollback.append("document_consistency")
        if rollback:
            result["rejected_output"] = result["output"]
            result["rejected_edits"] = result["edits"]
            result["rejected_block_reports"] = {
                name: result[name]
                for name in ("grammar_validation", "document_consistency")
                if name in result
            }
            result["output"], result["edits"] = text, []
            if self.blocks.get("grammar_validation"):
                result["grammar_validation"] = GrammarValidator().validate(text, text)
            if self.blocks.get("document_consistency"):
                result["document_consistency"] = DocumentConsistencyChecker().check(
                    text, text
                )
            if "reading_level_before" in result:
                result["reading_level_after"] = result["reading_level_before"]
            if self.technical_terms:
                from .technical_terms import preserved_terms

                result["term_preservation"] = {
                    **preserved_terms(nlp()(text), text),
                    "blocked_stages": rollback,
                }
        result["block_rollbacks"] = rollback
        if self.blocks.get("jargon_explanations"):
            result["jargon_explanations"] = JargonExplainer().explain(
                nlp()(result["output"]), self
            )
        if self.blocks.get("edit_quality"):
            detail = [
                {
                    "target": rank["target"],
                    "start": rank["start"],
                    "word": candidate["word"],
                    "evidence": candidate["detail_evidence"],
                    "scope": "Selected candidate for inspection; may be unapplied.",
                }
                for rank in result["trace"]
                for candidate in [rank.get("selected")]
                if candidate is not None and "detail_evidence" in candidate
            ]
            result["edit_quality"] = EditQualityEvaluator().evaluate(
                text, result["output"], self, result["edits"], detail
            )
        return result
