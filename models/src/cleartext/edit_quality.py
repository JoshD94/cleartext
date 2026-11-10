"""Separate readability and preservation measurements for proposed edits."""

from difflib import SequenceMatcher
import numpy as np

from .features import nlp
from .grammar_validation import GrammarValidator
from .document_consistency import DocumentConsistencyChecker


class EditQualityEvaluator:
    def evaluate(self, original, output, pipe=None, edits=(), detail_evidence=()):
        before, after = nlp()(original), nlp()(output)
        grammar = GrammarValidator().validate(before, after)
        consistency = DocumentConsistencyChecker().check(before, after, edits)
        readability = None
        if pipe is not None:

            def measure(doc):
                words = [
                    token.text
                    for token in doc
                    if token.is_alpha and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
                ]
                return {
                    "mean_word_complexity": float(np.mean(pipe.difficulty.score(words)))
                    if words
                    else None,
                    "reading_level": pipe.reading_level(doc) if words else None,
                }

            first, second = measure(before), measure(after)
            readability = {
                "before": first,
                "after": second,
                "word_complexity_reduction": first["mean_word_complexity"]
                - second["mean_word_complexity"]
                if first["mean_word_complexity"] is not None
                and second["mean_word_complexity"] is not None
                else None,
            }
        return {
            "changed": original != output,
            "token_difference_ratio": 1.0
            - SequenceMatcher(
                None, [t.text for t in before], [t.text for t in after], autojunk=False
            ).ratio(),
            "readability": readability,
            "grammar": grammar,
            "document_consistency": consistency,
            "candidate_detail_evidence": list(detail_evidence),
            "human_quality_score": None,
            "scope": "Separate model predictions and rule checks; no calibrated overall quality score.",
        }
