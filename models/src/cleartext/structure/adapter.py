"""Run one checked structural stage before lexical simplification."""

from ..features import nlp
from ..rewrites import structural
from .common import structure_checks


def rewrite(text, model=None):
    if model is None:
        output, edits = structural(text)
        return (
            output,
            edits,
            {
                "backend": "rules",
                "before": text,
                "after": output,
                "changed": bool(edits),
                "rule": edits[0]["rule"] if edits else None,
            },
        )
    details = {"backend": "ranker", "before": text, "after": text, "changed": False}
    if len(text.split()) > 70 or len(list(nlp()(text).sents)) != 1:
        return (
            text,
            [],
            {
                **details,
                "reason": "The learned splitter accepts one sentence of at most 70 words.",
            },
        )
    prediction = model.predict(text)
    output = prediction["output"]
    checks = structure_checks(text, output)
    details.update(
        {
            "score": prediction.get("score"),
            "checks": checks,
            "candidate_count": prediction.get("candidate_count"),
        }
    )
    if not checks["pass"] or output == text:
        return (
            text,
            [],
            {
                **details,
                "reason": "No candidate passed the model threshold and structural checks.",
            },
        )
    details.update({"after": output, "changed": True, "rule": "learned rewrite ranker"})
    return (
        output,
        [
            {
                "stage": "structure",
                "rule": details["rule"],
                "before": text,
                "after": output,
            }
        ],
        details,
    )
