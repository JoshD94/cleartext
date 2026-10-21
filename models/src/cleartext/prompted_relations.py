"""BERT proposal support as ranking features for existing WordNet candidates."""

from functools import lru_cache

import numpy as np

from .bert_candidates import masked_predictions, proposals
from .features import nlp, target_token
from .relations import features_relations


def support_map(candidates):
    return {
        candidate["word"].lower(): (
            candidate["bert_rank"],
            candidate["bert_probability"],
        )
        for candidate in candidates
    }


@lru_cache(1024)
def support_for(text, target, start, target_info):
    token = target_token(nlp()(text), target, start)
    if token is None or token.idx != start:
        raise ValueError("BERT support requires the exact target offset")
    raw = masked_predictions([(text, start, start + len(token))], top_k=20)[0]
    _, candidates = proposals(token, raw, target_info)
    return support_map(candidates)


def features_prompted_relations(case):
    if not case["candidates"]:
        return lambda case, candidate: None
    base = features_relations(case)
    support = case.get("bert_support")
    if support is None:
        if "start" not in case:
            raise ValueError(
                "BERT support requires cached predictions or an exact target offset"
            )
        support = support_for(
            case["text"], case["target"], case["start"], tuple(case["target_info"])
        )

    def row(case, candidate):
        rank, probability = support.get(candidate["word"].lower(), (0, 0.0))
        return np.r_[
            base(case, candidate),
            float(rank > 0),
            1 / rank if rank else 0.0,
            probability,
        ]

    return row
