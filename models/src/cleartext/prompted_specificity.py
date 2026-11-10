"""Experimental interactions between proposal support and WordNet specificity."""

import numpy as np

from .prompted_relations import features_prompted_relations


INTERACTION_NAMES = (
    "support_same_sense",
    "support_hypernym",
    "unsupported_hypernym",
    "support_hypernym_breadth",
)


def features_prompted_specificity(case):
    base = features_prompted_relations(case)

    def row(case, candidate):
        values = base(case, candidate)
        if values is None:
            return None
        # Prior columns: 39 original, 9 relation, 3 BERT support.
        same, broader, breadth = values[39], values[40], values[43]
        suggested, reciprocal_rank = values[48:50]
        return np.r_[
            values,
            reciprocal_rank * same,
            reciprocal_rank * broader,
            (1 - suggested) * broader,
            reciprocal_rank * breadth,
        ]

    return row
