"""Conservative discourse diagnostics shared by training and generation.

Exact marker counts disallow valid synonyms as well as genuine omissions.
Uncertainty scope handles shared-subject conjunctions, not general semantics.
"""

from collections import Counter

RELATIONS = {
    "so",
    "therefore",
    "however",
    "otherwise",
    "or",
    "yet",
    "instead",
    "since",
    "as",
    "though",
    "whereas",
    "once",
    "whether",
    "provided",
    "nevertheless",
    "nonetheless",
    "consequently",
    "thus",
    "hence",
}
UNCERTAINTY = {"maybe", "perhaps", "probably", "possibly"}


def uncertainty_scopes(doc):
    result = set()
    for predicate in doc:
        if predicate.pos_ not in {"VERB", "AUX", "ADJ"}:
            continue
        current, seen = predicate, set()
        while current.i not in seen:
            seen.add(current.i)
            result.update(
                (predicate.lemma_.casefold(), child.lower_)
                for child in current.children
                if child.dep_ == "advmod" and child.lower_ in UNCERTAINTY
            )
            if current.dep_ != "conj" or any(
                c.dep_ in {"nsubj", "nsubjpass"} for c in current.children
            ):
                break
            current = current.head
    return result


def discourse_checks(old, new):
    return {
        "discourse_markers": Counter(t.lower_ for t in old if t.lower_ in RELATIONS)
        == Counter(t.lower_ for t in new if t.lower_ in RELATIONS),
        "uncertainty_scope": uncertainty_scopes(old) == uncertainty_scopes(new),
    }
