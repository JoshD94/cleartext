"""Opt-in experiment that rejects direct WordNet broadening after model scoring.

This keeps all candidates in the learned features. It does not change v7's default,
and it cannot detect lost detail within a synset or a wrong contextual sense.
"""

from functools import lru_cache

from nltk.corpus import wordnet as wn

from .ensemble_pipeline import EnsembleClearText, LATEST


@lru_cache(20000)
def hypernym_links(lemma, senses):
    key = lemma.lower().replace(" ", "_")
    return tuple(
        (name, parent.name())
        for name in senses
        for parent in wn.synset(name).hypernyms()
        if key in {word.lower() for word in parent.lemma_names()}
    )


def broader_links(candidate, target_info=None):
    """Check actual hypernym edges. Generator's 'hypernym' bucket also has verb groups."""
    if candidate.get("source") != "hypernym":
        return ()
    senses = candidate.get("senses")
    if senses is None:
        # Cached decision tables omit candidate sense lists but retain the resolved target.
        from .generation import relation_map

        if target_info is None:
            raise ValueError(
                "candidate lacks sense provenance; target_info is required"
            )
        key = candidate["lemma"].lower().replace("_", " ")
        entries = [
            value[1]
            for name, value in relation_map(target_info[0], target_info[1]).items()
            if name.lower().replace("_", " ") == key
        ]
        if len(entries) != 1:
            raise ValueError(
                f"cannot recover WordNet provenance for {candidate['lemma']}"
            )
        senses = entries[0]
    return hypernym_links(candidate["lemma"], tuple(senses))


class ConservativeDetailClearText(EnsembleClearText):
    @classmethod
    def load(cls, run=LATEST):
        pipe = super().load(run)
        if pipe.decision is None:
            raise ValueError("detail policy requires a learned decision pipeline")
        return pipe

    def first_guarded(self, doc, rows, guard=True):
        for candidate in rows:
            links = broader_links(candidate)
            if links:
                candidate["detail_links"] = links
                candidate["rejections"].append(
                    "experimental detail policy: broader WordNet meaning"
                )
        return super().first_guarded(doc, rows, guard)
