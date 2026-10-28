"""Opt-in repair of verb lemmas that cannot reproduce the observed word.

This checks dictionary morphology, not contextual meaning. Imports of the native
dictionary and generator are lazy so the decision logic needs no model loading.
"""

from dataclasses import dataclass


INFLECTED_VERB_TAGS = frozenset({"VBD", "VBN", "VBG", "VBZ"})


def native_inflections(lemma, tag):
    from lemminflect import getInflection

    return getInflection(lemma, tag=tag, inflect_oov=False)


def native_lemmas(surface):
    from lemminflect import getLemma

    return getLemma(surface, upos="VERB", lemmatize_oov=False)


def native_senses(lemma):
    from .candidate_validation import exact_lemma_senses

    return tuple(sense.name() for sense in exact_lemma_senses(lemma, "v"))


@dataclass(frozen=True)
class TargetRepair:
    original: tuple | None
    target: tuple | None
    reason: str
    original_forms: tuple = ()
    alternatives: tuple = ()

    @property
    def changed(self):
        return self.target != self.original


def repair_target(
    surface,
    target,
    *,
    inflections=native_inflections,
    lemmas=native_lemmas,
    senses=native_senses,
):
    """Repair only a known incompatible lemma with one supported alternative.

    Unknown forms, ambiguous analyses, valid round trips, and equivalent spelling
    inventories keep the original analysis. POS and tense never change. A repaired
    lemma still needs the existing sense model, meaning floor and candidate guards.
    Callbacks return dictionary forms, dictionary lemmas and exact sense IDs.
    """
    original = None if target is None else tuple(target)
    if original is None:
        return TargetRepair(None, None, "unresolved_target")
    lemma, pos, tag, fallback = original
    if pos != "v" or tag not in INFLECTED_VERB_TAGS:
        return TargetRepair(original, original, "outside_inflected_verbs")
    forms = tuple(inflections(lemma.casefold(), tag) or ())
    if not forms:
        return TargetRepair(original, original, "unknown_original_form")
    word = surface.casefold()
    if word in {form.casefold() for form in forms}:
        return TargetRepair(original, original, "valid_roundtrip", forms)
    alternatives = []
    for base in lemmas(word) or ():
        base = base.casefold()
        if base == lemma.casefold() or base in alternatives or not base.isalpha():
            continue
        if not senses(base):
            continue
        if word in {form.casefold() for form in inflections(base, tag) or ()}:
            alternatives.append(base)
    alternatives = tuple(sorted(alternatives))
    if len(alternatives) != 1:
        reason = "ambiguous_alternatives" if alternatives else "no_supported_alternative"
        return TargetRepair(original, original, reason, forms, alternatives)
    replacement = alternatives[0]
    if set(senses(lemma.casefold())) == set(senses(replacement)):
        return TargetRepair(original, original, "equivalent_sense_inventory", forms, alternatives)
    return TargetRepair(
        original, (replacement, pos, tag, fallback), "unique_dictionary_roundtrip", forms, alternatives
    )


def candidates_for_target(member, token, target, relations):
    """Regenerate with genuine edges for the repaired lemma and the member's inflector."""
    lemma, pos, tag, fallback = target
    found = {}
    # Native WordNet pointer enumeration can vary between processes. This order
    # is local to repaired cases; unchanged cases retain their cached ordering.
    for name, (source, origins) in sorted(relations(lemma, pos).items()):
        if source not in member.sources or not name.replace("_", "").isalpha():
            continue
        if "_" in name and not member.multiword:
            continue
        word = member.inflect(name, tag, token.text)
        if not word or word.casefold() == token.text.casefold():
            continue
        found.setdefault(word.casefold(), {
            "word": word,
            "lemma": name.replace("_", " "),
            "senses": list(origins),
            "source": source,
            "multiword": "_" in name,
            "pos_fallback": fallback,
        })
    return list(found.values())


class RoundTripWordNetGenerator:
    """Wrap a WordNet generator explicitly; no default loader registration.

    Unchanged analyses return the member's original candidates, preserving their
    order and provenance. Changed analyses discard the old lemma's candidates.
    """

    name = "dictionary_roundtrip_wordnet"

    def __init__(self, member, *, repair=repair_target, relations=None):
        self.member, self.repair, self.relations = member, repair, relations

    def generate(self, token):
        target, candidates = self.member.generate(token)
        if token is None or target is None:
            return target, candidates
        result = self.repair(token.text, target)
        if not result.changed or (result.target[3] and not self.member.fallback):
            return target, candidates
        relations = self.relations
        if relations is None:
            from .generation import relation_map

            relations = relation_map
        return result.target, candidates_for_target(self.member, token, result.target, relations)
