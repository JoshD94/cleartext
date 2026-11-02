"""Candidate-specific dictionary sense priors for an optional learned ranker.

WordNet usage counts and verb frames are incomplete evidence. A shared synset
can contain an uncommon reading of the replacement; no feature certifies it.
"""

from functools import lru_cache
import math

import numpy as np
from nltk.corpus import wordnet as wn

from .context_similarity import features_prompted_similarity


USAGE_FEATURE_NAMES = (
    "same_sense_prior_mass",
    "linked_sense_prior_mass",
    "maximum_same_sense_prior",
    "prior_mass_times_sense_fit",
    "shared_verb_frame_coverage",
    "shared_verb_frame_available",
    "candidate_prior_entropy",
    "shared_prior_gap",
    "sense_evidence_available",
)


def _key(word):
    return word.casefold().replace(" ", "_")


@lru_cache(50000)
def _forms(word, pos):
    # synsets() expands all WordNet morphological analyses, not just the first.
    # For example, "found" has both found and find readings.
    return frozenset([_key(word), *wn._morphy(_key(word), "a" if pos == "s" else pos)])


def _lemmas(sense, word):
    forms = _forms(word, sense.pos())
    return [lemma for lemma in sense.lemmas() if _key(lemma.name()) in forms]


@lru_cache(50000)
def sense_priors(word, pos):
    """Immutable add-one lemma-count priors, including zero-count senses."""
    senses = wn.synsets(_key(word), pos="a" if pos == "s" else pos)
    weights = [(sense.name(), 1 + sum(l.count() for l in _lemmas(sense, word)))
               for sense in senses]
    total = sum(value for _, value in weights)
    return tuple((name, value / total) for name, value in weights)


def usage_evidence(target_lemma, pos, candidate_lemma, distribution, sense_fit):
    values = list(distribution.values())
    if any(not np.isfinite(p) or p < 0 for p in values):
        raise ValueError("sense probabilities must be finite and nonnegative")
    if values and not np.isclose(sum(values), 1.0, atol=1e-6):
        raise ValueError("sense probabilities must sum to one")
    prior = dict(sense_priors(candidate_lemma, pos))
    original_prior = dict(sense_priors(target_lemma, pos))
    same = linked = maximum = frames = frame_available = gap = 0.0
    for name, mass in distribution.items():
        original = wn.synset(name)
        canonical = original.name()
        target_entries = _lemmas(original, target_lemma)
        if not target_entries:
            raise ValueError("contextual sense lacks the original target lemma")
        shared = _lemmas(original, candidate_lemma)
        if shared:
            p = prior.get(canonical, 0.0)
            same += mass * p
            maximum = max(maximum, p)
            gap += mass * (original_prior.get(canonical, 0.0) - p)
            if pos == "v":
                source_frames = {f for lemma in target_entries for f in lemma.frame_ids()}
                candidate_frames = {f for lemma in shared for f in lemma.frame_ids()}
                if source_frames and candidate_frames:
                    frame_available += mass
                    frames += mass * len(source_frames & candidate_frames) / len(source_frames)
        neighbors = [original] + original.hypernyms() + original.verb_groups()
        neighbors += original.similar_tos() + original.also_sees()
        # One candidate sense can be reached by several edges. Count it once.
        names = {s.name() for s in neighbors if _lemmas(s, candidate_lemma)}
        linked += mass * sum(prior.get(n, 0.0) for n in names)
    entropy = -sum(p * math.log(p) for p in prior.values())
    if len(prior) > 1:
        entropy /= math.log(len(prior))
    return np.asarray([
        same, linked, maximum, same * sense_fit, frames, frame_available,
        entropy, gap, float(bool(distribution) and bool(prior)),
    ])


def features_sense_usage(case):
    if not case["candidates"]:
        return lambda case, candidate: None
    if "sense_distribution" not in case or not case.get("target_info"):
        raise ValueError("sense usage requires contextual senses and resolved target_info")
    base = features_prompted_similarity(case)

    def feature(case, candidate):
        lemma, pos = case["target_info"][:2]
        return np.r_[base(case, candidate), usage_evidence(
            lemma, pos, candidate["lemma"], case["sense_distribution"],
            float(candidate["x"][0]),
        )]

    return feature


features_sense_usage.requires_sense_distribution = True
