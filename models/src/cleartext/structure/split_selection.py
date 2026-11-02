"""Opt-in source-anchored split selection. No models load on import.

These signatures reject some plausible splits and miss some meaning changes.
They are mechanical constraints, never a semantic quality certificate.
"""

from collections import Counter


SUBJECTS = {"nsubj", "nsubjpass"}
ARGUMENTS = SUBJECTS | {"dobj", "obj", "iobj", "attr", "oprd"}
EMBEDDED = {"ccomp", "xcomp", "advcl", "relcl", "acl", "csubj"}


def _name(token):
    # Include nominal modifiers, so two people sharing a surname do not merge.
    return tuple(
        t.lemma_.casefold()
        for t in sorted(token.subtree, key=lambda t: t.i)
        if t == token or t.dep_ in {"compound", "flat", "amod", "nummod", "poss"}
    )


def _governors(token):
    """Follow shared-subject coordination only, with a cycle guard."""
    seen = set()
    while token.i not in seen:
        seen.add(token.i)
        yield token
        if token.dep_ != "conj" or any(c.dep_ in SUBJECTS for c in token.children):
            break
        token = token.head


def event_signatures(doc):
    """Bind participants, tense and embedding to each occurrence of a predicate.

    The existing separate lemma sets can accept a swap between two occurrences
    of the same verb. Counters retain those associations. Pronoun resolution and
    active/passive paraphrases deliberately receive no special case.
    """
    result = Counter()
    for token in doc:
        if token.pos_ not in {"VERB", "AUX", "ADJ"} or token.dep_ in {
            "aux", "auxpass", "cop"
        }:
            continue
        governors = list(_governors(token))
        children = list(token.children)
        if not any(c.dep_ in SUBJECTS for c in children):
            children += [
                c for g in governors[1:] for c in g.children if c.dep_ in SUBJECTS
            ]
        arguments = tuple(sorted(
            (c.dep_, _name(c)) for c in children if c.dep_ in ARGUMENTS
        ))
        tense = ()
        for governor in governors:
            finite = [
                c for c in [governor, *governor.children]
                if (c == governor or c.dep_ in {"aux", "auxpass", "cop"})
                and "Fin" in c.morph.get("VerbForm")
            ]
            if finite:
                tense = tuple(sorted(
                    (tuple(c.morph.get("Tense")), tuple(c.morph.get("Mood")),
                     c.lemma_.casefold() if c != governor else "")
                    for c in finite
                ))
                break
        current, seen, scope = token, set(), []
        while current.i not in seen:
            seen.add(current.i)
            if current.dep_ in EMBEDDED:
                scope.append((current.dep_, current.head.lemma_.casefold(),
                              tuple(sorted(_name(c) for c in current.head.children
                                           if c.dep_ in SUBJECTS))))
            if current.head == current:
                break
            current = current.head
        operators = tuple(sorted({
            (c.dep_, c.lemma_.casefold())
            for governor in governors for c in governor.children
            if c.dep_ == "neg" or (c.dep_ == "aux" and c.tag_ == "MD")
        }))
        result[(token.lemma_.casefold(), arguments, tense, tuple(scope), operators)] += 1
    return result


def anchored_words(old, new):
    """Allow punctuation, boundary 'and' removal, and exact subject repetition.

    All other words must appear in source order, including prepositions and other
    function words that content-lemma checks miss. A copied subject is allowed
    only at a sentence start before its source coordinated predicate.
    """
    source = [t for t in old if not t.is_punct and not t.is_space]
    sentences = [[t.lower_ for t in s if not t.is_punct and not t.is_space]
                 for s in new.sents]
    if not source or not sentences or any(not s for s in sentences):
        return False
    states = {0}
    for index, words in enumerate(sentences):
        following = set()
        for start in states:
            cursor = start
            if index and cursor < len(source) and source[cursor].lower_ == "and":
                conjunction = source[cursor]
                if conjunction.dep_ != "cc":
                    continue
                cursor += 1
            variants = [words]
            if index and cursor < len(source):
                predicate = source[cursor]
                if predicate.dep_ == "conj" and not any(
                    c.dep_ in SUBJECTS for c in predicate.children
                ):
                    subjects = [c for c in predicate.head.children if c.dep_ in SUBJECTS]
                    if len(subjects) == 1:
                        subject = [t.lower_ for t in sorted(subjects[0].subtree,
                                                          key=lambda t: t.i)
                                   if not t.is_punct and not t.is_space]
                        if words[:len(subject)] == subject:
                            variants.append(words[len(subject):])
                        auxiliaries = [t.lower_ for t in sorted(
                            (c for c in predicate.head.children
                             if c.dep_ in {"aux", "auxpass", "cop"}), key=lambda t: t.i)]
                        prefix = subject + auxiliaries
                        if auxiliaries and words[:len(prefix)] == prefix:
                            variants.append(words[len(prefix):])
            for variant in variants:
                end = cursor + len(variant)
                if variant and [t.lower_ for t in source[cursor:end]] == variant:
                    following.add(end)
        states = following
    return len(source) in states


def split_checks(old, new):
    old_lengths = [sum(t.is_alpha for t in s) for s in old.sents]
    new_lengths = [sum(t.is_alpha for t in s) for s in new.sents]
    # Shared generation checks protect code bytes. Newly exposed splits also
    # abstain on quote/bracket delimiters rather than infer their scope.
    protected = any(t.is_quote or t.text in {"`", "``", "'''", "(", ")", "[", "]", "{", "}"}
                    or "`" in t.text for t in old)
    return {
        "unquoted_split": not protected,
        "shorter_split": bool(old_lengths and new_lengths)
        and len(new_lengths) > len(old_lengths)
        and max(new_lengths) < max(old_lengths),
        "source_anchored_words": anchored_words(old, new),
        "participant_events": event_signatures(old) == event_signatures(new),
    }


def select_split(source, proposals, old, *, parse=None, check=None):
    """Choose maximum longest-sentence reduction, with model rank as tie break.

    Every raw beam receives the unchanged runtime/boundary checks. No lower-beam
    repair, reference access, inference, or learned threshold is involved.
    """
    if parse is None:
        from ..features import nlp

        parse = nlp()
    if check is None:
        from .search import checked

        check = checked

    inspected, eligible = [], []
    for rank, proposal in enumerate(proposals):
        text = proposal["text"]
        new = parse(text)
        checks = check(source, text, old, proposal["terminated"])
        extra = split_checks(old, new)
        checks["checks"].update(extra)
        checks["failed"].extend(k for k, v in extra.items() if not v)
        checks["pass"] = not checks["failed"]
        candidate = {"output": text, "beam_rank": rank, "origin": "beam", "checks": checks}
        inspected.append(candidate)
        if checks["pass"]:
            longest = max(sum(t.is_alpha for t in s) for s in new.sents)
            eligible.append((longest, rank, candidate))
    chosen = min(eligible, key=lambda item: item[:2])[2] if eligible else None
    return {
        "raw": proposals[0]["text"] if proposals else source,
        "raw_checks": inspected[0]["checks"] if inspected else check(source, source, old),
        "output": chosen["output"] if chosen else source,
        "output_checks": chosen["checks"] if chosen else check(source, source, old),
        "selected_beam": chosen["beam_rank"] if chosen else None,
        "selected_origin": "source_anchored_split" if chosen else "identity_fallback",
        "selection_policy": "source_anchored_split",
        "candidates": inspected,
    }
