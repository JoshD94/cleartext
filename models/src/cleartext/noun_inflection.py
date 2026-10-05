"""Opt-in grammatical-head inflection for WordNet noun phrases."""

from functools import lru_cache

from lemminflect import getInflection, getLemma
from nltk.corpus import wordnet as wn

from .features import nlp
from .generation import WordNetGenerator, inflect


@lru_cache(1)
def phrase_exceptions():
    """Use the existing WordNet noun exceptions, without a project word list."""
    plural = {}
    with wn.open("noun.exc") as stream:
        for line in stream:
            form, *bases = line.split()
            for base in bases:
                if "_" in base and form.replace("_", "").isalpha():
                    plural.setdefault(base, []).append(form)
    return {base: tuple(forms) for base, forms in plural.items()}


def match_case(word, original):
    if original.istitle():
        return word[:1].upper() + word[1:]
    return word.upper() if original.isupper() else word


@lru_cache(10000)
def noun_head(lemma):
    words = lemma.split("_")
    doc = nlp()("the " + " ".join(words))
    content = list(doc)[1:]
    if [token.text for token in content] != words:
        return None
    heads = [
        i
        for i, token in enumerate(content)
        if token.dep_ == "ROOT" and token.pos_ in {"NOUN", "PROPN"}
    ]
    if len(heads) != 1:
        return None
    index = heads[0]
    # Some postpositive adjectives are tagged as nouns in isolation. If both
    # readings are possible, the parse alone is insufficient to choose a head.
    if (
        index
        and content[index - 1].pos_ in {"NOUN", "PROPN"}
        and wn.synsets(words[index], pos="a")
    ):
        return None
    if content[index].pos_ == "PROPN" and any(
        token.pos_ == "PROPN" for token in content[:index]
    ):
        return None
    lemmas = getLemma(words[index], upos="NOUN", lemmatize_oov=False)
    return (index, lemmas[0] if lemmas else words[index])


def headed_inflect(lemma, tag, original):
    if "_" not in lemma or tag not in {"NN", "NNS"}:
        return inflect(lemma, tag, original)
    if tag == "NNS":
        exceptions = phrase_exceptions().get(lemma)
        if exceptions:
            return match_case(exceptions[0].replace("_", " "), original)
        # The inflection lexicon also knows some hyphenated compounds, including
        # postpositive heads that a short isolated parse cannot reliably find.
        forms = getInflection(lemma.replace("_", "-"), tag=tag, inflect_oov=False)
        if forms and all(part.isalpha() for part in forms[0].split("-")):
            return match_case(forms[0].replace("-", " "), original)
    head = noun_head(lemma)
    if head is None:
        return None
    index, base = head
    if tag == "NNS":
        from .ensemble_pipeline import mass_noun

        if mass_noun(base):
            return None
    forms = getInflection(base, tag=tag, inflect_oov=False)
    if not forms:
        return None
    words = lemma.split("_")
    words[index] = forms[0]
    return match_case(" ".join(words), original)


class HeadedWordNetGenerator(WordNetGenerator):
    def inflect(self, lemma, tag, original):
        return headed_inflect(lemma, tag, original)
