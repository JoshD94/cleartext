"""Differential grammar checks for local simplification edits."""

from collections import Counter

from .features import nlp, pronunciations
from .ensemble_pipeline import article, mass_noun


def issues(doc):
    found = []

    def add(code, tokens):
        found.append(
            {
                "code": code,
                "start": min(t.idx for t in tokens),
                "anchor": " ".join(t.text.lower() for t in tokens),
            }
        )

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            determiners = [child for child in token.children if child.dep_ == "det"]
            for determiner in determiners:
                if determiner.lower_ in {"a", "an"}:
                    following = (
                        doc[determiner.i + 1] if determiner.i + 1 < len(doc) else token
                    )
                    if (
                        following.is_alpha
                        and following.lower_ in pronunciations()
                        and article(following.text) != determiner.lower_
                    ):
                        add("indefinite_article_sound", [determiner, following])
                    if token.tag_ == "NNS":
                        add("indefinite_article_plural", [determiner, token])
                    elif token.pos_ == "NOUN" and mass_noun(token.text):
                        add("indefinite_article_mass_noun", [determiner, token])
                if determiner.lower_ in {"these", "those"} and token.tag_ == "NN":
                    add("demonstrative_number", [determiner, token])
        if token.tag_ not in {"VBZ", "VBP", "VBD"} or token.pos_ not in {"VERB", "AUX"}:
            continue
        subjects = [
            child for child in token.children if child.dep_ in {"nsubj", "nsubjpass"}
        ]
        if not subjects and token.dep_ in {"aux", "auxpass", "cop"}:
            subjects = [
                child
                for child in token.head.children
                if child.dep_ in {"nsubj", "nsubjpass"}
            ]
        if len(subjects) != 1:
            continue
        subject = subjects[0]
        if subject.pos_ not in {"NOUN", "PROPN", "PRON"}:
            continue
        conjunctions = [child for child in subject.children if child.dep_ == "conj"]
        if conjunctions:
            coordinators = [
                child
                for noun in [subject] + conjunctions
                for child in noun.children
                if child.dep_ == "cc"
            ]
            if not any(child.lower_ == "and" for child in coordinators):
                continue
            if any(
                child.pos_ not in {"NOUN", "PROPN", "PRON"} for child in conjunctions
            ):
                continue
            plural = True
        elif subject.lower_ in {"i", "you", "we", "they"}:
            plural = subject.lower_ != "i"
        elif subject.lower_ in {"he", "she", "it"}:
            plural = False
        elif subject.tag_ in {"NN", "NNS"}:
            plural = subject.tag_ == "NNS"
        else:
            continue
        if token.lemma_ == "be":
            expected = (
                ("were" if plural else "was")
                if token.lower_ in {"was", "were"}
                else ("am" if subject.lower_ == "i" else "are" if plural else "is")
            )
            if (
                token.lower_ in {"am", "is", "are", "was", "were"}
                and token.lower_ != expected
            ):
                add("subject_verb_number", [subject, token])
        elif token.tag_ in {"VBZ", "VBP"}:
            expected = "VBP" if plural or subject.lower_ == "i" else "VBZ"
            if token.tag_ != expected:
                add("subject_verb_number", [subject, token])
    return found


class GrammarValidator:
    def validate(self, original, output):
        original_doc = original if hasattr(original, "text") else nlp()(original)
        output_doc = output if hasattr(output, "text") else nlp()(output)
        before, after = issues(original_doc), issues(output_doc)
        existing = Counter(issue["code"] for issue in before)
        new = []
        for issue in after:
            key = issue["code"]
            if existing[key]:
                existing[key] -= 1
            else:
                new.append(issue)
        return {
            "pass": not new,
            "new_issues": new,
            "original_issues": before,
            "output_issues": after,
            "grammar_certified": False,
            "scope": "Increased counts of article, demonstrative and simple subject-verb agreement issues.",
        }
