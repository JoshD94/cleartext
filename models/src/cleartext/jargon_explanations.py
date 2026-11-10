"""Attributable dictionary glosses and document-defined acronym expansions."""

from nltk.corpus import wordnet as wn

from . import ensemble as E
from .document_consistency import acronym_definitions
from .generation import resolve_target
from .technical_terms import detect_terms


class JargonExplainer:
    def explain(self, doc, pipe=None, limit=20):
        if not 1 <= limit <= 100:
            raise ValueError("gloss limit must be between 1 and 100")
        spans = detect_terms(doc)
        definitions = acronym_definitions(doc)
        annotations = []
        targets = {
            (span.start, span.end): {
                "start": span.start,
                "end": span.end,
                "term": span.text,
            }
            for span in spans
            if span.protected
        }
        if pipe is not None:
            tokens = [
                token
                for token in doc
                if token.is_alpha
                and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
                and not token.is_stop
                and not token.ent_type_
            ]
            for token in tokens:
                if any(
                    span.protected and span.start <= token.idx < span.end
                    for span in spans
                ):
                    continue
                if pipe.flagged(doc, token):
                    targets[token.idx, token.idx + len(token)] = {
                        "start": token.idx,
                        "end": token.idx + len(token),
                        "term": token.text,
                    }
        for key, row in sorted(targets.items()):
            if len(annotations) >= limit:
                break
            unique = {}
            for definition in definitions:
                if definition["acronym"] == row["term"]:
                    unique.setdefault(
                        " ".join(definition["expansion"].casefold().split()),
                        definition["expansion"],
                    )
            expansions = sorted(unique.values())
            if expansions:
                row.update(
                    source="document_definition",
                    status="available" if len(expansions) == 1 else "ambiguous",
                    explanation=expansions[0] if len(expansions) == 1 else None,
                    alternatives=expansions,
                )
            else:
                tokens = [
                    token
                    for token in doc
                    if row["start"] <= token.idx < row["end"] and token.is_alpha
                ]
                lemma = "_".join(token.lemma_.lower() for token in tokens)
                senses = wn.synsets(lemma, pos="n") if len(tokens) > 1 else []
                distribution = {}
                if len(tokens) == 1:
                    target = resolve_target(tokens[0])
                    if target:
                        senses = wn.synsets(target[0], pos=target[1])
                        if pipe is not None:
                            distribution = E.target_senses(
                                E.Slot(doc, tokens[0], target=target),
                                pipe.fit.members[0].sense_model,
                            )
                if not senses:
                    row.update(
                        source="wordnet",
                        status="unavailable",
                        explanation=None,
                        alternatives=[],
                    )
                else:
                    ordered = sorted(
                        senses, key=lambda sense: -distribution.get(sense.name(), 0.0)
                    )
                    mass = distribution.get(ordered[0].name(), 0.0)
                    margin = (
                        mass - distribution.get(ordered[1].name(), 0.0)
                        if len(ordered) > 1
                        else mass
                    )
                    available = (
                        len(senses) == 1
                        or bool(distribution)
                        and mass >= 0.75
                        and margin >= 0.20
                    )
                    row.update(
                        source="wordnet",
                        status="available" if available else "ambiguous",
                        explanation=ordered[0].definition() if available else None,
                        sense_id=ordered[0].name() if available else None,
                        model_sense_probability=float(mass) if distribution else None,
                        alternatives=[
                            {"sense_id": sense.name(), "definition": sense.definition()}
                            for sense in ordered[:3]
                        ],
                    )
            row["semantic_certified"] = False
            annotations.append(row)
        return annotations
