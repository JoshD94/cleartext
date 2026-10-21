"""Document-wide term, definition and literal-fact consistency checks."""

from collections import Counter, defaultdict
import re

from .features import nlp
from .lexical import guardrails
from .technical_terms import detect_terms, preserved_terms, surface_pattern


def acronym_definitions(doc):
    spans = detect_terms(doc)
    short = [span for span in spans if "defined_acronym" in span.reasons]
    long = [span for span in spans if "acronym_definition" in span.reasons]
    definitions = []
    for name in short:
        for expansion in long:
            first, second = sorted([name, expansion], key=lambda span: span.start)
            if first.end > second.start:
                continue
            if re.fullmatch(
                r"\s*\(\s*", doc.text[first.end : second.start]
            ) and re.match(r"\s*\)", doc.text[second.end :]):
                definitions.append(
                    {
                        "acronym": name.text,
                        "expansion": expansion.text,
                        "start": name.start,
                        "long_start": expansion.start,
                    }
                )
    return definitions


def conflicting_definitions(doc):
    meanings = defaultdict(set)
    for definition in acronym_definitions(doc):
        meanings[definition["acronym"]].add(
            " ".join(definition["expansion"].casefold().split())
        )
    return {
        name: sorted(values) for name, values in meanings.items() if len(values) > 1
    }


class DocumentConsistencyChecker:
    def check(self, original, output, edits=()):
        old = original if hasattr(original, "text") else nlp()(original)
        new = output if hasattr(output, "text") else nlp()(output)
        terms = preserved_terms(old, new.text)
        facts = guardrails(old.text, new.text, old)
        failed = [
            name
            for name in ("numbers", "quantities", "negation", "modality")
            if not facts["checks"][name]
        ]
        entity_counts = Counter(entity.text.casefold() for entity in old.ents)
        missing_entities = [
            entity
            for entity, count in entity_counts.items()
            if len(list(re.finditer(surface_pattern(entity), new.text, re.I))) < count
        ]
        if missing_entities:
            failed.append("entity_occurrences")
        if not terms["pass"]:
            failed.append("protected_term_occurrences")
        before, after = conflicting_definitions(old), conflicting_definitions(new)
        introduced = {
            name: values for name, values in after.items() if before.get(name) != values
        }
        if introduced:
            failed.append("conflicting_acronym_definitions")
        grouped = defaultdict(set)
        for edit in edits:
            if edit.get("stage") == "word":
                grouped[edit["source"].casefold(), edit.get("sense_id")].add(
                    edit["replacement"].casefold()
                )
        inconsistent, review = [], []
        for (source, sense), replacements in grouped.items():
            if len(replacements) < 2:
                continue
            row = {
                "source": source,
                "sense_id": sense,
                "replacements": sorted(replacements),
            }
            (inconsistent if sense else review).append(row)
        if inconsistent:
            failed.append("inconsistent_sense_replacements")
        return {
            "pass": not failed,
            "failed": failed,
            "terms": terms,
            "missing_entities": missing_entities,
            "conflicting_definitions": after,
            "introduced_conflicts": introduced,
            "inconsistent_replacements": inconsistent,
            "replacement_review": review,
            "semantic_certified": False,
        }
