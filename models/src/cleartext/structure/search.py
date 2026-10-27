"""Opt-in checked beam search with bounded, word-preserving boundary repair."""

from functools import lru_cache

from ..features import nlp
from .common import complete_clause, normalize, structure_checks
from .seq2seq import PREFIX, SentenceGenerator, complement_scopes, predicate_tenses

COORDINATORS = {"and", "but", "or", "nor", "yet", "so", "for"}
SUBORDINATORS = {
    "if",
    "unless",
    "because",
    "although",
    "while",
    "when",
    "until",
    "before",
    "after",
    "as",
    "since",
    "though",
    "whereas",
    "once",
    "provided",
}


@lru_cache(maxsize=1024)
def independent(text):
    """Parse short pieces separately to avoid a whole-sentence repair parse."""
    sentences = list(nlp()(text).sents)
    return len(sentences) == 1 and complete_clause(sentences[0])


def splice_boundary(sentence):
    """Find a comma between two independently complete clauses.

    Coordinating conjunctions after commas already make a valid boundary.
    Parenthetical and quoted text abstains from repair. Parser errors remain.
    """
    if any(t.is_quote or t.text in {"(", ")", "[", "]", "—", "–"} for t in sentence):
        return None
    start = sentence.start_char
    for token in sentence:
        if token.text != ",":
            continue
        if (
            token.i + 1 < sentence.end
            and sentence.doc[token.i + 1].lower_ in COORDINATORS
        ):
            continue
        left = sentence.text[: token.idx - start].strip()
        right = sentence.text[token.idx + 1 - start :].strip()
        if left and right and independent(left) and independent(right):
            return token.idx
    return None


def boundary_checks(source_doc, output_doc):
    """Additional grammar diagnostics without changing saved ranker features."""
    source_roots = {s.root.lemma_.lower() for s in source_doc.sents}
    sentences = list(output_doc.sents)
    anchored = True
    for sentence in sentences:
        first = next((t for t in sentence if t.is_alpha), None)
        if first is not None and first.lower_ in SUBORDINATORS:
            # A source-relative anchor catches promoted embedded roots even
            # when the output parser incorrectly calls the fragment complete.
            anchored &= sentence.root.lemma_.lower() in source_roots
    return {
        "independent_boundaries": anchored,
        "no_comma_splice": all(splice_boundary(s) is None for s in sentences),
    }


def checked(source, raw, source_doc, terminated=True, boundaries=True):
    new = nlp()(raw)
    result = structure_checks(source, raw, source_doc, new)
    added = {
        "predicate_tense": predicate_tenses(source_doc) == predicate_tenses(new),
        "complement_scope": complement_scopes(source_doc) == complement_scopes(new),
        "generation_terminated": terminated,
    }
    if boundaries:
        added.update(boundary_checks(source_doc, new))
    result["checks"].update(added)
    result["failed"].extend(k for k, v in added.items() if not v)
    result["pass"] = not result["failed"]
    return result


def repair_boundaries(text):
    """Change at most three separators; never invent, delete or reorder words."""
    for _ in range(3):
        doc = nlp()(text)
        sentences = list(doc.sents)
        if len(sentences) >= 4:
            break
        boundary = next(
            (point for s in sentences if (point := splice_boundary(s)) is not None),
            None,
        )
        if boundary is None:
            break
        right = text[boundary + 1 :].lstrip()
        text = text[:boundary].rstrip() + ". " + right[:1].upper() + right[1:]
    return text


def select_candidate(source, proposals, source_doc=None):
    """Try model order, then its punctuation repair, with identity fallback.

    The model's preferred valid copy stays a copy. A failed high-ranked beam
    cannot hide a passing alternative. Selection never sees a reference.
    """
    old = source_doc if source_doc is not None else nlp()(source)
    inspected, seen = [], set()
    for rank, proposal in enumerate(proposals):
        raw, terminated = proposal["text"], proposal["terminated"]
        variants = [(raw, "beam")]
        repaired = repair_boundaries(raw)
        if repaired != raw:
            variants.append((repaired, "boundary_repair"))
        for text, origin in variants:
            key = (normalize(text), terminated)
            if key in seen:
                continue
            seen.add(key)
            checks = checked(source, text, old, terminated)
            candidate = {
                "output": text,
                "beam_rank": rank,
                "origin": origin,
                "checks": checks,
            }
            inspected.append(candidate)
            if checks["pass"]:
                return {
                    "raw": proposals[0]["text"],
                    "output": text,
                    "raw_checks": inspected[0]["checks"],
                    "output_checks": checks,
                    "selected_beam": rank,
                    "selected_origin": origin,
                    "candidates": inspected,
                }
    return {
        "raw": proposals[0]["text"] if proposals else source,
        "output": source,
        "raw_checks": inspected[0]["checks"]
        if inspected
        else checked(source, source, old),
        "output_checks": checked(source, source, old),
        "selected_beam": None,
        "selected_origin": "identity_fallback",
        "candidates": inspected,
    }


class CandidateGenerator(SentenceGenerator):
    """Four deterministic model beams, sharing one encoder/model in memory."""

    def predict(self, source, doc=None):
        import torch

        old = doc if doc is not None else nlp()(source)
        encoded = self.tokenizer(PREFIX + source, return_tensors="pt")
        if encoded["input_ids"].shape[1] > 192:
            result = select_candidate(source, [], old)
            result["reason"] = "source exceeds token limit"
            return result
        with torch.inference_mode():
            ids = self.model.generate(
                **{k: v.to(self.device) for k, v in encoded.items()},
                num_beams=4,
                num_return_sequences=4,
                do_sample=False,
                max_new_tokens=256,
                no_repeat_ngram_size=0,
            )
        proposals = [
            {
                "text": self.tokenizer.decode(row, skip_special_tokens=True).strip(),
                "terminated": self.tokenizer.eos_token_id in row.tolist()[1:],
            }
            for row in ids
        ]
        result = select_candidate(source, proposals, old)
        result["model_candidates"] = proposals
        return result
