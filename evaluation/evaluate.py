#!/usr/bin/env python3
"""Baseline implementation of the Week 1 formula metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
NUMBER_PATTERN = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,4}[/-]\d{1,2}[/-]\d{1,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
NEGATIONS = {
    "not", "no", "never", "neither", "without", "cannot", "can't",
    "doesn't", "isn't", "won't", "don't", "didn't", "n't",
}
DEFAULT_TECHNICAL_TERMS = {
    "algorithm", "api", "asynchronous", "cache", "database", "encryption",
    "latency", "mitochondria", "photosynthesis", "regression", "variance",
}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def words(text: str) -> list[str]:
    return [match.casefold() for match in WORD_PATTERN.findall(text)]


def syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.casefold())
    if len(word) <= 3:
        return 1
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def word_complexity(
    word: str,
    context: Iterable[str] = (),
    *,
    weights: dict[str, float] | None = None,
    zipf_frequency: float | None = None,
    pos: float = 0.0,
) -> float:
    """Calculate C(w,s), using supplied Zipf/POS values when available.

    The fallback frequency is a transparent length-based proxy for this
    dependency-free baseline; production evaluation should provide Zipf data.
    """
    weights = {
        "A": 0.0, "B": 1.0, "C": 0.08, "D": 0.25, "E": 0.2, "F": 0.1,
        **(weights or {}),
    }
    frequency = zipf_frequency if zipf_frequency is not None else 6.0 - len(word) / 2
    context = list(context)
    context_complexity = (
        sum(len(item) + syllables(item) for item in context) / len(context)
        if context else 0.0
    )
    value = (
        weights["A"]
        + weights["B"] * -frequency
        + weights["C"] * len(word)
        + weights["D"] * syllables(word)
        + weights["E"] * context_complexity
        + weights["F"] * pos
    )
    return sigmoid(value)


def readability(text: str) -> dict[str, float]:
    tokens = words(text)
    sentence_count = max(1, len(re.findall(r"[.!?]+", text)))
    word_count = max(1, len(tokens))
    syllable_count = sum(syllables(token) for token in tokens)
    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllable_count / word_count
    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    fog = 0.4 * (words_per_sentence + 100 * sum(len(w) >= 7 for w in tokens) / word_count)
    smog = 1.043 * math.sqrt(30 * sum(syllables(w) >= 3 for w in tokens)) + 3.1291
    return {
        "flesch": flesch,
        "grade": grade,
        "fog": fog,
        "smog": smog,
        "sentence_length": words_per_sentence,
        "word_length": sum(map(len, tokens)) / word_count,
    }


def readability_score(text: str) -> float:
    values = readability(text)
    components = [
        clamp(values["flesch"] / 100),
        1 - clamp(values["grade"] / 18),
        1 - clamp(values["fog"] / 20),
        1 - clamp(values["smog"] / 20),
        1 - clamp(values["sentence_length"] / 40),
        1 - clamp(values["word_length"] / 12),
    ]
    return sum(components) / len(components)


def normalized_reduction(before: int, after: int) -> float:
    return clamp((before - after) / max(1, before))


def information_preservation(original: str, simplified: str, technical_terms: set[str]) -> dict[str, float]:
    original_words, simplified_words = words(original), words(simplified)
    original_counts, simplified_counts = Counter(original_words), Counter(simplified_words)
    shared = sum((original_counts & simplified_counts).values())
    tfidf = shared / max(1, len(original_words))
    embedding_proxy = len(set(original_words) & set(simplified_words)) / max(
        1, len(set(original_words) | set(simplified_words))
    )
    nli_forward = shared / max(1, len(original_words))
    nli_backward = shared / max(1, len(simplified_words))
    nli_proxy = (nli_forward + nli_backward) / 2
    entities = re.findall(r"\b[A-Z][a-z]+\b", original)
    preserved_entities = sum(entity.casefold() in simplified.casefold() for entity in entities)
    numbers = [normalize_number(match) for match in NUMBER_PATTERN.findall(original)]
    preserved_numbers = sum(number in [normalize_number(n) for n in NUMBER_PATTERN.findall(simplified)] for number in numbers)
    dates = [date.casefold() for date in DATE_PATTERN.findall(original)]
    preserved_dates = sum(date in simplified.casefold() for date in dates)
    quantities = re.findall(r"\b\d[\d,]*(?:\.\d+)?\s*(?:kg|g|lb|lbs|m|cm|km|%|percent)\b", original, re.I)
    preserved_quantities = sum(quantity.casefold() in simplified.casefold() for quantity in quantities)
    original_negations = set(original_words) & NEGATIONS
    preserved_negations = sum(item in simplified.casefold() for item in original_negations)
    technical = set(original_words) & technical_terms
    preserved_technical = sum(item in simplified_words for item in technical)
    protected = []
    for count, preserved in (
        (len(entities), preserved_entities),
        (len(numbers), preserved_numbers),
        (len(dates), preserved_dates),
        (len(quantities), preserved_quantities),
        (len(original_negations), preserved_negations),
        (len(technical), preserved_technical),
    ):
        if count:
            protected.append(preserved / count)
    mean = (tfidf + embedding_proxy + nli_proxy) / 3
    critical = sum(protected) / len(protected) if protected else 1.0
    return {
        "M_tfidf": tfidf,
        "M_embedding": embedding_proxy,
        "M_nli": nli_proxy,
        "M_avg": mean,
        "P_critical": critical,
        "P_meaning": 0.5 * mean + 0.5 * critical,
        "loss_info": 1 - (0.5 * mean + 0.5 * critical),
    }


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    original = record.get("original")
    simplified = record.get("simplified")
    if not isinstance(original, str) or not isinstance(simplified, str):
        raise ValueError("each record requires string 'original' and 'simplified' fields")
    technical_terms = set(record.get("technical_terms", DEFAULT_TECHNICAL_TERMS))
    before_words, after_words = words(original), words(simplified)
    before_complexity = sum(word_complexity(w) for w in before_words) / max(1, len(before_words))
    after_complexity = sum(word_complexity(w) for w in after_words) / max(1, len(after_words))
    lexical = clamp(before_complexity - after_complexity)
    jargon_before = sum(word in technical_terms for word in before_words)
    jargon_after = sum(word in technical_terms for word in after_words)
    jargon = normalized_reduction(jargon_before, jargon_after)
    read_before, read_after = readability_score(original), readability_score(simplified)
    readability_simplicity = clamp(read_after - read_before + 0.5)
    components = {"S_lex": lexical, "S_read": readability_simplicity, "S_jargon": jargon}
    weights = record.get("sentence_weights", {"G": 1 / 3, "H": 1 / 3, "I": 1 / 3})
    if not all(isinstance(weights.get(label), (int, float)) for label in ("G", "H", "I")):
        raise ValueError("sentence_weights must contain numeric G, H, and I values")
    if not math.isclose(sum(float(weights[label]) for label in ("G", "H", "I")), 1.0):
        raise ValueError("sentence_weights G, H, and I must sum to 1")
    sentence_score = sum(components[key] * float(weights[label]) for key, label in zip(components, ("G", "H", "I")))
    info = information_preservation(original, simplified, technical_terms)
    utility = sentence_score - float(record.get("lambda", 1.0)) * info["loss_info"]
    threshold = float(record.get("threshold", 0.2))
    return {
        "word_complexity_before": before_complexity,
        "word_complexity_after": after_complexity,
        **components,
        "S": sentence_score,
        **info,
        "U": utility,
        "accepted": info["loss_info"] <= threshold and info["P_critical"] == 1.0,
    }


def normalize_number(value: str) -> str:
    return value.replace(",", "").casefold()


def evaluate(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    if not records:
        raise ValueError("the evaluation dataset is empty")
    results = [evaluate_record(record) for record in records]
    keys = [
        key for key, value in results[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    metrics = {key: sum(float(result[key]) for result in results) / len(results) for key in keys}
    metrics["records"] = len(results)
    metrics["accepted_rate"] = sum(result["accepted"] for result in results) / len(results)
    return metrics


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number} is not valid JSON: {error.msg}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object")
                records.append(record)
    except OSError as error:
        raise ValueError(f"unable to read {path}: {error}") from error
    return records


def main() -> int:
    metric_help = """
Metric definitions:
  word_complexity_before/after
      Average word-level complexity before and after simplification.
  S_lex
      Lexical simplicity improvement from reduced word complexity.
  S_read
      Readability simplicity score based on Flesch, grade, Fog, SMOG,
      sentence length, and word length.
  S_jargon
      Normalized reduction in weighted, replaceable technical jargon.
  S
      Weighted sentence simplicity score:
      G*S_lex + H*S_read + I*S_jargon.
  M_tfidf
      Baseline weighted-token overlap between original and simplified text.
  M_embedding
      Baseline vocabulary overlap proxy for embedding similarity.
  M_nli
      Baseline bidirectional overlap proxy for entailment.
  M_avg
      Average of M_tfidf, M_embedding, and M_nli.
  P_critical
      Average preservation of entities, numbers, dates, quantities, negation,
      and technical terms when those categories occur.
  P_meaning
      Combined meaning score: 0.5*M_avg + 0.5*P_critical.
  loss_info
      Information loss: 1 - P_meaning. Lower is better.
  U
      Candidate utility: S - lambda*loss_info. Higher is better.
  accepted_rate
      Fraction of records passing loss_info <= threshold and P_critical == 1.
"""
    parser = argparse.ArgumentParser(
        description="Evaluate Week 1 simplicity and information-preservation formulas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="JSONL file with original and simplified text",
    )
    parser.epilog = metric_help
    args = parser.parse_args()
    try:
        metrics = evaluate(read_jsonl(args.dataset))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
