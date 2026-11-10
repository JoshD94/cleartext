"""Flag candidate data using saved grammar checks, without changing any text or rows.

No covered issue means only that these rules did not identify a problem. This
is not a grammaticality classifier or a replacement for the full evaluation set.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .grammar_preprocessing import AGREEMENT_RULES

SCREEN_POLICY = "grammar-screen-v1"
ERROR_RULES = AGREEMENT_RULES | {
    "NON3PRS_VERB",
    "MD_BASEFORM",
    "EN_A_VS_AN",
    "EN_CONTRACTION_SPELLING",
    "WONT_CONTRACTION",
    "IT_IS",
    "ITS_TO_IT_S",
    "THIS_PLURAL_OF",
    "ENGLISH_WORD_REPEAT_RULE",
    "EN_UNPAIRED_BRACKETS",
    "SENTENCE_WHITESPACE",
}
STYLE_RULES = {
    "COMMA_COMPOUND_SENTENCE",
    "COMMA_COMPOUND_SENTENCE_2",
    "WHETHER",
    "EXTREME_ADJECTIVES",
    "DIFFERENT_THAN",
    "SENT_START_CONJUNCTIVE_LINKING_ADVERB_COMMA",
    "EN_SPECIFIC_CASE",
}


def screen_result(grammar):
    """Classify the ORIGINAL input's warnings, not the corrected output.

    An exclusion list for a future clean-input panel can use `flagged` and
    `review`. The original benchmark must still be reported in full.
    """
    result = {
        "policy": SCREEN_POLICY,
        "status": "unavailable",
        "grammar_certified": False,
        "error_rules": [],
        "review_rules": [],
        "ignored_rules": [],
    }
    if not isinstance(grammar, dict):
        return result
    suggestions = grammar.get("suggestions")
    if (
        grammar.get("available") is not True
        or grammar.get("reason")
        or not isinstance(suggestions, list)
        or any(
            not isinstance(s, dict)
            or not isinstance(s.get("rule"), str)
            or not s["rule"]
            for s in suggestions
        )
    ):
        return result
    errors, review, ignored = set(), set(), set()
    for suggestion in suggestions:
        rule = suggestion["rule"]
        if (
            suggestion.get("reason") == "protected_span_or_insertion"
            or rule in STYLE_RULES
        ):
            ignored.add(rule)
        elif rule in ERROR_RULES:
            errors.add(rule)
        else:
            # Spelling can flag valid names and jargon. Unknown rules require
            # inspection, rather than silently disqualifying an input.
            review.add(rule)
    return {
        **result,
        "status": "flagged" if errors else "review" if review else "no_covered_issue",
        "error_rules": sorted(errors),
        "review_rules": sorted(review),
        "ignored_rules": sorted(ignored),
    }


def manifest(rows, source_sha256):
    records = []
    for row in rows:
        grammar = row["grammar"]
        if grammar.get("original") != row["source"]:
            raise ValueError(
                "Grammar warnings must refer to the unchanged original input"
            )
        records.append(
            {
                "id": row["id"],
                "input_sha256": hashlib.sha256(row["source"].encode()).hexdigest(),
                **screen_result(grammar),
            }
        )
    return {
        "policy": SCREEN_POLICY,
        "source_sha256": source_sha256,
        "screening_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "input_n": len(rows),
        "retained_n": len(records),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "flagged_rule_counts": dict(
            Counter(rule for r in records for rule in r["error_rules"])
        ),
        "limits": "No rows removed or text rewritten. No-covered-issue inputs can still be ungrammatical. Flags are suggestions for separate candidate-data curation, not certified errors. Full benchmark remains required; subset results must disclose selection and counts. Screening errors are unavailable, never clean.",
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.saved_outputs.read_bytes()
    result = manifest(json.loads(data), hashlib.sha256(data).hexdigest())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
