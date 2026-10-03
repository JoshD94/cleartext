"""
Measure how accurate the phrase finder is, using hand-labeled sentences.

Usage:
    python evaluate.py                      # uses eval_sentences.csv
    python evaluate.py my_sentences.csv     # use a different labeled file
    python evaluate.py --all                # also list sentences with no errors

Labeled file format (CSV with two columns):
    sentence,expected
    "Prior to the exam, we studied.",prior to => before
    The cat sat on the mat.,

    - "expected" lists the phrases that SHOULD be detected, each as
      "complex phrase => acceptable replacement | another acceptable one"
    - separate multiple phrases with ";"
    - leave "expected" empty if nothing should be detected

Scores:
    precision           - of the phrases detected, how many were correct
                          (low = too many false alarms)
    recall              - of the phrases that should be detected, how many were
                          (low = too many phrases missed)
    F1                  - a single score combining precision and recall
    suggestion accuracy - of the correctly detected phrases, how many got an
                          acceptable replacement
"""

import csv
import sys
from collections import Counter

from phrase_finder import find_complex_phrases


def parse_expected(text):
    """'a => b | c ; d => e'  ->  {'a': {'b', 'c'}, 'd': {'e'}}"""
    expected = {}
    for item in text.split(";"):
        if "=>" not in item:
            continue
        phrase, replacements = item.split("=>", 1)
        expected[phrase.strip().lower()] = {r.strip().lower() for r in replacements.split("|")}
    return expected


def evaluate(path, show_all=False):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    tp = fp = fn = 0
    good_suggestion = 0
    by_source = Counter()          # (source, "correct"/"wrong") -> count

    print(f"Evaluating {len(rows)} sentences from {path}\n")
    for i, row in enumerate(rows, 1):
        sentence = row["sentence"]
        expected = parse_expected(row.get("expected") or "")
        detected = find_complex_phrases(sentence)

        lines = []
        found = set()
        for p in detected:
            text = p["text"].lower()
            if text in expected:
                tp += 1
                found.add(text)
                by_source[(p["source"], "correct")] += 1
                ok = p["suggestion"].lower() in expected[text]
                good_suggestion += ok
                if not ok:
                    lines.append(f'   WRONG SUGGESTION  "{p["text"]}" -> "{p["suggestion"]}"'
                                 f'  (expected: {" | ".join(sorted(expected[text]))})  [{p["source"]}]')
                elif show_all:
                    lines.append(f'   OK                "{p["text"]}" -> "{p["suggestion"]}"  [{p["source"]}]')
            else:
                fp += 1
                by_source[(p["source"], "wrong")] += 1
                partial = any(text in e or e in text for e in expected)
                label = "PARTIAL MATCH    " if partial else "FALSE ALARM      "
                lines.append(f'   {label} "{p["text"]}" -> "{p["suggestion"]}"  [{p["source"]}]')

        for phrase in expected:
            if phrase not in found:
                fn += 1
                lines.append(f'   MISSED            "{phrase}"')

        if lines:
            print(f"{i}. {sentence}")
            print("\n".join(lines))
            print()
        elif show_all:
            print(f"{i}. {sentence}\n   (all correct)\n")

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    sugg_acc = good_suggestion / tp if tp else 0

    print("=" * 60)
    print(f"Correct detections: {tp}   False alarms: {fp}   Missed: {fn}")
    print(f"Precision:           {precision:6.1%}")
    print(f"Recall:              {recall:6.1%}")
    print(f"F1:                  {f1:6.1%}")
    print(f"Suggestion accuracy: {sugg_acc:6.1%}  ({good_suggestion}/{tp})")
    print("\nBy source:")
    for source in ("manual", "simpleppdb"):
        c, w = by_source[(source, "correct")], by_source[(source, "wrong")]
        if c + w:
            print(f"  {source:<11} {c} correct, {w} wrong  (precision {c / (c + w):.1%})")
        else:
            print(f"  {source:<11} no detections")
    print("\nNote: PARTIAL MATCH means the right area was found but the phrase")
    print("boundaries differ from the label; it counts as both a false alarm and a miss.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    evaluate(args[0] if args else "eval_sentences.csv", show_all="--all" in sys.argv)
