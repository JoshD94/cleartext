#!/usr/bin/env python3
"""Score the dependency compound splitter against WikiSplit reference splits.

WikiSplit pairs each long Wikipedia sentence with a human two-sentence split.
A pair is a labelled conjunction positive when the first reference sentence is
exactly the text before an "and"/"but"/"so"/"yet" and the second reference
sentence ends with exactly the text after it. Its prefix may only repeat the
connector and/or up to six subject words. Every other sentence containing a
connector is unlabelled: a split there is reported, not scored as wrong.

Use `tune.tsv` while developing and `test.tsv` once, for the final numbers.
No LLM, decoder or training is involved. Raw sentences stay outside Git.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.structure.boundary_tagger import BoundaryTagger
from cleartext.structure.common import detokenize, normalize
from cleartext.structure.compound import DependencyCompoundSplitter

CONNECTORS = {"and", "but", "so", "yet"}


def label(complex_tokens, simple1, simple2):
    """Return the connector index for a clean conjunction split, else None."""
    first = [t.lower() for t in simple1 if t != "."]
    second = [t.lower() for t in simple2]
    lowered = [t.lower() for t in complex_tokens]
    for k, token in enumerate(lowered):
        if token not in CONNECTORS or k == 0:
            continue
        left = lowered[:k]
        if left and left[-1] == ",":
            left = left[:-1]
        right = lowered[k + 1:]
        if left != first or len(right) < 3 or second[-len(right):] != right:
            continue
        prefix = second[:len(second) - len(right)]
        if prefix[:1] and prefix[0] in CONNECTORS | {"however", ","}:
            prefix = [t for t in prefix[1:] if t != ","]
        if len(prefix) <= 6:
            return k
    return None


def load(path, limit):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        complex_text, _, simple = line.partition("\t")
        parts = simple.split(" <::::> ")
        if len(parts) != 2:
            continue
        tokens = complex_text.split()
        if not CONNECTORS & {t.lower() for t in tokens}:
            continue
        s1, s2 = parts[0].split(), parts[1].split()
        rows.append({
            "source": detokenize(tokens),
            "reference": detokenize(s1) + " " + detokenize(s2),
            "reference_first": detokenize(s1),
            "connector_index": label(tokens, s1, s2),
        })
        if limit and len(rows) >= limit:
            break
    return rows


def first_sentence(text):
    return re.split(r"(?<=\.)\s+", text, maxsplit=1)[0]


def evaluate(rows, splitter):
    totals = Counter()
    reasons = Counter()
    samples = []
    for row in rows:
        result = splitter.predict(row["source"])
        split = result["output"] != row["source"]
        positive = row["connector_index"] is not None
        group = "positive" if positive else "unlabelled"
        totals[f"{group}_sentences"] += 1
        totals[f"{group}_split"] += split
        for detail in result["boundaries"]:
            reasons[detail["reason"]] += 1
        if positive and split:
            boundary = normalize(first_sentence(result["output"])) == normalize(row["reference_first"])
            exact = normalize(result["output"]) == normalize(row["reference"])
            totals["positive_boundary_correct"] += boundary
            totals["positive_exact"] += exact
        samples.append({**row, "output": result["output"], "split": split,
                        "reasons": [d["reason"] for d in result["boundaries"]]})
    split_positive = totals["positive_split"]
    summary = {
        **totals,
        "positive_recall": totals["positive_boundary_correct"] / max(1, totals["positive_sentences"]),
        "positive_boundary_precision": totals["positive_boundary_correct"] / max(1, split_positive),
        "positive_exact_rate_of_splits": totals["positive_exact"] / max(1, split_positive),
        "unlabelled_split_rate": totals["unlabelled_split"] / max(1, totals["unlabelled_sentences"]),
        "boundary_reasons": dict(reasons.most_common()),
    }
    return summary, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", help="WikiSplit tune.tsv or test.tsv")
    parser.add_argument("--limit", type=int, default=0, help="first N connector sentences")
    parser.add_argument("--guards", choices=["scoped", "strict"], default="scoped")
    parser.add_argument("--tagger", type=Path, help="trained BoundaryTagger (.joblib)")
    parser.add_argument("--threshold", type=float, help="override the tagger threshold")
    parser.add_argument("--samples", type=Path, help="write per-sentence outputs (local only)")
    parser.add_argument("--out", type=Path, help="write the aggregate summary JSON")
    args = parser.parse_args()
    start = time.monotonic()
    rows = load(args.tsv, args.limit)
    tagger = BoundaryTagger.load(args.tagger) if args.tagger else None
    if tagger is not None and args.threshold is not None:
        tagger.threshold = args.threshold
    summary, samples = evaluate(rows, DependencyCompoundSplitter(guards=args.guards, tagger=tagger))
    summary.update(file=Path(args.tsv).name, guards=args.guards, limit=args.limit,
                   tagger=str(args.tagger) if args.tagger else None,
                   threshold=tagger.threshold if tagger else None,
                   seconds=round(time.monotonic() - start, 1))
    if args.samples:
        args.samples.write_text("".join(json.dumps(s) + "\n" for s in samples), encoding="utf-8")
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
