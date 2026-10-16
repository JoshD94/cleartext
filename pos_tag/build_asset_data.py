"""
Build training data for the complex-phrase detector from the ASSET dataset.

ASSET has 2,357 sentences, each rewritten in simpler English by 10 different
people. This script finds phrases that several people independently replaced
with shorter wording (e.g. "prior to" -> "before") and turns them into:

  asset_phrases.jsonl        one sentence per line: tokens, B/I/O labels, and
                             the phrases with the replacements people used
  learned_replacements.csv   complex phrase -> replacement, with counts
                             (built from the TRAINING split only)

Get ASSET first (once):
    git clone https://github.com/facebookresearch/asset.git

Usage:
    python build_asset_data.py asset
    python build_asset_data.py asset --min-votes 2      # more, noisier labels

Labels: a word is part of a complex phrase if at least --min-votes of the 10
annotators replaced it (inside a phrase of 2-6 words) with something shorter.
Runs of such words become phrases: B = first word, I = inside, O = other.

Splits: ASSET's validation sentences are shuffled into train (85%) and dev (15%);
ASSET's test sentences become our test split.

Source: Alva-Manchego et al. (2020). ASSET: A Dataset for Tuning and Evaluation
of Sentence Simplification Models with Multiple Rewriting Transformations. ACL.
License: CC BY-NC 4.0.
"""

import argparse
import csv
import difflib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from pos_tag import nlp

DETERMINERS = {"a", "an", "the"}
CONTRACTIONS = {"'s", "n't", "'re", "'ve", "'ll", "'m", "'d"}


def tokens(text):
    return [t.text for t in nlp.make_doc(text.strip())]


def is_good_edit(orig_words, simp_words, orig_doc, a, b):
    """Decide whether replacing orig[a:b] with simp_words is a real phrase simplification."""
    if not (2 <= b - a <= 6):
        return False                                   # phrases only, not single words
    if not (1 <= len(simp_words) < b - a):
        return False                                   # must get shorter, not vanish
    span = orig_doc[a:b]
    for t in span:
        if not t.text.replace("-", "").isalpha():
            return False                               # no punctuation / numbers
        if t.pos_ == "PROPN":
            return False                               # names ("United States" -> "US")
    if any(w in CONTRACTIONS for w in simp_words) or not all(w.replace("-", "").replace("'", "").isalpha() for w in simp_words):
        return False                                   # contractions, sentence splits
    o = [w for w in orig_words[a:b] if w not in DETERMINERS]
    s = [w for w in simp_words if w not in DETERMINERS]
    if not s or set(s) <= set(o):
        return False                                   # just deleted words ("an american" -> "a")
    return True


def label_sentence(orig, simps, min_votes):
    """Return tokens, BIO labels, and phrases with the replacements people used."""
    doc = nlp(orig.strip())
    words = [t.text for t in doc]
    lower = [w.lower() for w in words]
    votes = [0] * len(words)
    edits = []                                         # (a, b, replacement)

    for simp in simps:
        s = [w.lower() for w in tokens(simp)]
        covered = set()
        for tag, a, b, c, d in difflib.SequenceMatcher(None, lower, s, autojunk=False).get_opcodes():
            if tag == "replace" and is_good_edit(lower, s[c:d], doc, a, b):
                covered.update(range(a, b))
                edits.append((a, b, " ".join(s[c:d])))
        for i in covered:                              # one vote per annotator
            votes[i] += 1

    labels = ["O"] * len(words)
    phrases = []
    i = 0
    while i < len(words):
        if votes[i] >= min_votes:
            j = i
            while j < len(words) and votes[j] >= min_votes:
                j += 1
            if j - i >= 2:                             # phrases of 2+ words only
                labels[i] = "B"
                for k in range(i + 1, j):
                    labels[k] = "I"
                # replacements people used for (most of) this phrase
                reps = Counter(r for a, b, r in edits if a >= i and b <= j and b - a >= j - i - 1)
                if not reps:
                    reps = Counter(r for a, b, r in edits if a < j and b > i)
                phrases.append({"start": i, "end": j, "text": " ".join(lower[i:j]),
                                "replacements": dict(reps.most_common())})
            i = j
        else:
            i += 1
    return words, labels, phrases


def read_split(asset_dir, split):
    folder = Path(asset_dir) / "dataset"
    orig = (folder / f"asset.{split}.orig").read_text(encoding="utf-8").splitlines()
    simps = [(folder / f"asset.{split}.simp.{k}").read_text(encoding="utf-8").splitlines()
             for k in range(10)]
    return [(orig[i], [s[i] for s in simps]) for i in range(len(orig))]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("asset_dir", help="folder from 'git clone .../asset.git'")
    parser.add_argument("--min-votes", type=int, default=3,
                        help="annotators (of 10) who must replace a phrase (default 3)")
    parser.add_argument("--out", default="asset_phrases.jsonl")
    args = parser.parse_args()

    valid = read_split(args.asset_dir, "valid")
    test = read_split(args.asset_dir, "test")
    random.seed(13)
    order = list(range(len(valid)))
    random.shuffle(order)
    n_dev = round(len(valid) * 0.15)
    split_of = {i: ("dev" if k < n_dev else "train") for k, i in enumerate(order)}

    items = [(split_of[i], o, s) for i, (o, s) in enumerate(valid)] + [("test", o, s) for o, s in test]
    learned = defaultdict(Counter)
    stats = Counter()

    print(f"Labeling {len(items):,} sentences (min votes = {args.min_votes})...")
    with open(args.out, "w", encoding="utf-8") as f:
        for n, (split, orig, simps) in enumerate(items, 1):
            words, labels, phrases = label_sentence(orig, simps, args.min_votes)
            f.write(json.dumps({"split": split, "tokens": words, "labels": labels,
                                "phrases": phrases}) + "\n")
            stats[f"{split}_sentences"] += 1
            stats[f"{split}_phrases"] += len(phrases)
            if split == "train":
                for p in phrases:
                    learned[p["text"]].update(p["replacements"])
            if n % 500 == 0:
                print(f"  {n:,} done")

    with open("learned_replacements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["complex", "simple", "count", "total"])
        for phrase, reps in sorted(learned.items()):
            if not reps:
                continue
            best, count = reps.most_common(1)[0]
            w.writerow([phrase, best, count, sum(reps.values())])

    print(f"\nWrote {args.out}")
    for split in ("train", "dev", "test"):
        print(f"  {split:<5} {stats[split + '_sentences']:>5,} sentences, "
              f"{stats[split + '_phrases']:>5,} labeled phrases")
    print(f"Wrote learned_replacements.csv ({len(learned):,} phrases from the train split)")


if __name__ == "__main__":
    main()
