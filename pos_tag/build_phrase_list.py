"""
Build a complex-phrase list from SimplePPDB++ for phrase_finder.py.

Run this ONCE after downloading SimplePPDB++. It reads the (very large)
dataset, keeps only good multi-word simplifications, and writes a much
smaller file, simpleppdb_phrases.csv, which phrase_finder.py loads.

Download (about 260 MB, compressed):
    https://github.com/mounicam/lexical_simplification
    -> open the SimplePPDBpp folder -> click simpleppdbpp_xl.tsv.gz -> Download
    (A plain "git clone" will NOT include it unless you have Git LFS.)

Usage:
    python build_phrase_list.py simpleppdbpp_xl.tsv.gz
    python build_phrase_list.py simpleppdbpp_xl.tsv.gz --min-complexity 1.2 --min-ppdb 4.2

Source:
    Maddela & Xu (2018). A Word-Complexity Lexicon and A Neural Readability
    Ranking Model for Lexical Simplification. EMNLP.
    Based on SimplePPDB: Pavlick & Callison-Burch (2016). ACL.

File format (from the dataset's README), one rule per line, tab-separated:
    phrase1  phrase2  complexity_score  ppdb_score
A positive complexity score means phrase1 is more complex than phrase2.
"""

import argparse
import csv
import gzip
import statistics

IGNORABLE = {"a", "an", "the"}

# Forms of "be". If the replacement adds one that the original doesn't have,
# it can't fit in the same spot: "diplomatic relations" -> "diplomacy is",
# "a largely" -> "is mostly".
BE_WORDS = {"is", "are", "was", "were", "be", "been", "being", "am", "'s", "'re", "'m"}


# Pieces of contractions ("I'd" -> "I" + "'d"). Phrases starting or ending
# with one are leftover fragments, not real expressions.
CLITICS = {"'d", "'s", "'re", "'ve", "'ll", "'m", "n't", "'"}

# Common French / Spanish / German / Italian / Portuguese words that leaked
# into the original dataset ("'agriculture et", "'administration des").
FOREIGN_WORDS = {
    "des", "et", "le", "les", "du", "de", "la", "au", "aux", "une", "est",
    "il", "ils", "elle", "pour", "avec", "dans", "sur", "pas", "qui", "que",
    "el", "los", "las", "y", "del", "por", "con", "para", "una", "es",
    "der", "die", "das", "und", "ist", "nicht", "mit", "von", "zu", "ein", "eine",
    "di", "della", "che", "il", "gli", "nel", "da", "e", "do", "dos", "na", "em",
}


def is_clean(phrase):
    """
    True if the phrase looks like a normal English expression:
    - only plain English letters, hyphens and apostrophes (no numbers,
      symbols, accented letters or PPDB tags like [NP])
    - does not start or end with a contraction piece ('d, 're, n't, ...)
    - does not start with an apostrophe ('administration)
    - contains no common foreign-language words
    """
    words = phrase.split()
    if not words:
        return False
    for w in words:
        letters = w.replace("-", "").replace("'", "")
        if not (letters.isascii() and letters.isalpha()) and w not in CLITICS:
            return False
        if w in FOREIGN_WORDS:
            return False
    if words[0] in CLITICS or words[-1] in CLITICS or words[0].startswith("'"):
        return False
    return True


def read_rules(path):
    """Yield (complex, simple, complexity, ppdb) from the dataset, skipping bad lines."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            try:
                p1, p2 = fields[0].strip().lower(), fields[1].strip().lower()
                complexity = float(fields[2])
                ppdb = float(fields[3]) if len(fields) > 3 else None
            except ValueError:
                continue
            # Make sure the complex phrase always comes first
            if complexity < 0:
                p1, p2, complexity = p2, p1, -complexity
            yield p1, p2, complexity, ppdb


def keep(complex_p, simple_p, complexity, ppdb, args):
    """Decide whether a rule is a good phrase-level simplification."""
    c_words, s_words = complex_p.split(), simple_p.split()
    if not (args.min_words <= len(c_words) <= args.max_words):
        return False                      # single words are the teammate's job
    if not s_words or len(s_words) > len(c_words):
        return False                      # replacement shouldn't be longer
    if complexity < args.min_complexity:
        return False                      # not clearly simpler
    if ppdb is not None and ppdb < args.min_ppdb:
        return False                      # meaning may not match
    if not (is_clean(complex_p) and is_clean(simple_p)):
        return False
    if set(c_words) - IGNORABLE == set(s_words) - IGNORABLE:
        return False                      # only dropped "the"/"a"
    if (set(s_words) & BE_WORDS) - set(c_words):
        return False                      # replacement adds "is"/"are"/...: breaks grammar
    if c_words[0] == s_words[0] or c_words[-1] == s_words[-1]:
        return False                      # padded copy of a shorter rule:
                                          # "prior to the" -> "before the",
                                          # "attend the conference" -> "attend"
    c_only, s_only = set(c_words) - set(s_words), set(s_words) - set(c_words)
    if len(c_only) <= 1 and len(s_only) <= 1:
        return False                      # really a one-word swap inside a
                                          # fragment ("of the committee" ->
                                          # "of the panel"): teammate's job
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="path to the SimplePPDB++ file (.tsv, .txt or .gz)")
    parser.add_argument("--out", default="simpleppdb_phrases.csv")
    parser.add_argument("--min-complexity", type=float, default=1.0,
                        help="how much simpler the replacement must be (default 1.0)")
    parser.add_argument("--min-ppdb", type=float, default=4.0,
                        help="minimum paraphrase quality score (default 4.0)")
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=6)
    args = parser.parse_args()

    best = {}              # complex phrase -> (simple, complexity, ppdb)
    total = multiword = 0
    complexity_seen, ppdb_seen = [], []

    print(f"Reading {args.dataset} (this can take several minutes)...")
    for complex_p, simple_p, complexity, ppdb in read_rules(args.dataset):
        total += 1
        if total % 1_000_000 == 0:
            print(f"  {total:,} rules read, {len(best):,} phrases kept so far")
        if len(complex_p.split()) >= args.min_words:
            multiword += 1
            if len(complexity_seen) < 200_000:      # sample for statistics
                complexity_seen.append(complexity)
                if ppdb is not None:
                    ppdb_seen.append(ppdb)

        if not keep(complex_p, simple_p, complexity, ppdb, args):
            continue
        # For each complex phrase, keep the simplest replacement
        # (ties broken by the better paraphrase score)
        rank = (complexity, ppdb if ppdb is not None else 0)
        current = best.get(complex_p)
        if current is None or rank > (current[1], current[2] or 0):
            best[complex_p] = (simple_p, complexity, ppdb)

    if total == 0:
        print("No rules could be read. Check the file path and that it is the "
              "real data file, not a small Git LFS pointer file.")
        return

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["complex", "simple", "complexity", "ppdb"])
        for complex_p, (simple_p, complexity, ppdb) in sorted(best.items()):
            writer.writerow([complex_p, simple_p, round(complexity, 3),
                             "" if ppdb is None else round(ppdb, 3)])

    print(f"\nRead {total:,} rules ({multiword:,} with multi-word phrases).")
    print(f"Kept {len(best):,} phrases -> {args.out}")
    if complexity_seen:
        q = statistics.quantiles(complexity_seen, n=4)
        print(f"Complexity scores (multi-word sample): 25%={q[0]:.2f} 50%={q[1]:.2f} 75%={q[2]:.2f}")
    if ppdb_seen:
        q = statistics.quantiles(ppdb_seen, n=4)
        print(f"PPDB scores (multi-word sample):       25%={q[0]:.2f} 50%={q[1]:.2f} 75%={q[2]:.2f}")
    print("\nTip: open the CSV and skim it. If many pairs look wrong, rerun with a higher "
          "--min-complexity or --min-ppdb. If too few were kept, lower them.")


if __name__ == "__main__":
    main()