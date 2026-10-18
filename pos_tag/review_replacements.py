"""
Judge the tool's actual replacements by hand, and use the judgments to tune it.

Step 1 - run the full pipeline and collect every replacement:
    python review_replacements.py collect eval_dev.csv
    -> writes replacements_review.csv (before / after sentence, scores, verdict)

Step 2 - open replacements_review.csv and fill in "verdict" for each row:
    good  the new sentence is correct, means the same, and is easier to read
    bad   anything else (wrong meaning, broken grammar, not simpler)
    Save it as CSV.

Step 3 - get the results and tune the filters:
    python review_replacements.py apply
    -> prints judged precision for each source, and saves the best cutoffs for
       BERT confidence and fluency drop in replacer_settings.json
       (used by phrase_replacer.py)

Step 4 - check what the tool now actually outputs, with the tuned filters on:
    python review_replacements.py collect --final eval_dev.csv
    -> writes replacements_final.csv; mark any rows without a verdict, then
    python review_replacements.py apply --final
    -> prints precision only (does not change the filters)

Input for "collect" can be CSVs with a "sentence" column or plain .txt files.
Running "collect" again keeps verdicts you already filled in.
Use DEVELOPMENT text only (eval_dev.csv), not eval_test.csv.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVIEW_FILE = HERE / "replacements_review.csv"
FINAL_FILE = HERE / "replacements_final.csv"
FIELDS = ["source", "method", "phrase", "suggestion", "bert", "similarity",
          "fluency_drop", "entailment", "bert_rank", "rejected_table", "before", "after", "verdict"]


def read_sentences(path, nlp):
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            return [r["sentence"] for r in csv.DictReader(f) if r["sentence"].strip()]
    with open(path, encoding="utf-8") as f:
        return [s.text.strip() for s in nlp(f.read()).sents if s.text.strip()]


def apply_one(sentence, p):
    new = p["suggestion"]
    if p["sentence_start"]:
        new = new[0].upper() + new[1:]
    return sentence[:p["start_char"]] + new + sentence[p["end_char"]:]


def collect(paths, final=False):
    import phrase_replacer
    out_file = FINAL_FILE if final else REVIEW_FILE
    if not final:
        phrase_replacer.MIN_BERT_SCORE = None      # see everything, so the cutoffs can be chosen
        phrase_replacer.MAX_FLUENCY_DROP = None
        phrase_replacer.MIN_ENTAILMENT = None
    from phrase_finder import find_complex_phrases
    from pos_tag import nlp

    # Reuse verdicts from both files, so nothing has to be judged twice
    old = {}
    for path in (REVIEW_FILE, FINAL_FILE):
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["verdict"].strip():
                        old[(r["before"], r["phrase"], r["suggestion"])] = r["verdict"]

    rows = []
    for path in paths:
        for sentence in read_sentences(path, nlp):
            for p in find_complex_phrases(sentence):
                d = phrase_replacer.DETAILS.get((sentence, p["start_index"], p["end_index"]), {})
                method = d.get("method", p["source"]) if p["source"] == "model" else p["source"]
                rows.append({
                    "source": p["source"], "method": method,
                    "phrase": p["text"], "suggestion": p["suggestion"],
                    "bert": d.get("bert", ""), "similarity": d.get("similarity", ""),
                    "fluency_drop": d.get("fluency_drop", ""),
                    "entailment": d.get("entailment", ""),
                    "bert_rank": d.get("bert_rank", ""),
                    "rejected_table": d.get("rejected_table", ""),
                    "before": sentence, "after": apply_one(sentence, p),
                    "verdict": old.get((sentence, p["text"], p["suggestion"]), ""),
                })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} replacements -> {out_file.name}"
          + ("  (filters ON: what the tool really outputs)" if final else "  (filters OFF)"))
    print(f"By method: {dict(Counter(r['method'] for r in rows))}")
    blocked = [d["rejected_table"] for d in phrase_replacer.DETAILS.values() if d.get("rejected_table")]
    if blocked:
        print(f"Learned-table suggestions rejected: {blocked}")
    print(f"{sum(1 for r in rows if not r['verdict'])} need a verdict (good / bad).")
    for r in rows[:10]:
        print(f'  "{r["phrase"]}" -> "{r["suggestion"]}"  [{r["method"]}]')


def verdict(r):
    v = r["verdict"].strip().lower()
    return True if v in ("good", "g", "yes", "y") else False if v in ("bad", "b", "no", "n") else None


def apply(final=False):
    in_file = FINAL_FILE if final else REVIEW_FILE
    if not in_file.exists():
        print(f"{in_file.name} not found. Run 'collect{' --final' if final else ''}' first.")
        return
    with open(in_file, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    judged = [r for r in rows if verdict(r) is not None]
    print(f"{len(judged)} of {len(rows)} replacements judged.\n")

    print("Judged precision (share of replacements marked good):")
    by = defaultdict(list)
    for r in judged:
        by[r["method"]].append(verdict(r))
    for method, vs in sorted(by.items()):
        print(f"  {method:<10} {sum(vs)}/{len(vs)} good  ({sum(vs) / len(vs):.0%})")
    allv = [verdict(r) for r in judged]
    if allv:
        print(f"  {'overall':<10} {sum(allv)}/{len(allv)} good  ({sum(allv) / len(allv):.0%})")
    if final:
        unjudged = len(rows) - len(judged)
        if unjudged:
            print(f"\n{unjudged} rows still need a verdict; mark them for a complete number.")
        print("\n(--final: filters were not changed.)")
        return

    # BERT confidence is shown for reference only. On the development text it did
    # not separate good from bad replacements, and tuning a cutoff for it on top of
    # the fluency cutoff removed good replacements, so it is never used as a filter.
    choose_cutoff(judged, "bert", keep_high=True,
                  title="BERT confidence (for reference only; not used as a filter)")
    settings = {
        "min_bert_score": None,
        "max_fluency_drop": choose_cutoff(
            judged, "fluency_drop", keep_high=False,
            title="fluency drop (higher = the rewrite reads less naturally)"),
    }
    # The meaning check is tuned on the replacements the fluency cutoff keeps,
    # because in the tool it only ever sees those.
    fd = settings["max_fluency_drop"]
    survivors = [r for r in judged if fd is None or r["fluency_drop"] == ""
                 or float(r["fluency_drop"]) <= fd]
    settings["min_entailment"] = choose_cutoff(
        survivors, "entailment", keep_high=True,
        title="entailment, among replacements that pass the fluency check "
              "(higher = more surely means the same)")
    (HERE / "replacer_settings.json").write_text(json.dumps(settings))
    print("\nSaved to replacer_settings.json (phrase_replacer.py reads it automatically):")
    print(f"  {settings}")
    if len(judged) < 20:
        print(f"\nNote: only {len(judged)} judged replacements, so these cutoffs are rough "
              "estimates. Collect from more development text to make them more reliable.")


def choose_cutoff(judged, column, keep_high, title):
    """
    Find the cutoff on one score that makes the most right decisions
    (keeps good replacements, rejects bad ones). Returns None if no cutoff helps.
    """
    items = sorted(((float(r[column]), verdict(r), r) for r in judged if r.get(column, "") != ""),
                   key=lambda x: x[0], reverse=not keep_high)
    print(f"\nReplacements by {title}:")
    if not items:
        print("  (no scores recorded)")
        return None
    for value, ok, r in items:
        print(f"  {value:7.3f}  {'good' if ok else 'bad ':<4}  [{r['method']:<7}] "
              f"\"{r['phrase']}\" -> \"{r['suggestion']}\"")

    def keeps(value, cutoff):
        return cutoff is None or (value >= cutoff if keep_high else value <= cutoff)

    def right(cutoff):
        return sum(keeps(v, cutoff) == ok for v, ok, _ in items)

    # Ties go to the loosest cutoff (None first, then the one that keeps the most)
    options = [None] + [v for v, _, _ in items]
    best = max(options, key=lambda c: (right(c), c is None,
                                       sum(keeps(v, c) for v, _, _ in items)))
    good_total = sum(ok for _, ok, _ in items)
    print(f"  no cutoff: keeps {len(items)}, of which {good_total} good")
    if best is None:
        print("  -> no cutoff helps; none set.")
        return None
    kept = [ok for v, ok, _ in items if keeps(v, best)]
    # Put the cutoff in the middle of the gap to the next value it rejects,
    # so a slightly different good replacement isn't rejected by accident.
    rejected = [v for v, _, _ in items if not keeps(v, best)]
    if rejected:
        nearest = max(rejected) if keep_high else min(rejected)
        best = round((best + nearest) / 2, 3)
    word = "minimum" if keep_high else "maximum"
    print(f"  -> {word} {best:.3f}: keeps {len(kept)}, of which {sum(kept)} good")
    return best


if __name__ == "__main__":
    args = sys.argv[1:]
    final = "--final" in args
    args = [a for a in args if a != "--final"]
    if len(args) >= 2 and args[0] == "collect":
        collect(args[1:], final)
    elif args == ["apply"]:
        apply(final)
    else:
        print(__doc__)