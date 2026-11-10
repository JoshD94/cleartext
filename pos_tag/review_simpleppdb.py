"""
Review the SimplePPDB phrases that actually appear in real text.

Instead of checking all ~39,000 SimplePPDB phrases, this finds the few that
match in your development text, so you only review those.

Step 1 - collect the matches:
    python review_simpleppdb.py collect eval_dev.csv
    -> writes simpleppdb_review.csv

Step 2 - open simpleppdb_review.csv (Excel, Numbers, ...) and fill in the
    "verdict" column for each row: good or bad. Save it as CSV.

Step 3 - turn the good ones into a verified list:
    python review_simpleppdb.py apply
    -> writes verified_phrases.csv, which phrase_finder.py loads automatically

Input for "collect" can be a CSV with a "sentence" column (like eval_dev.csv)
or a plain .txt file. You can pass several files. Running "collect" again
keeps the verdicts you already filled in.

Only use DEVELOPMENT text here, never eval_test.csv: the test set must stay
unseen so the final scores are honest.
"""

import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVIEW_FILE = HERE / "simpleppdb_review.csv"
VERIFIED_FILE = HERE / "verified_phrases.csv"
FIELDS = ["complex", "simple", "times_found", "example", "verdict"]


def read_sentences(path, nlp):
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            return [row["sentence"] for row in csv.DictReader(f) if row["sentence"].strip()]
    with open(path, encoding="utf-8") as f:
        return [s.text.strip() for s in nlp(f.read()).sents if s.text.strip()]


def collect(paths):
    # Manual phrases still win overlaps, so we only see SimplePPDB phrases
    # that would really be used. Verified is left out so every match is listed.
    os.environ["PHRASE_LISTS"] = "manual,simpleppdb"
    from phrase_finder import find_complex_phrases
    from pos_tag import nlp

    old = {}
    if REVIEW_FILE.exists():
        with open(REVIEW_FILE, newline="", encoding="utf-8") as f:
            old = {r["complex"]: r.get("verdict", "") for r in csv.DictReader(f)}

    found = {}
    for path in paths:
        for sentence in read_sentences(path, nlp):
            for p in find_complex_phrases(sentence):
                if p["source"] != "simpleppdb":
                    continue
                key = p["pattern"]
                if key not in found:
                    found[key] = {"complex": key, "simple": p["suggestion"], "times_found": 0,
                                  "example": sentence, "verdict": old.get(key, "")}
                found[key]["times_found"] += 1

    rows = sorted(found.values(), key=lambda r: (-r["times_found"], r["complex"]))
    with open(REVIEW_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    todo = sum(1 for r in rows if not r["verdict"].strip())
    print(f"\nFound {len(rows)} different SimplePPDB phrases -> {REVIEW_FILE.name}")
    print(f"{todo} still need a verdict (good / bad).")
    for r in rows[:15]:
        print(f'  "{r["complex"]}" -> "{r["simple"]}"  (x{r["times_found"]})')
    if len(rows) > 15:
        print(f"  ... and {len(rows) - 15} more in the file")


def apply():
    if not REVIEW_FILE.exists():
        print(f"{REVIEW_FILE.name} not found. Run 'collect' first.")
        return
    with open(REVIEW_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    good = [r for r in rows if r["verdict"].strip().lower() in ("good", "g", "yes", "y")]
    bad = [r for r in rows if r["verdict"].strip().lower() in ("bad", "b", "no", "n")]
    unmarked = len(rows) - len(good) - len(bad)

    with open(VERIFIED_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["complex", "simple"])
        for r in good:
            writer.writerow([r["complex"], r["simple"]])

    reviewed = len(good) + len(bad)
    print(f"Good: {len(good)}   Bad: {len(bad)}   Not marked: {unmarked}")
    if reviewed:
        print(f"SimplePPDB precision on your development text: {len(good) / reviewed:.1%}")
    print(f"Wrote {len(good)} phrases -> {VERIFIED_FILE.name}")
    if unmarked:
        print("Unmarked rows were left out; mark them and run 'apply' again.")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "collect":
        collect(sys.argv[2:])
    elif len(sys.argv) == 2 and sys.argv[1] == "apply":
        apply()
    else:
        print(__doc__)
