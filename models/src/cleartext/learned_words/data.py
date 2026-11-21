"""Public lexical-simplification data as one instance format.

Every reader parses downloaded files as plain data (no code from them runs).
Files live under ``CLEARTEXT_LEXSIMP_RAW`` (default
``~/.local/state/cleartext/lexsimp-data-20261118/raw``) or, for the sets the
v7 build already pinned, under ``models/data/raw``. Licenses are recorded in
``LICENSES`` and in ``models/docs/LEARNED_WORDS.md``.

An instance is a dict::

    {id, dataset, split, text, start, end, target, group,
     gold: {lowercased substitute: weight}, rated: [(substitute, share, n)]}

``gold`` holds human simplifications (weights are annotator counts where the
set has them, else 1). ``rated`` holds SWORDS substitutes with the share of
annotators that accepted them; it is empty elsewhere.

Splits: ``train`` = BenchLS minus NNSeval's instances (LexMTurk counts
attached where the sentence matches), TSAR-2022 trial and SWORDS dev.
``test`` = TSAR-2022 test, NNSeval and SWORDS test. Nothing in ``test`` is
used for fitting or for choosing a threshold.
"""

import csv
import gzip
import html
import json
import os
import re
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

from ..data import RAW as V7_RAW

RAW = Path(os.environ.get(
    "CLEARTEXT_LEXSIMP_RAW",
    Path.home() / ".local/state/cleartext/lexsimp-data-20261118/raw"))

LICENSES = {
    "tsar2022": "CC BY-NC-SA 4.0 (datasets/ in LaSTUS-TALN-UPF/TSAR-2022-Shared-Task); research, non-commercial",
    "benchls": "CC BY 4.0 (Zenodo 2552393)",
    "nnseval": "no license stated (ghpaetzold.github.io/data); research use, not redistributed",
    "lexmturk": "no license stated (Horn et al. 2014; authors ask users to e-mail them); research use, not redistributed",
    "swords": "CC BY 3.0 US (p-lambda/swords)",
    "cwi2018": "no license file (sheffieldnlp teaching mirror); research use, not redistributed",
    "complex": "CC BY 4.0 (MMU-TDMLab/CompLex)",
    "plainlanguage": "US federal government work, public domain (plainlanguage.gov via web.archive.org)",
    "nist_glossary": "US federal government work, public domain (csrc.nist.gov glossary export)",
    "mesh": "NLM MeSH 2026, free to use under NLM terms; term strings only",
    "uscourts_glossary": "US federal government work, public domain (uscourts.gov/glossary)",
    "wikisplit": "CC BY-SA 4.0 (google-research-datasets/wiki-split)",
}

_SPACE = re.compile(r"\s+")


def norm(text):
    return _SPACE.sub(" ", text.strip().lower())


def _token_span(text, index):
    """Character span of whitespace token ``index`` in a pre-tokenized line."""
    pos = 0
    for i, piece in enumerate(text.split(" ")):
        if i == index:
            return pos, pos + len(piece)
        pos += len(piece) + 1
    raise IndexError(index)


def _find(text, target):
    m = re.search(r"(?<![\w-])" + re.escape(target) + r"(?![\w-])", text)
    if m is None:
        m = re.search(re.escape(target), text, re.I)
    return (m.start(), m.end()) if m else None


def _instance(dataset, split, i, text, span, gold, rated=()):
    start, end = span
    target = text[start:end]
    gold = {k: v for k, v in gold.items() if k and k != target.lower()}
    return {"id": f"{dataset}-{i}", "dataset": dataset, "split": split, "text": text,
            "start": start, "end": end, "target": target, "group": target.lower(),
            "gold": gold, "rated": list(rated)}


def tsar(split):
    rows = []
    path = V7_RAW / f"tsar_{split}.tsv"
    for i, line in enumerate(path.read_text().splitlines()):
        f = line.split("\t")
        span = _find(f[0], f[1])
        if span is None:
            continue
        gold = Counter(x.strip().lower() for x in f[2:] if x.strip())
        rows.append(_instance(f"tsar_{split}", "test" if split == "test" else "train", i, f[0], span, gold))
    return rows


def _ranked_lines(text):
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) < 4:
            continue
        subs = [x.split(":", 1)[1].strip().lower() for x in f[3:] if ":" in x]
        yield f[0], f[1], int(f[2]), subs


def benchls_lines():
    path = V7_RAW / "BenchLS.zip"
    return list(_ranked_lines(zipfile.ZipFile(path).read("BenchLS/BenchLS.txt").decode("utf8")))


def nnseval_lines():
    return list(_ranked_lines((RAW / "nnseval/NNSeval.txt").read_text(encoding="utf8")))


def lexmturk_counts():
    """{(normalized sentence, target): Counter of substitutes} from 50 workers."""
    out = {}
    with open(RAW / "lexmturk/lex.mturk.14/lex.mturk.txt", encoding="latin-1") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            out[(norm(p[0].strip('"')), p[1].strip().lower())] = Counter(
                x.strip().lower() for x in p[2:] if x.strip())
    return out


def _span_or_find(text, target, index):
    try:
        span = _token_span(text, index)
        if text[span[0]:span[1]].lower() == target.lower():
            return span
    except IndexError:
        pass
    return _find(text, target)


def ls_sets():
    """BenchLS (train part, LexMTurk counts attached), NNSeval (test)."""
    nns = nnseval_lines()
    nns_keys = {(norm(s), t.lower()) for s, t, _, _ in nns}
    turk = lexmturk_counts()
    rows = []
    for i, (s, t, idx, subs) in enumerate(benchls_lines()):
        if (norm(s), t.lower()) in nns_keys:
            continue
        span = _span_or_find(s, t, idx)
        if span is None:
            continue
        counts = turk.get((norm(s), t.lower()))
        gold = {w: (counts.get(w, 0) + 1 if counts else 1) for w in subs}
        rows.append(_instance("benchls", "train", i, s, span, gold))
    for i, (s, t, idx, subs) in enumerate(nns):
        span = _span_or_find(s, t, idx)
        if span is None:
            continue
        rows.append(_instance("nnseval", "test", i, s, span, {w: 1 for w in subs}))
    return rows


def _sentence_at(context, offset):
    starts = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+", context)]
    a = max(s for s in starts if s <= offset)
    nxt = [s for s in starts if s > offset]
    b = nxt[0] if nxt else len(context)
    return a, b


def swords(split):
    data = json.load(gzip.open(V7_RAW / f"swords_{split}.json.gz"))
    rated = {}
    for sid, s in data["substitutes"].items():
        labels = data["substitute_labels"][sid]
        word = s["substitute"].strip()
        if not word or not all(ch.isalpha() or ch in " -'" for ch in word):
            continue
        share = sum(x == "TRUE" for x in labels) / len(labels)
        rated.setdefault(s["target_id"], []).append((word, share, len(labels)))
    rows = []
    for i, (tid, t) in enumerate(sorted(data["targets"].items())):
        context = data["contexts"][t["context_id"]]["context"]
        offset, word = t["offset"], t["target"]
        if context[offset:offset + len(word)] != word or " " in word:
            continue
        a, b = _sentence_at(context, offset)
        text = context[a:b].rstrip()
        if not text or len(text) > 600:
            continue
        subs = sorted(rated.get(tid, []))
        gold = {w.lower(): share for w, share, n in subs if share >= 0.5 and n >= 3}
        rows.append(_instance(f"swords_{split}", "train" if split == "dev" else "test", i,
                              text, (offset - a, offset - a + len(word)), gold, subs))
    return rows


def instances():
    rows = ls_sets() + tsar("trial") + tsar("test") + swords("dev") + swords("test")
    test_text = {norm(r["text"]) for r in rows if r["split"] == "test"}
    return [r for r in rows if r["split"] == "test" or norm(r["text"]) not in test_text]


# --- held-out human marks (Joshua's 60 blind swaps) ---------------------------

HUMAN = Path.home() / ".local/state/cleartext/human-rating-20261118"


def joshua60():
    """The 60 swaps Joshua marked on November 18. Held out: evaluation only."""
    key = json.load(open(HUMAN / "key.json"))
    marks = {}
    with open(HUMAN / "cleartext-marks.csv") as f:
        for r in csv.DictReader(f):
            marks[r["key"]] = r["mark"]
    rows = []
    for i, s in enumerate(key["swaps"]):
        span = _find(s["before"], s["word"])
        if span is None:
            continue
        rows.append(dict(_instance("joshua60", "heldout", i, s["before"], span, {}),
                         key=s["key"], old_replacement=s["rep"], mark=marks.get(s["key"], "")))
    return rows


# --- word lists ---------------------------------------------------------------

def _single(word):
    word = word.strip()
    return word if re.fullmatch(r"[A-Za-z][a-z]{2,}", word) else None


@lru_cache(1)
def plain_pairs():
    """{hard word: [plain replacements]} from plainlanguage.gov's
    "Use simple words and phrases" table (single-word entries only)."""
    page = (RAW / "plainlanguage/use-simple-words-phrases.html").read_text(encoding="utf8")
    out = {}
    for row in re.findall(r"<tr>(.*?)</tr>", page, re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
        if len(cells) != 2:
            continue
        hard = re.sub(r"\(.*?\)", "", cells[0]).strip().lower()
        if not re.fullmatch(r"[a-z]{3,}", hard):
            continue
        says = []
        for opt in re.split(r"[,;]| or ", re.sub(r"\(.*?\)", "", cells[1])):
            opt = opt.strip().lower()
            if re.fullmatch(r"[a-z]+( [a-z]+){0,2}", opt) and opt != hard and "omit" not in opt:
                says.append(opt)
        if says:
            out[hard] = says
    return out


@lru_cache(1)
def glossary_terms():
    """{glossary name: set of single-word lowercase terms}."""
    out = {}
    nist = json.loads((RAW / "nist-glossary/glossary-export.json").read_text(encoding="utf-8-sig"))
    entries = nist.get("parentTerms", nist) if isinstance(nist, dict) else nist
    out["nist"] = {w.lower() for e in entries if (w := _single(str(e.get("term", ""))))}
    page = (RAW / "uscourts-glossary/glossary.html").read_text(encoding="utf8")
    out["uscourts"] = {w.lower() for t in re.findall(r"<dt>(.*?)</dt>", page)
                       if (w := _single(html.unescape(re.sub(r"<[^>]+>|\(.*?\)", "", t))))}
    out["mesh"] = mesh_terms()
    return out


def mesh_terms():
    import xml.etree.ElementTree as ET
    terms = set()
    with gzip.open(RAW / "mesh/desc2026.gz") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag == "Term":
                s = el.findtext("String") or ""
                w = _single(s)
                if w:
                    terms.add(w.lower())
                el.clear()
            elif el.tag == "DescriptorRecord":
                el.clear()
    return terms
