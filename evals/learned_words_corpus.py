"""Round 6 technical reference corpus (November 20, 2026).

  learned-words corpus fetch [--max-seconds N]   download (cached, resumable)
  learned-words corpus count                     n-gram counts -> corpus/counts.pkl

Public documents from the same five technical sources as test-set v3, but
from other years and never a v3 document (dev or test):

  rfc      RFC proposed/internet standards from 2015-2019 (v3: 2020-2024)
  fedreg   significant rules and proposed rules, 2022-01-01..2023-06-30
           (v3: rules 2024-01-01..2025-06-30)
  openfda  prescription labels effective in 2022 (v3: 2024), excluding
           every generic name in the v3 panel or the round-4..6 panels
  sec10k   10-Ks filed 2022-01-01..2023-06-30 by companies 300-399 of the
           SEC ticker file, never a v3 company (skipped and recorded when
           SEC refuses the request: no contact address is sent)

Every corpus sentence that shares a word 8-gram with any v3 passage (dev
or test), or with a round-4..6 panel passage, is dropped; the drop count
is recorded. v3 test text is read only for that exclusion check, as
`panel build` already does. Raw text stays in the state directory.
"""

import gzip
import hashlib
import json
import pickle
import random
import re
import time
import urllib.parse
from collections import Counter
from pathlib import Path

from .learned_words import STATE

CORPUS = STATE / "corpus"
V3_STATE = Path.home() / ".local/state/cleartext/testset-v3-20261116"
SEED = 20261121
CAPS = {"rfc": 300, "fedreg": 400, "openfda": 2000, "sec10k": 60}
WORD = re.compile(r"[a-z]+(?:['’-][a-z]+)*")


def tokens(text):
    return WORD.findall(text.lower().replace("’", "'"))


def _excluded_text():
    panel = json.load(open(V3_STATE / "panel/all.json"))
    extra = []
    for name in ("panel-r4", "panel-r5", "panel-r6"):
        p = STATE / name / "passages.json"
        if p.exists():
            extra += json.load(open(p))
    grams = set()
    for p in panel + extra:
        t = tokens(p["original"])
        grams.update(tuple(t[i:i + 8]) for i in range(len(t) - 7))
    docs = {p["source_doc"] for p in panel + extra}
    generics = {m.group(1).lower() for p in panel if p["domain"] == "health_medical"
                for m in [re.search(r"\(([^)]+)\) prescribing", p["source_title"])] if m}
    ciks = {re.match(r"cik(\d+)", d).group(1) for d in docs if d.startswith("cik")}
    return grams, docs, generics, ciks


# --- pools (document keys), seeded and capped ---------------------------------

def rfc_pool(f):
    import xml.etree.ElementTree as ET

    xml = f.get("https://www.rfc-editor.org/rfc-index.xml")
    ns = {"r": "https://www.rfc-editor.org/rfc-index"}
    pool = []
    for e in ET.fromstring(xml.encode()).findall("r:rfc-entry", ns):
        num = e.findtext("r:doc-id", namespaces=ns)[3:].lstrip("0")
        year = int(e.findtext("r:date/r:year", namespaces=ns))
        status = e.findtext("r:current-status", namespaces=ns)
        if 2015 <= year <= 2019 and status in ("PROPOSED STANDARD", "INTERNET STANDARD"):
            pool.append(f"{num}|{e.findtext('r:title', namespaces=ns)}")
    return pool


def fedreg_pool(f):
    pool = []
    for kind in ("RULE", "PRORULE"):
        url = ("https://www.federalregister.gov/api/v1/documents.json?per_page=1000&order=oldest"
               f"&conditions[type][]={kind}&conditions[significant]=1"
               "&conditions[publication_date][gte]=2022-01-01&conditions[publication_date][lte]=2023-06-30"
               "&fields[]=document_number&fields[]=title&fields[]=raw_text_url&fields[]=publication_date")
        while url:
            page = f.json(url)
            pool += [f"{r['document_number']}|{r['publication_date']}|{r['raw_text_url']}|{r['title']}"
                     for r in page.get("results", []) if r.get("raw_text_url")]
            url = page.get("next_page_url")
    return sorted(set(pool))


def sec_pool(f, ciks):
    tickers = list(f.json("https://www.sec.gov/files/company_tickers.json", 0.2).values())
    pool, seen = [], set()
    for row in tickers[300:400]:
        cik = str(row["cik_str"])
        if cik in seen or cik in ciks:
            continue
        seen.add(cik)
        try:
            sub = f.json(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", 0.2)
        except Exception:
            continue
        r = sub["filings"]["recent"]
        for form, date, acc, doc in zip(r["form"], r["filingDate"], r["accessionNumber"], r["primaryDocument"]):
            if form == "10-K" and "2022-01-01" <= date <= "2023-06-30":
                pool.append(f"{cik}|{acc}|{doc}|{date}|{sub['name']}")
                break
    return sorted(pool)


_FDA = ("https://api.fda.gov/drug/label.json?search=effective_time:[20220101+TO+20221231]+AND+"
        "openfda.product_type:%22HUMAN+PRESCRIPTION+DRUG%22+AND+_exists_:clinical_pharmacology")
_FDA_FIELDS = ("indications_and_usage", "dosage_and_administration", "clinical_pharmacology",
               "warnings_and_cautions", "adverse_reactions", "use_in_specific_populations", "clinical_studies")


def fda_labels(f, generics, deadline):
    """Labels in pages of 100 (openFDA caps skip at 25,000); one per generic name."""
    out, names = [], set()
    for skip in range(0, 25000, 100):
        if len(out) >= CAPS["openfda"] or time.monotonic() > deadline:
            break
        try:
            page = f.json(f"{_FDA}&limit=100&skip={skip}", 0.3)
        except Exception:
            break
        if not page.get("results"):
            break
        for lab in page["results"]:
            gen = [g.lower() for g in lab.get("openfda", {}).get("generic_name", [])]
            if not gen or any(g in generics for g in gen) or gen[0] in names:
                continue
            names.add(gen[0])
            text = " ".join(t for k in _FDA_FIELDS for t in lab.get(k, []))
            out.append((f"fda-{lab['set_id']}", text))
    return out


# --- fetch --------------------------------------------------------------------

def _blocks(B, source, f, key):
    loader = {"rfc": B.rfc_load, "fedreg": B.fedreg_load, "sec10k": B.sec_load}[source]
    if source == "rfc":
        num, title = key.split("|", 1)
        key = f"{num}|{title}"
    loaded = loader(f, key)
    if loaded is None:
        return None
    meta, blocks = loaded
    return meta["source_doc"], " \n".join(blocks)


def fetch(argv):
    from . import build_testset_v3 as B

    max_seconds = float(argv[argv.index("--max-seconds") + 1]) if "--max-seconds" in argv else 1500
    deadline = time.monotonic() + max_seconds
    f = B.Fetcher(CORPUS / "http")
    grams, docs, generics, ciks = _excluded_text()
    CORPUS.mkdir(parents=True, exist_ok=True)
    stats = {}
    for source in ("rfc", "fedreg", "sec10k", "openfda"):
        out = CORPUS / f"{source}.jsonl.gz"
        done = CORPUS / f"{source}.done"
        if done.exists():
            stats[source] = json.loads(done.read_text())
            continue
        if source == "openfda":
            items = fda_labels(f, generics, deadline)
            if time.monotonic() > deadline:
                print("PARTIAL (openfda pages cached; rerun)")
                return
        else:
            try:
                pool = {"rfc": rfc_pool, "fedreg": fedreg_pool}.get(source, lambda f: sec_pool(f, ciks))(f)
            except Exception as e:  # SEC refuses clients without a contact address; none is sent
                stats[source] = {"skipped": f"{type(e).__name__}: {e}"}
                done.write_text(json.dumps(stats[source]))
                print(source, stats[source], flush=True)
                continue
            rng = random.Random(f"{SEED}:{source}")
            rng.shuffle(pool)
            items = []
            for key in pool[:CAPS[source]]:
                if time.monotonic() > deadline:
                    print(f"PARTIAL ({source}: {len(items)} documents cached; rerun)")
                    return
                try:
                    r = _blocks(B, source, f, key)
                except Exception as e:  # one unreachable document does not stop the corpus
                    print(f"skip {source} {key[:40]}: {type(e).__name__}")
                    continue
                if r:
                    items.append(r)
        kept = dropped = 0
        with gzip.open(out, "wt") as w:
            for doc, text in items:
                if doc in docs:
                    continue
                for s in B.sentences(B.clean(text)):
                    t = tokens(s)
                    if len(t) < 4:
                        continue
                    if any(tuple(t[i:i + 8]) in grams for i in range(len(t) - 7)):
                        dropped += 1
                        continue
                    kept += 1
                    w.write(json.dumps({"doc": doc, "s": s}) + "\n")
        stats[source] = {"documents": len(items), "sentences": kept, "dropped_8gram_overlap": dropped,
                         "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        done.write_text(json.dumps(stats[source]))
        print(source, stats[source], flush=True)
    json.dump(stats, open(CORPUS / "manifest.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
    print("COMPLETE")


# --- counts -------------------------------------------------------------------

def count(argv):
    uni, bi, tri = Counter(), Counter(), Counter()
    n = 0
    per_source = {}
    for source in ("rfc", "fedreg", "sec10k", "openfda"):
        if not (CORPUS / f"{source}.jsonl.gz").exists():
            continue
        src_n = 0
        with gzip.open(CORPUS / f"{source}.jsonl.gz", "rt") as r:
            for line in r:
                t = tokens(json.loads(line)["s"])
                uni.update(t)
                bi.update(zip(t, t[1:]))
                tri.update(zip(t, t[1:], t[2:]))
                src_n += len(t)
        per_source[source] = src_n
        n += src_n
    bi = {k: v for k, v in bi.items() if v >= 2}
    tri = {k: v for k, v in tri.items() if v >= 2}
    with open(CORPUS / "counts.pkl", "wb") as w:
        pickle.dump({"tokens": n, "uni": dict(uni), "bi": bi, "tri": tri}, w, protocol=4)
    info = {"tokens": n, "per_source_tokens": per_source, "types": len(uni), "bigrams_ge2": len(bi),
            "trigrams_ge2": len(tri), "sha256": hashlib.sha256((CORPUS / "counts.pkl").read_bytes()).hexdigest()}
    json.dump(info, open(CORPUS / "counts.json", "w"), indent=1)
    print(json.dumps(info, indent=1))
    print("COMPLETE")


def main(argv):
    if argv[:1] == ["fetch"]:
        return fetch(argv[1:])
    if argv[:1] == ["count"]:
        return count(argv[1:])
    raise SystemExit("usage: learned-words corpus fetch|count")
