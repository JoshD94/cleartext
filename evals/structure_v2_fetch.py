"""Fresh public text for the structure v2 rating panels (November 22, 2026).

    python -I -m evals.structure_v2_fetch --out DIR [--max-seconds N]

Three sources no earlier ClearText panel drew from, downloaded as untrusted
data into DIR (an on-disk cache plus sentence files) and only parsed:

  news     English Wikinews, Category:Published (CC BY 2.5), titles the
           word-stage genre corpus did not take
  goodwiki English Wikipedia, Category:Good articles (CC BY-SA 4.0), lead
           sections; the word stage used Featured articles
  fedreg   Federal Register significant rules, 2023-07-01..2023-12-31 (public
           domain); the word stage used 2022-01..2023-06, test set v3 2024-25

Requests carry a generic client name and no contact or project details.
Raw text stays in DIR (outside the checkout).
"""

import argparse
import gzip
import html
import json
import random
import re
import time
from pathlib import Path

from . import build_testset_v3 as B

# A generic client name: no contact address or project link is sent.
B.USER_AGENT = "research-corpus-fetch/0.1"

SEED = 20261122
CAPS = {"news": 150, "goodwiki": 260, "fedreg": 60}
APIS = {"news": ("https://en.wikinews.org/w/api.php", "Category:Published"),
        "goodwiki": ("https://en.wikipedia.org/w/api.php", "Category:Good articles")}
CUT = re.compile(r"^(sources|references|see also|external links|related news|notes|further reading|bibliography|"
                 r"citations|footnotes|works cited|interviews?|sister links|external sources)\s*$", re.I)
WORDS_GENRE = Path.home() / ".local/state/cleartext/learned-words-20261118/genre"


class Fetcher(B.Fetcher):
    """build_testset_v3.Fetcher with a slower pace and Retry-After on 429."""

    def get(self, url, delay=3.0):
        import hashlib
        import urllib.error
        key = hashlib.sha256(url.encode()).hexdigest()
        if (self.cache / (key + ".body")).exists():
            return super().get(url, delay)
        for attempt in range(8):
            try:
                return super().get(url, max(delay, 3.0))
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 7:
                    raise
                ra = e.headers.get("Retry-After") if e.headers else None
                time.sleep(min(180, float(ra)) if ra and ra.isdigit() else 30 * (attempt + 1))


def used_titles():
    used = set()
    for name in ("news", "wiki"):
        path = WORDS_GENRE / f"{name}.jsonl.gz"
        if path.exists():
            used |= {json.loads(line)["doc"] for line in gzip.open(path, "rt")}
    return used


def titles(f, api, cat, deadline):
    out, cont = [], None
    while time.monotonic() < deadline:
        q = {"action": "query", "list": "categorymembers", "cmtitle": cat, "cmnamespace": "0", "cmlimit": "500",
             "cmtype": "page", "format": "json"}
        if cont:
            q["cmcontinue"] = cont
        d = f.json(api + "?" + B.urllib.parse.urlencode(q), delay=3.0)
        out += [m["title"] for m in d["query"]["categorymembers"]]
        cont = d.get("continue", {}).get("cmcontinue")
        if not cont:
            return sorted(set(out))
    return None


def extracts(f, api, chunk, intro):
    q = {"action": "query", "prop": "extracts", "explaintext": "1", "exlimit": "20",
         "titles": "|".join(chunk), "format": "json"}
    if intro:
        q["exintro"] = "1"
    d = f.json(api + "?" + B.urllib.parse.urlencode(q), delay=3.0)
    out = {}
    for page in d["query"].get("pages", {}).values():
        keep = []
        for line in (page.get("extract") or "").split("\n"):
            if CUT.match(line.strip()):
                break
            if len(line.split()) >= 6:
                keep.append(line.strip())
        out[page.get("title")] = keep
    return out


def fedreg_keys(f):
    keys = []
    url = ("https://www.federalregister.gov/api/v1/documents.json?per_page=1000&order=oldest"
           "&conditions[type][]=RULE&conditions[significant]=1"
           "&conditions[publication_date][gte]=2023-07-01&conditions[publication_date][lte]=2023-12-31"
           "&fields[]=document_number&fields[]=title&fields[]=raw_text_url&fields[]=publication_date")
    while url:
        page = f.json(url, delay=1.0)
        keys += [f"{r['document_number']}|{r['publication_date']}|{r['raw_text_url']}|{r['title']}"
                 for r in page.get("results", []) if r.get("raw_text_url")]
        url = page.get("next_page_url")
    return sorted(set(keys))


# Round 3: a third fresh window, disjoint from rounds 1-2 and the word stage.
FDA_2023 = ("https://api.fda.gov/drug/label.json?search=effective_time:[20230101+TO+20231231]+AND+"
            "openfda.product_type:%22HUMAN+PRESCRIPTION+DRUG%22+AND+_exists_:clinical_pharmacology")
FDA_FIELDS = ("indications_and_usage", "dosage_and_administration", "clinical_pharmacology",
              "warnings_and_cautions", "adverse_reactions", "use_in_specific_populations", "clinical_studies")
R3_CAPS = {"news": 150, "goodwiki": 300, "fedreg": 40, "openfda": 150}
WORDS_CORPUS = Path.home() / ".local/state/cleartext/learned-words-20261118/corpus"


def used_generics():
    """Generic names in every earlier openFDA panel source (word-stage corpus, test set v3)."""
    out = set()
    path = WORDS_CORPUS / "openfda.jsonl.gz"
    v3 = Path.home() / ".local/state/cleartext/testset-v3-20261116/panel/all.json"
    for p in json.loads(v3.read_text()):
        m = re.search(r"\(([^)]+)\) prescribing information", p.get("source_title", ""))
        if m:
            out.add(m.group(1).lower())
    return out


WINDOWS = {3: {"fedreg": ("2021-07-01", "2021-12-31"), "openfda": "[20230101+TO+20231231]"},
           4: {"fedreg": ("2021-01-01", "2021-06-30"), "openfda": "[20210101+TO+20211231]"},
           5: {"fedreg": ("2020-07-01", "2020-12-31"), "openfda": "[20200101+TO+20201231]"},
           # Structure v3 (rounds 6-7, November 23): two more disjoint windows.
           6: {"fedreg": ("2020-01-01", "2020-06-30"), "openfda": "[20190101+TO+20191231]"},
           7: {"fedreg": ("2019-07-01", "2019-12-31"), "openfda": "[20180101+TO+20181231]"},
           # Structure v4 (rounds 8-9, November 23): more news and encyclopedic text, fewer rules.
           8: {"fedreg": ("2019-01-01", "2019-06-30"), "openfda": "[20170101+TO+20171231]"},
           9: {"fedreg": ("2018-07-01", "2018-12-31"), "openfda": "[20160101+TO+20161231]"},
           # Round 8 nominalization items (Federal Register only: the op fires on ~0.03% of sentences).
           10: {"fedreg": ("2018-01-01", "2018-06-30")},
           # Round 9 nominalization items: 18 months of Federal Register rules.
           11: {"fedreg": ("2016-07-01", "2017-12-31")},
           # Round 10 (passive_v7, November 23): one more disjoint window of every source.
           12: {"fedreg": ("2016-01-01", "2016-06-30"), "openfda": "[20150101+TO+20151231]"}}
# Rounds 8-9 take more Wikinews and Good-article documents (v3's pass leaned on Federal Register text).
R8_CAPS = {"news": 400, "goodwiki": 800, "fedreg": 20, "openfda": 150}
R10_CAPS = {"fedreg": 120}
R11_CAPS = {"fedreg": 400}


def round3(args, deadline):
    """Rounds 3 and 4: new titles and date windows, disjoint from every earlier window."""
    f = Fetcher(args.out / "http")
    previous = used_titles()
    olds = [Path(p) for p in args.previous.split(",")]
    for d in olds:
        for old in (d / "news.jsonl.gz", d / "goodwiki.jsonl.gz"):
            if old.exists():
                previous |= {json.loads(line)["doc"] for line in gzip.open(old, "rt")}
    old_fda = set()
    for path in [WORDS_CORPUS / "openfda.jsonl.gz", *(d / "openfda.jsonl.gz" for d in olds)]:
        if path.exists():
            old_fda |= {json.loads(line)["doc"] for line in gzip.open(path, "rt")}
    window = WINDOWS[args.round]
    fda_url = FDA_2023.replace("[20230101+TO+20231231]", window.get("openfda", "[20230101+TO+20231231]"))
    generics = used_generics()
    caps = R11_CAPS if args.round == 11 else R10_CAPS if args.round == 10 else R8_CAPS if args.round >= 8 else R3_CAPS
    stats = {}
    for name in ("goodwiki", "fedreg", "openfda", "news"):
        if name not in caps:
            continue
        out, done = args.out / f"{name}.jsonl.gz", args.out / f"{name}.done"
        if done.exists():
            stats[name] = json.loads(done.read_text())
            continue
        rng = random.Random(f"{SEED}:r{args.round}:{name}")
        docs, partial = [], False
        if name == "fedreg":
            keys = []
            url = ("https://www.federalregister.gov/api/v1/documents.json?per_page=1000&order=oldest"
                   "&conditions[type][]=RULE&conditions[significant]=1"
                   f"&conditions[publication_date][gte]={window['fedreg'][0]}"
                   f"&conditions[publication_date][lte]={window['fedreg'][1]}"
                   "&fields[]=document_number&fields[]=title&fields[]=raw_text_url&fields[]=publication_date")
            while url:
                page = f.json(url, delay=1.0)
                keys += [f"{r['document_number']}|{r['publication_date']}|{r['raw_text_url']}|{r['title']}"
                         for r in page.get("results", []) if r.get("raw_text_url")]
                url = page.get("next_page_url")
            keys = sorted(set(keys))
            rng.shuffle(keys)
            for key in keys:
                if len(docs) >= caps[name] or time.monotonic() > deadline:
                    break
                loaded = B.fedreg_load(f, key)
                if loaded:
                    docs.append((loaded[0]["source_doc"], [b for b in loaded[1] if len(b.split()) >= 6]))
        elif name == "openfda":
            names = set()
            for skip in range(0, 5000, 100):
                if len(docs) >= caps[name] or time.monotonic() > deadline:
                    break
                try:
                    page = f.json(f"{fda_url}&limit=100&skip={skip}", delay=0.5)
                except B.urllib.error.HTTPError as e:
                    if e.code == 404:
                        break  # past the last label of a small window (round 7: 2018)
                    raise
                for lab in page.get("results", []):
                    gen = [g.lower() for g in lab.get("openfda", {}).get("generic_name", [])]
                    doc = f"fda-{lab['set_id']}"
                    if not gen or gen[0] in names or any(g in generics for g in gen) or doc in old_fda:
                        continue
                    names.add(gen[0])
                    paras = []
                    for k in FDA_FIELDS:
                        for t in lab.get(k, []):
                            t = B._FDA_HEADINGS.sub(" 99 ", B.clean(t))
                            paras += [x for x in re.split(r"(?:^|\s)\d{1,2}(?:\.\d{1,2})?\s+(?=[A-Z])", t)
                                      if len(x.split()) >= 6]
                    docs.append((doc, paras))
        else:
            api, cat = APIS[name]
            listed = titles(f, api, cat, deadline)
            if listed is None:
                print(f"PARTIAL ({name} titles)")
                return
            order = [t for t in listed if t not in previous]
            rng.shuffle(order)
            picked = order[: caps[name]]
            step = 20 if name == "goodwiki" else 1
            for i in range(0, len(picked), step):
                if time.monotonic() > deadline:
                    partial = True
                    break
                got = extracts(f, api, picked[i:i + step], intro=(name == "goodwiki"))
                docs += [(t, got[t]) for t in picked[i:i + step] if t in got]
        n = 0
        with gzip.open(out, "wt") as w:
            for doc, paras in docs:
                for para in paras:
                    for sent in B.sentences(B.clean(html.unescape(para))):
                        w.write(json.dumps({"doc": doc, "s": sent}, ensure_ascii=False) + "\n")
                        n += 1
        stats[name] = {"documents": len(docs), "sentences": n, "stopped_at_deadline": partial}
        done.write_text(json.dumps(stats[name]))
        print(name, stats[name], flush=True)
    (args.out / "manifest.json").write_text(json.dumps(stats, indent=1))
    print("COMPLETE")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-seconds", type=float, default=1500)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--previous", default=str(Path.home() / ".local/state/cleartext/structure-20261122/download"))
    args = ap.parse_args(argv)
    deadline = time.monotonic() + args.max_seconds
    if args.round in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
        args.out.mkdir(parents=True, exist_ok=True)
        return round3(args, deadline)
    f = Fetcher(args.out / "http")
    used = used_titles()
    stats = {}
    for name in ("news", "goodwiki", "fedreg"):
        out, done = args.out / f"{name}.jsonl.gz", args.out / f"{name}.done"
        if done.exists():
            stats[name] = json.loads(done.read_text())
            continue
        rng = random.Random(f"{SEED}:{name}")
        docs, partial = [], False
        if name == "fedreg":
            keys = fedreg_keys(f)
            rng.shuffle(keys)
            for key in keys:
                if len(docs) >= CAPS[name] or time.monotonic() > deadline:
                    break
                loaded = B.fedreg_load(f, key)
                if loaded is None:
                    continue
                meta, blocks = loaded
                docs.append((meta["source_doc"], [b for b in blocks if len(b.split()) >= 6]))
        else:
            api, cat = APIS[name]
            listed = titles(f, api, cat, deadline)
            if listed is None:
                print(f"PARTIAL ({name} titles)")
                return
            order = [t for t in listed if t not in used]
            rng.shuffle(order)
            picked = order[: CAPS[name]]
            # Full-page extracts come one page per request; lead sections 20.
            step = 20 if name == "goodwiki" else 1
            for i in range(0, len(picked), step):
                if time.monotonic() > deadline:
                    if name == "news" and docs:
                        partial = True  # rate-limited: keep what was fetched
                        break
                    print(f"PARTIAL ({name})")
                    return
                got = extracts(f, api, picked[i:i + step], intro=(name == "goodwiki"))
                docs += [(t, got[t]) for t in picked[i:i + step] if t in got]
        if time.monotonic() > deadline and not partial:
            print(f"PARTIAL ({name})")
            return
        n = 0
        with gzip.open(out, "wt") as w:
            for doc, paras in docs:
                for para in paras:
                    for s in B.sentences(B.clean(html.unescape(para))):
                        w.write(json.dumps({"doc": doc, "s": s}, ensure_ascii=False) + "\n")
                        n += 1
        stats[name] = {"documents": len(docs), "sentences": n, "titles_listed_excluded": len(used),
                       "stopped_at_deadline": partial}
        done.write_text(json.dumps(stats[name]))
        print(name, stats[name], flush=True)
    (args.out / "manifest.json").write_text(json.dumps(stats, indent=1))
    print("COMPLETE")


if __name__ == "__main__":
    main()
