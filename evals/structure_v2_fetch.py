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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-seconds", type=float, default=1500)
    args = ap.parse_args(argv)
    deadline = time.monotonic() + args.max_seconds
    f = Fetcher(args.out / "http")
    used = used_titles()
    stats = {}
    for name in ("news", "goodwiki", "fedreg"):
        out, done = args.out / f"{name}.jsonl.gz", args.out / f"{name}.done"
        if done.exists():
            stats[name] = json.loads(done.read_text())
            continue
        rng = random.Random(f"{SEED}:{name}")
        docs = []
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
                    print(f"PARTIAL ({name})")
                    return
                got = extracts(f, api, picked[i:i + step], intro=(name == "goodwiki"))
                docs += [(t, got[t]) for t in picked[i:i + step] if t in got]
        if time.monotonic() > deadline:
            print(f"PARTIAL ({name})")
            return
        n = 0
        with gzip.open(out, "wt") as w:
            for doc, paras in docs:
                for para in paras:
                    for s in B.sentences(B.clean(html.unescape(para))):
                        w.write(json.dumps({"doc": doc, "s": s}, ensure_ascii=False) + "\n")
                        n += 1
        stats[name] = {"documents": len(docs), "sentences": n, "titles_listed_excluded": len(used)}
        done.write_text(json.dumps(stats[name]))
        print(name, stats[name], flush=True)
    (args.out / "manifest.json").write_text(json.dumps(stats, indent=1))
    print("COMPLETE")


if __name__ == "__main__":
    main()
