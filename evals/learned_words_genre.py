"""Round 21 genre corpora: news and encyclopedic text outside the technical corpus (November 22, 2026).

  learned-words genre fetch [--max-seconds N]   download (cached, resumable), then sentence files

Two public sources, both downloaded as untrusted data into their own directory
(`genre/http/`, an on-disk cache) and only parsed:

  news   English Wikinews articles in Category:Published (CC BY 2.5)
  wiki   English Wikipedia articles in Category:Featured articles (CC BY-SA 4.0)

Titles are listed in full, shuffled with a fixed seed, and the first CAP are fetched as plain
text (TextExtracts, lead section only, 20 titles per request: a Wikinews article body, or a
featured article's lead). Text after a references-type heading is cut. Sentences come from
test-set v3's splitter. Any sentence sharing a word 8-gram with test-set v3's panel file
(read only for that exclusion, as earlier corpus builds did) is dropped. The panel rule in
round 21 adds the per-panel, dev-swap and rewrite-corpus exclusions. Requests carry a generic
client name and no contact or project details. Raw text stays in the state directory.
"""

import gzip
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .learned_words import STATE

GENRE = STATE / "genre"
HTTP = GENRE / "http"
V3 = Path.home() / ".local/state/cleartext/testset-v3-20261116/panel/all.json"
SEED = 20261122
CAP = 220
AGENT = "research-corpus-fetch/0.1"
SITES = {"news": ("https://en.wikinews.org/w/api.php", "Category:Published"),
         "wiki": ("https://en.wikipedia.org/w/api.php", "Category:Featured articles")}
CUT = re.compile(r"^(sources|references|see also|external links|related news|notes|further reading|bibliography|"
                 r"citations|footnotes|works cited|interviews?|sister links|external sources)\s*$", re.I)
WORD = re.compile(r"[a-z]+(?:['’-][a-z]+)*")


def tokens(text):
    return WORD.findall(text.lower().replace("’", "'"))


def _get(url, delay=2.0, last={}):
    key = hashlib.sha256(url.encode()).hexdigest()
    body = HTTP / (key + ".body")
    if body.exists():
        return body.read_text("utf-8", "replace")
    wait = last.get("t", 0) + delay - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept-Encoding": "identity"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:  # 429: honour Retry-After
            if attempt == 5:
                raise
            ra = e.headers.get("Retry-After") if e.headers else None
            time.sleep(min(120, float(ra)) if ra and ra.isdigit() else 20 * (attempt + 1))
        except Exception:
            if attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))
    last["t"] = time.monotonic()
    body.write_bytes(data)
    return data.decode("utf-8", "replace")


def _titles(api, cat, deadline):
    out, cont = [], None
    while True:
        q = {"action": "query", "list": "categorymembers", "cmtitle": cat, "cmnamespace": "0", "cmlimit": "500",
             "cmtype": "page", "format": "json"}
        if cont:
            q["cmcontinue"] = cont
        d = json.loads(_get(api + "?" + urllib.parse.urlencode(q)))
        out += [m["title"] for m in d["query"]["categorymembers"]]
        cont = d.get("continue", {}).get("cmcontinue")
        if not cont:
            return sorted(set(out))
        if time.monotonic() > deadline:
            return None


def _texts(api, titles):
    """{title: (page id, paragraphs)} for up to 20 titles (lead sections, plain text)."""
    q = {"action": "query", "prop": "extracts", "explaintext": "1", "exintro": "1", "exlimit": "20",
         "titles": "|".join(titles), "format": "json"}
    d = json.loads(_get(api + "?" + urllib.parse.urlencode(q)))
    out = {}
    for page in d["query"].get("pages", {}).values():
        keep = []
        for line in (page.get("extract") or "").split("\n"):
            if CUT.match(line.strip()):
                break
            line = line.strip()
            if len(line.split()) >= 6:  # paragraphs only; headings and captions are short
                keep.append(line)
        out[page.get("title")] = (page.get("pageid"), keep)
    return out


def fetch(argv):
    from . import build_testset_v3 as B

    deadline = time.monotonic() + (float(argv[argv.index("--max-seconds") + 1]) if "--max-seconds" in argv else 1500)
    HTTP.mkdir(parents=True, exist_ok=True)
    grams = set()
    for p in json.load(open(V3)):
        t = tokens(p["original"])
        grams.update(tuple(t[i:i + 8]) for i in range(len(t) - 7))
    stats = {}
    for name, (api, cat) in SITES.items():
        out, done = GENRE / f"{name}.jsonl.gz", GENRE / f"{name}.done"
        if done.exists():
            stats[name] = json.loads(done.read_text())
            continue
        titles = _titles(api, cat, deadline)
        if titles is None:
            print(f"PARTIAL ({name} title list cached; rerun)", flush=True)
            return
        order = list(titles)
        random.Random(f"{SEED}:{name}").shuffle(order)
        docs = []
        picked = order[:CAP]
        for i in range(0, len(picked), 20):
            if time.monotonic() > deadline:
                print(f"PARTIAL ({name}: {len(docs)} documents cached; rerun)", flush=True)
                return
            got = _texts(api, picked[i:i + 20])
            for t in picked[i:i + 20]:
                if t in got:
                    docs.append((t, got[t][0], got[t][1]))
        kept = dropped = 0
        with gzip.open(out, "wt") as w:
            for t, rev, paras in docs:
                for para in paras:
                    for s in B.sentences(B.clean(para)):
                        tt = tokens(s)
                        if len(tt) < 4:
                            continue
                        if any(tuple(tt[i:i + 8]) in grams for i in range(len(tt) - 7)):
                            dropped += 1
                            continue
                        kept += 1
                        w.write(json.dumps({"doc": t, "pageid": rev, "s": s}) + "\n")
        stats[name] = {"titles_listed": len(titles), "documents": len(docs), "sentences": kept,
                       "dropped_v3_8gram_overlap": dropped, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        done.write_text(json.dumps(stats[name]))
        print(name, stats[name], flush=True)
    json.dump(stats, open(GENRE / "manifest.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
    print("COMPLETE")


def main(argv):
    if argv[:1] == ["fetch"]:
        return fetch(argv[1:])
    raise SystemExit("usage: learned-words genre fetch [--max-seconds N]")
