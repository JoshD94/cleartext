"""Build the test-set v3 panel: ~300 sourced passages with a locked test split.

No LLM writes or chooses text. Every source has a rule-defined document pool
(an API query, an index filter or a category listing), sorted and sampled with
a fixed seed. Windows are 60-120 words of consecutive sentences inside one
cleaned paragraph; figures, tables, headings and page furniture are hard
breaks. A document contributes its first non-overlapping windows from a seeded
shuffle of its candidates; a document with too few is skipped for the next.

The split is by document (no document has passages in both), stratified by
domain: one third of each domain's passages are dev, the rest are test.

Raw downloads and passage text stay in the local state directory. Git gets
this script, the manifest (IDs, sources, licences, spans, text SHA-256, no
text) and the split ID lists.

    python -m evals.build_testset_v3 build --state DIR   # fetch (cached) + select + split
    python -m evals.build_testset_v3 verify --state DIR  # local panel matches the manifest
"""

import argparse
import hashlib
import html
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evals/data/testset_v3"
SEED = 20261116
MIN_WORDS, MAX_WORDS = 60, 120
# SEC EDGAR asks automated clients to name a contact; set CLEARTEXT_CONTACT to one.
USER_AGENT = f"ClearText-research/0.3 ({os.environ.get('CLEARTEXT_CONTACT', 'https://github.com/JoshD94/cleartext')})"

# domain: (source, documents, windows per document, kind, id prefix)
PLAN = {
    "computer_science": ("rfc", 21, 2, "standard", "cs"),
    "corporate_communication": ("sec10k", 21, 2, "standard", "corp"),
    "political_science": ("speech", 21, 2, "standard", "pol"),
    "health_medical": ("openfda", 21, 2, "standard", "health"),
    "legal_government": ("fedreg", 21, 2, "standard", "legal"),
    "science": ("arxiv", 42, 1, "standard", "sci"),
    "simple_english": ("simplewiki", 48, 1, "simple_control", "simple"),
}
DEV_FRACTION = 1 / 3

LICENSES = {
    "rfc": "IETF Trust Legal Provisions (RFC text may be reproduced with attribution); text kept local",
    "sec10k": "Public SEC EDGAR filing (17 CFR 232); text kept local, not redistributed",
    "speech": "US Government work (public domain); transcript via The American Presidency Project",
    "openfda": "openFDA drug label data, CC0 1.0 (https://open.fda.gov/license/)",
    "fedreg": "US Government work (public domain), Federal Register",
    "arxiv": "CC BY 4.0 (per-record arXiv licence field; only CC BY 4.0 records are eligible)",
    "simplewiki": "CC BY-SA 4.0, Simple English Wikipedia (revision pinned)",
}

# The week 7 sourced panel's documents are excluded, so no passage is reused.
EXCLUDED = {"rfc": {"8446", "9000", "9110"}, "sec10k": {"320193", "1018724", "789019"},
            "speech": {"/documents/inaugural-address-53", "/documents/inaugural-address-52",
                       "/documents/address-before-joint-session-the-congress-the-state-the-union-30"}}


# --- fetching (cached) --------------------------------------------------------

class Fetcher:
    """GET with an on-disk cache, so a rebuild reads the same bytes."""

    def __init__(self, cache, offline=False):
        self.cache, self.offline, self.last = Path(cache), offline, {}
        self.cache.mkdir(parents=True, exist_ok=True)

    def get(self, url, delay=0.5):
        key = hashlib.sha256(url.encode()).hexdigest()
        body, meta = self.cache / (key + ".body"), self.cache / (key + ".json")
        if body.exists():
            return body.read_bytes().decode("utf-8", "replace")
        if self.offline:
            raise RuntimeError(f"not cached (offline): {url}")
        host = urllib.parse.urlparse(url).netloc
        wait = self.last.get(host, 0) + delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(url.replace("[", "%5B").replace("]", "%5D"),
                                         headers={"User-Agent": USER_AGENT})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        self.last[host] = time.monotonic()
        body.write_bytes(data)
        meta.write_text(json.dumps({"url": url, "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    "sha256": hashlib.sha256(data).hexdigest()}))
        return data.decode("utf-8", "replace")

    def json(self, url, delay=0.5):
        return json.loads(self.get(url, delay))


# --- text helpers -------------------------------------------------------------

_ABBREV = {"u.s", "e.g", "i.e", "dr", "mr", "mrs", "ms", "inc", "no", "nos", "st", "vs", "fig", "al", "co",
           "corp", "ltd", "jr", "sr", "sec", "secs", "approx", "etc", "u.s.c", "u.k", "jan", "feb", "mar",
           "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "p", "pp", "ref", "refs", "cf",
           "eq", "vol", "ch", "art", "para", "op", "gen", "gov", "rep", "sen", "a.m", "p.m", "n.a", "l.l.c"}
_BREAK = re.compile(r"[.!?][\"'”’)\]]*\s+")


def sentences(text):
    """Sentence split: a terminator plus whitespace before an uppercase letter,
    digit or opening quote, except after a known abbreviation or an initial."""
    out, start = [], 0
    for m in _BREAK.finditer(text):
        nxt = text[m.end():m.end() + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in "\"'“‘("):
            continue
        word = text[start:m.start() + 1].split()[-1].rstrip(".").lower().lstrip("(\"'“")
        if text[m.start()] == "." and (word in _ABBREV or re.fullmatch(r"[a-z]", word)):
            continue
        out.append(text[start:m.end()].strip())
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


def join_lines(lines):
    """Join wrapped lines; a line ending in a hyphen continues the same word."""
    out = ""
    for line in lines:
        out = out + line if out.endswith("-") else (out + " " + line if out else line)
    return clean(out)


def clean(text):
    text = html.unescape(text).replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def word_count(text):
    return len(text.split())


_BAD = re.compile(r"https?:|www\.|@|\*|\[|\]|\\|\||[{}<>$^~_]|\.\.\.\.|\t")


def acceptable(text, source):
    """Window filter: prose only, no markup or cross-reference debris."""
    if not MIN_WORDS <= word_count(text) <= MAX_WORDS:
        return False
    if _BAD.search(text) or not re.search(r"[.!?][\"'”’)]*$", text):
        return False
    if not (text[0].isupper() or text[0] in "\"“"):
        return False
    if text.count("(") != text.count(")") or text.count("\"") % 2 or text.count("“") != text.count("”"):
        return False
    tokens = text.split()
    if sum(any(c.isalpha() for c in t) for t in tokens) / len(tokens) < 0.85:
        return False
    if sum(t.isupper() and len(t) > 1 for t in tokens) / len(tokens) > 0.2:
        return False
    if source == "speech" and re.search(r"\bThank you\b|\bGod bless\b|Speaker,|Vice President,|"
                                        r"Members of (the )?Congress|ladies and gentlemen", text, re.I):
        return False
    if source == "openfda" and re.search(r"\(\s*\d+(\.\d+)*\s*\)|see (Warnings|Clinical|Dosage|Use)|"
                                         r"\d\s+-\d|\d -[a-z]|\bx\s*10\b|\s:", text):
        return False
    if source == "openfda" and any(re.match(r"[A-Z][\w-]+( (Inhibitors|Inducers|Substrates))? (No|The|In|A|An|When|There|Following|Co-?administration|"
                                            r"Concomitant)\b", x) or re.match(r"([A-Z][\w-]*\s+){3}|[^.:]{1,40}:\s", x)
                                   for x in sentences(text)):
        return False  # a run-in drug-name subheading ("Lithium No relevant ...")
    if re.search(r"[a-z]- [a-z]", text):  # a word broken across a line or cell
        return False
    return all(word_count(s) >= 4 for s in sentences(text))


def windows(blocks, source):
    """Every acceptable window: (block index, first sentence, last sentence, text)."""
    found = []
    for b, block in enumerate(blocks):
        sents = sentences(block)
        for i in range(len(sents)):
            for j in range(i, len(sents)):
                text = " ".join(sents[i:j + 1])
                if word_count(text) > MAX_WORDS:
                    break
                if acceptable(text, source):
                    found.append((b, i, j, text))
    return found


def pick_windows(candidates, k, rng):
    """First k pairwise non-overlapping windows from a seeded shuffle."""
    order = sorted(candidates)
    rng.shuffle(order)
    chosen = []
    for c in order:
        if all(c[0] != d[0] or c[2] < d[1] or c[1] > d[2] for d in chosen):
            chosen.append(c)
        if len(chosen) == k:
            break
    return sorted(chosen) if len(chosen) == k else None


class Blocks(HTMLParser):
    """HTML to text blocks: block tags break, tables and hidden parts are skipped."""

    BLOCK = {"p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol",
             "table", "blockquote", "section", "dd", "dt", "center", "font_break"}
    SKIP = {"table", "style", "script", "sup", "ix:header", "math", "figure", "noscript"}

    def __init__(self, only_p=False, skip_classes=()):
        super().__init__(convert_charrefs=True)
        self.blocks, self.buf, self.skip, self.only_p, self.p = [], [], 0, only_p, 0
        self.skip_classes, self.stack = set(skip_classes), []

    def _flush(self):
        text = clean("".join(self.buf))
        if text:
            self.blocks.append(text)
        self.buf = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        style = (attrs.get("style") or "").replace(" ", "").lower()
        classes = set((attrs.get("class") or "").split())
        hide = tag in self.SKIP or "display:none" in style or bool(classes & self.skip_classes)
        if tag not in ("br", "img", "hr", "meta", "link", "input"):
            self.stack.append(hide)
            self.skip += hide
        if tag in self.BLOCK:
            self._flush()
        if tag == "p":
            self.p += 1

    def handle_endtag(self, tag):
        if tag in ("br", "img", "hr", "meta", "link", "input"):
            return
        if self.stack:
            self.skip -= self.stack.pop()
        if tag in self.BLOCK:
            self._flush()
        if tag == "p":
            self.p -= 1

    def handle_data(self, data):
        if not self.skip and (not self.only_p or self.p > 0):
            self.buf.append(data)

    def close(self):
        super().close()
        self._flush()
        return self.blocks


def html_blocks(text, **kwargs):
    parser = Blocks(**kwargs)
    parser.feed(text)
    return parser.close()


# --- sources: each returns (pool of doc keys, loader doc -> metadata + blocks) -

def rfc_pool(f):
    xml = f.get("https://www.rfc-editor.org/rfc-index.xml")
    ns = {"r": "https://www.rfc-editor.org/rfc-index"}
    tree, pool = ET.fromstring(xml.encode()), []
    for e in tree.findall("r:rfc-entry", ns):
        num = e.findtext("r:doc-id", namespaces=ns)[3:].lstrip("0")
        year = int(e.findtext("r:date/r:year", namespaces=ns))
        status = e.findtext("r:current-status", namespaces=ns)
        if (int(num) >= 8650 and 2020 <= year <= 2024 and status in ("PROPOSED STANDARD", "INTERNET STANDARD")
                and e.find("r:obsoleted-by", ns) is None and num not in EXCLUDED["rfc"]):
            date = f'{e.findtext("r:date/r:month", namespaces=ns)} {year}'
            pool.append(f'{num}|{clean(e.findtext("r:title", namespaces=ns))}, {date}')
    return sorted(pool, key=lambda k: int(k.split("|")[0]))


def rfc_load(f, key):
    num, title = key.split("|", 1)
    url = f"https://www.rfc-editor.org/rfc/rfc{num}.txt"
    lines = f.get(url).splitlines()
    start = next((k for k, l in enumerate(lines) if re.match(r"1\.\s+[A-Z]", l)), None)
    if start is None:
        return None
    end = next((k for k in range(start, len(lines)) if re.match(
        r"(\d+(\.\d+)*\.\s+)?(Normative References|Informative References|References|Acknowledg|Authors' Addresses|Appendix)",
        lines[k])), len(lines))
    blocks, cur = [], []
    for line in lines[start:end]:
        if re.match(r"   \S", line) and not re.match(r"   [*o-]\s", line):
            cur.append(line.strip())
        else:
            if cur:
                blocks.append(join_lines(cur))
            cur = []
    if cur:
        blocks.append(join_lines(cur))
    return {"source_doc": f"rfc{num}", "source_title": f"RFC {num}, {title}", "source_url": url,
            "license": LICENSES["rfc"]}, blocks


def sec_pool(f):
    tickers = f.json("https://www.sec.gov/files/company_tickers.json", 0.2)
    pool, seen = [], set()
    for row in tickers.values():  # file order, as published at fetch time
        cik = str(row["cik_str"])
        if cik in seen or cik in EXCLUDED["sec10k"]:
            continue
        seen.add(cik)
        sub = f.json(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", 0.2)
        recent = sub["filings"]["recent"]
        for form, date, acc, doc in zip(recent["form"], recent["filingDate"], recent["accessionNumber"],
                                        recent["primaryDocument"]):
            if form == "10-K" and "2024-11-01" <= date <= "2026-10-31":
                pool.append(f"{cik}|{acc}|{doc}|{date}|{sub['name']}")
                break
        if len(pool) == 50:
            break
    return sorted(pool)


def sec_load(f, key):
    cik, acc, doc, date, name = key.split("|")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
    blocks = html_blocks(f.get(url, 0.2))
    # Body only: from the "Item 1. Business" heading that is followed by prose
    # (table-of-contents entries are not) to the signatures.
    heads = [k for k, b in enumerate(blocks) if re.match(r"(part i\W+)?item\s*1\s*\.?\s*\W?\s*business\b", b, re.I)
             and any(word_count(x) >= 40 for x in blocks[k + 1:k + 6])]
    if not heads:
        return None
    end = next((k for k in range(heads[0], len(blocks)) if re.fullmatch(r"signatures?", blocks[k], re.I)), len(blocks))
    blocks = blocks[heads[0]:end]
    return {"source_doc": f"cik{cik}-10k-{date[:4]}", "source_title": f"{name}, Form 10-K filed {date} "
            f"(accession {acc})", "source_url": url, "license": LICENSES["sec10k"]}, blocks


_UCSB = "https://www.presidency.ucsb.edu"
_ROW = re.compile(r'about="(/documents/[^"]+)".*?content="(\d{4}-\d{2}-\d{2})T.*?/people/president/[^"]*">([^<]+)<',
                  re.S)


def speech_pool(f):
    pool = []
    for cat in ("inaugural-addresses", "state-the-union-addresses"):
        for page in range(3):
            listing = f.get(f"{_UCSB}/documents/app-categories/spoken-addresses-and-remarks/presidential/"
                            f"{cat}?items_per_page=100&page={page}")
            rows = _ROW.findall(listing)
            for path, date, who in rows:
                if date >= "1981-01-20" and path not in EXCLUDED["speech"]:
                    pool.append(f"{path}|{date}|{html.unescape(who)}")
            if len(rows) < 100:
                break
    return sorted(set(pool))


def speech_load(f, key):
    path, date, who = key.split("|")
    page = f.get(_UCSB + path)
    title = clean(re.sub("<[^>]+>", "", re.search(r"<h1>(.*?)</h1>", page, re.S).group(1)))
    body = page.split('class="field-docs-content"', 1)[1].split('class="field-docs-', 1)[0]
    blocks = []
    for block in html_blocks("<div " + body, only_p=True):
        block = clean(re.sub(r"\[[^\]]*\]", " ", block))  # stage directions such as [Applause]
        if block:
            blocks.append(block)
    return {"source_doc": path.rsplit("/", 1)[1], "source_title": f"{who}, {title}, {date}",
            "source_url": _UCSB + path, "license": LICENSES["speech"]}, blocks


_FDA = ("https://api.fda.gov/drug/label.json?search=effective_time:[20240101+TO+20241231]+AND+"
        "openfda.product_type:%22HUMAN+PRESCRIPTION+DRUG%22+AND+_exists_:clinical_pharmacology+AND+"
        "_exists_:warnings_and_cautions")


# Unnumbered run-in subheadings in label text ("Metabolism Desmopressin is ...").
_FDA_HEADINGS = re.compile(
    r"(?:^|(?<=[.:]\s))(?:Absorption|Distribution|Metabolism|Elimination|Excretion|Drug Interactions|"
    r"Pharmacokinetics|Pharmacodynamics|Specific Populations|Microbiology|Mechanism of Action|Pulmonary Function|"
    r"Hemodynamics?|Hemodynamic Effects|Electrophysiology|Electrophysiologic Effects|Cardiac Electrophysiology|"
    r"Food Effect|Effect of Food|Pediatrics?|Geriatrics?|Renal Impairment|Hepatic Impairment|Pregnancy|Lactation|"
    r"Limitations of Use|Mechanism of Resistance|Resistance|Cross-Resistance|Carcinogenesis|Special Populations|"
    r"Pediatric Patients|Geriatric Patients|Patients with Renal Impairment|Patients with Hepatic Impairment|"
    r"Gender|Race|Age|Sex)(?::\s+|\s+(?=[A-Z]))")


def fda_pool(f):
    counts = f.json(_FDA.replace("label.json?", "label.json?count=openfda.generic_name.exact&limit=1000&"))
    return sorted(r["term"] for r in counts["results"] if not re.search(r"\bAND\b|,|/", r["term"]))


def fda_load(f, generic):
    quoted = urllib.parse.quote(f'"{generic}"')
    found = f.json(_FDA + f"+AND+openfda.generic_name.exact:{quoted}&limit=100")["results"]
    label = min(found, key=lambda r: r["set_id"])
    blocks = []
    for field in ("indications_and_usage", "clinical_pharmacology", "warnings_and_cautions"):
        for text in label.get(field, []):
            # Subsection headings ("12.1 Mechanism of Action") are hard breaks; the
            # first sentence of each part carries the heading words and is dropped.
            text = _FDA_HEADINGS.sub(" 99 ", clean(text))  # run-in subheadings break like numbered ones
            for part in re.split(r"(?:^|\s)\d{1,2}(?:\.\d{1,2})?\s+(?=[A-Z])", text):
                sents = sentences(part)
                if len(sents) > 1:
                    blocks.append(" ".join(sents[1:]))
    brand = (label.get("openfda", {}).get("brand_name") or [generic])[0]
    return {"source_doc": f"fda-{label['set_id']}-v{label['version']}",
            "source_title": f"{brand} ({generic.lower()}) prescribing information, set {label['set_id']}, "
                            f"version {label['version']}, effective {label['effective_time']}",
            "source_url": f"https://api.fda.gov/drug/label.json?search=set_id:{label['set_id']}",
            "license": LICENSES["openfda"]}, blocks


_FR = ("https://www.federalregister.gov/api/v1/documents.json?per_page=1000&order=oldest"
       "&conditions[type][]=RULE&conditions[significant]=1"
       "&conditions[publication_date][gte]=2024-01-01&conditions[publication_date][lte]=2025-06-30"
       "&fields[]=document_number&fields[]=title&fields[]=raw_text_url&fields[]=publication_date")


def fedreg_pool(f):
    pool, url = [], _FR
    while url:
        page = f.json(url)
        pool += [f"{r['document_number']}|{r['publication_date']}|{r['raw_text_url']}|{r['title']}"
                 for r in page.get("results", [])]
        url = page.get("next_page_url")
    return sorted(set(pool))


def fedreg_load(f, key):
    number, date, url, title = key.split("|", 3)
    raw = html.unescape(re.sub(r"<[^>]+>", "", f.get(url)))
    if "SUPPLEMENTARY INFORMATION:" not in raw:
        return None
    raw = raw.split("SUPPLEMENTARY INFORMATION:", 1)[1].split("List of Subjects", 1)[0]
    raw = raw.replace("``", "\u201c").replace("''", "\u201d")  # GPO text marks quotes as ``...'' 
    blocks, cur = [], []
    for line in raw.splitlines():
        if not line.strip() or line.startswith("    "):
            if cur:
                blocks.append(join_lines(cur))
            cur = [line.strip()] if line.strip() else []
        else:
            cur.append(line.strip())
    if cur:
        blocks.append(join_lines(cur))
    return {"source_doc": f"fr-{number}", "source_title": f"{title}, Federal Register doc. {number}, {date}",
            "source_url": url, "license": LICENSES["fedreg"]}, blocks


_OAI = "https://oaipmh.arxiv.org/oai"
_ARX = "{http://arxiv.org/OAI/arXivRaw/}"


def arxiv_records(f):
    if getattr(f, "arxiv", None) is not None:
        return f.arxiv
    records = []
    for oai_set in ("physics", "q-bio"):
        url = f"{_OAI}?verb=ListRecords&metadataPrefix=arXivRaw&set={oai_set}&from=2025-03-03&until=2025-03-05"
        while url:
            tree = ET.fromstring(f.get(url, 4).encode())
            for raw in tree.iter(_ARX + "arXivRaw"):
                records.append({k: (raw.findtext(_ARX + k) or "").strip() for k in
                                ("id", "title", "license", "abstract", "categories")}
                               | {"v1": raw.find(_ARX + "version").findtext(_ARX + "date"),
                                  "versions": len(raw.findall(_ARX + "version"))})
            token = tree.find(".//{http://www.openarchives.org/OAI/2.0/}resumptionToken")
            url = (f"{_OAI}?verb=ListRecords&resumptionToken={urllib.parse.quote(token.text)}"
                   if token is not None and token.text else None)
    f.arxiv = {r["id"]: r for r in records}
    return f.arxiv


def arxiv_pool(f):
    return sorted(i for i, r in arxiv_records(f).items()
                  if r["license"] == "http://creativecommons.org/licenses/by/4.0/" and "2025" in r["v1"]
                  and not r["categories"].startswith("cs."))


def arxiv_load(f, ident):
    r = arxiv_records(f)[ident]
    return {"source_doc": f"arxiv-{ident}v{r['versions']}", "source_title": f"arXiv:{ident}v{r['versions']}, "
            f"{clean(r['title'])} (abstract; {r['categories']})", "source_url": f"https://arxiv.org/abs/{ident}v{r['versions']}",
            "license": LICENSES["arxiv"]}, [clean(r["abstract"])]


_WIKI = "https://simple.wikipedia.org/w/api.php?format=json&formatversion=2&"


def simplewiki_pool(f):
    titles = []
    for cat in ("Very_good_articles", "Good_articles"):
        data = f.json(_WIKI + f"action=query&list=categorymembers&cmtitle=Category:{cat}&cmlimit=500&cmnamespace=0")
        titles += [m["title"] for m in data["query"]["categorymembers"]]
    return sorted(set(titles))


def simplewiki_load(f, title):
    info = f.json(_WIKI + "action=query&prop=revisions&rvprop=ids&titles=" + urllib.parse.quote(title))
    revid = info["query"]["pages"][0]["revisions"][0]["revid"]
    parsed = f.json(_WIKI + f"action=parse&oldid={revid}&prop=text")
    blocks = html_blocks(parsed["parse"]["text"], only_p=True,
                         skip_classes=("reference", "mw-editsection", "noprint", "IPA", "mw-ref"))
    return {"source_doc": f"simplewiki-{revid}", "source_title": f"{title} (Simple English Wikipedia, revision {revid})",
            "source_url": f"https://simple.wikipedia.org/w/index.php?oldid={revid}",
            "license": LICENSES["simplewiki"]}, blocks


SOURCES = {"rfc": (rfc_pool, rfc_load), "sec10k": (sec_pool, sec_load), "speech": (speech_pool, speech_load),
           "openfda": (fda_pool, fda_load), "fedreg": (fedreg_pool, fedreg_load),
           "arxiv": (arxiv_pool, arxiv_load), "simplewiki": (simplewiki_pool, simplewiki_load)}


# --- build --------------------------------------------------------------------

def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def select(fetcher, log=print):
    """Seeded selection of every domain's passages; returns (passages, pool sizes)."""
    passages, pools = [], {}
    old = json.loads((ROOT / "evals/data/week7_sourced_passages.json").read_text())
    old += json.loads((ROOT / "evals/data/week7_passages.json").read_text())
    old_text = [p["original"] for p in old]
    for domain, (source, docs, per_doc, kind, prefix) in PLAN.items():
        rng = random.Random(f"{SEED}:{domain}")
        pool = SOURCES[source][0](fetcher)
        pools[domain] = len(pool)
        order = list(pool)
        rng.shuffle(order)
        taken, tried = [], 0
        for key in order:
            if len(taken) == docs:
                break
            tried += 1
            loaded = SOURCES[source][1](fetcher, key)
            if loaded is None:
                continue
            meta, blocks = loaded
            found = [c for c in windows(blocks, source) if not any(c[3] in t or t in c[3] for t in old_text)]
            chosen = pick_windows(found, per_doc, rng)
            if chosen:
                taken.append((meta, chosen))
        if len(taken) < docs:
            raise SystemExit(f"{domain}: only {len(taken)} of {docs} documents qualified from {len(pool)}")
        log(f"{domain}: {docs} documents from a pool of {len(pool)} ({tried} opened)")
        n = 0
        for meta, chosen in taken:
            for block, first, last, text in chosen:
                n += 1
                passages.append({"id": f"{prefix}-v3-{n:03d}", "domain": domain, "kind": kind, **meta,
                                 "sentence_span": [block, first, last], "word_count": word_count(text),
                                 "original": text})
    texts = [p["original"] for p in passages]
    if len(set(texts)) != len(texts):
        raise SystemExit("Duplicate passage text")
    return passages, pools


def split(passages):
    """Document-level split per domain: shuffled documents fill dev to one third."""
    for domain in PLAN:
        rows = [p for p in passages if p["domain"] == domain]
        docs = sorted({p["source_doc"] for p in rows})
        random.Random(f"{SEED}:split:{domain}").shuffle(docs)
        target, dev = round(len(rows) * DEV_FRACTION), set()
        for doc in docs:
            if sum(p["source_doc"] in dev for p in rows) >= target:
                break
            dev.add(doc)
        for p in rows:
            p["split"] = "dev" if p["source_doc"] in dev else "test"
    return passages


FIELDS = ("id", "domain", "kind", "split", "source_doc", "source_title", "source_url", "license",
          "sentence_span", "word_count")


def build(args):
    state = args.state.resolve()
    if state == ROOT or ROOT in state.parents:
        raise SystemExit("State must be outside the checkout; raw text stays local.")
    fetcher = Fetcher(state / "raw" / "http", offline=args.offline)
    passages, pools = select(fetcher)
    split(passages)
    manifest = {
        "name": "testset-v3", "seed": SEED, "words": [MIN_WORDS, MAX_WORDS], "dev_fraction": DEV_FRACTION,
        "builder": "evals/build_testset_v3.py", "plan": {d: dict(zip(("source", "documents", "per_document",
                                                                     "kind", "prefix"), v)) for d, v in PLAN.items()},
        "pool_sizes": pools, "licenses": LICENSES,
        "passages": [{**{k: p[k] for k in FIELDS}, "text_sha256": sha_text(p["original"])} for p in passages],
    }
    out = state / "panel"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("all", passages), ("dev", [p for p in passages if p["split"] == "dev"]),
                       ("test-LOCKED", [p for p in passages if p["split"] == "test"])):
        path = out / f"{name}.json"
        if path.exists():
            path.chmod(0o644)
        path.write_text(json.dumps([{k: p[k] for k in (*FIELDS, "original")} for p in rows], indent=1,
                                   ensure_ascii=False))
    (out / "test-LOCKED.json").chmod(0o444)
    if args.write_manifest:
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
        for name in ("dev", "test"):
            (DATA / f"{name}_ids.txt").write_text(
                "".join(p["id"] + "\n" for p in passages if p["split"] == name))
    counts = {}
    for p in passages:
        counts.setdefault(p["domain"], {}).setdefault(p["split"], 0)
        counts[p["domain"]][p["split"]] += 1
    print(json.dumps(counts, indent=1))
    print(f"{len(passages)} passages -> {out}")


def verify(args):
    """The local panel matches the committed manifest (IDs, split, text hashes)."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    expected = {p["id"]: p for p in manifest["passages"]}
    rows = json.loads((args.state / "panel" / "all.json").read_text())
    problems = [p["id"] for p in rows if p["id"] not in expected
                or expected[p["id"]]["text_sha256"] != sha_text(p["original"])
                or expected[p["id"]]["split"] != p["split"]]
    if problems or len(rows) != len(expected):
        raise SystemExit(f"Panel differs from manifest: {problems[:10]} ({len(rows)} vs {len(expected)})")
    print(f"{len(rows)} passages match the manifest")


ONESTOP_COMMIT = "37f8db3945cd2f3cc0caafe45674147b224349be"
ONESTOP = f"https://raw.githubusercontent.com/nishkalavallabhi/OneStopEnglishCorpus/{ONESTOP_COMMIT}/Sentence-Aligned/"


def onestop_pairs(text):
    """{advanced sentence: simplified sentence} from a Sentence-Aligned file."""
    out = {}
    for block in text.split("*******"):
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if len(lines) == 2:
            out.setdefault(lines[0], lines[1])
    return out


def onestop(args):
    """OneStopEnglish (CC BY-SA 4.0) multi-reference set: advanced sentences
    aligned to both an intermediate and an elementary version. Seeded split,
    one third dev; the test part is locked like the passage panel."""
    fetcher = Fetcher(args.state.resolve() / "raw" / "http", offline=args.offline)
    ele = onestop_pairs(fetcher.get(ONESTOP + "ADV-ELE.txt"))
    inter = onestop_pairs(fetcher.get(ONESTOP + "ADV-INT.txt"))
    rows = [{"id": "ose-" + sha_text(adv)[:12], "original": adv, "references": [inter[adv], ele[adv]]}
            for adv in sorted(ele) if adv in inter]
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("OneStopEnglish ID collision")
    random.Random(f"{SEED}:onestop").shuffle(rows)
    cut = round(len(rows) * DEV_FRACTION)
    out = args.state.resolve() / "panel"
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("onestop-dev", rows[:cut]), ("onestop-test-LOCKED", rows[cut:])):
        path = out / f"{name}.json"
        if path.exists():
            path.chmod(0o644)
        path.write_text(json.dumps(part, indent=1, ensure_ascii=False))
    (out / "onestop-test-LOCKED.json").chmod(0o444)
    if args.write_manifest:
        for name, part in (("dev", rows[:cut]), ("test", rows[cut:])):
            (DATA / f"onestop_{name}_ids.txt").write_text("".join(r["id"] + "\n" for r in part))
    print(f"OneStopEnglish @{ONESTOP_COMMIT[:7]}: {len(rows)} sentences with INT and ELE references; "
          f"{cut} dev, {len(rows) - cut} test")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--state", type=Path, required=True)
    b.add_argument("--offline", action="store_true", help="use the download cache only")
    b.add_argument("--write-manifest", action="store_true", help="(re)write the committed manifest and ID lists")
    v = sub.add_parser("verify")
    v.add_argument("--state", type=Path, required=True)
    o = sub.add_parser("onestop")
    o.add_argument("--state", type=Path, required=True)
    o.add_argument("--offline", action="store_true")
    o.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()
    {"build": build, "verify": verify, "onestop": onestop}[args.command](args)


if __name__ == "__main__":
    main()
