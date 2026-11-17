"""Deterministic body-text extraction from a two-column ACL-style PDF.

Reads `pdftohtml -xml -i -fontfullname` output (poppler) and keeps the title,
abstract and section paragraphs. Font and position rules drop references,
tables, figure text, captions, display equations and footnotes; line-end
hyphenation and PDF accent pairs are repaired. No model touches the text.

    pdftohtml -xml -i -q -fontfullname paper.pdf out   # writes out.xml
    python -m evals.paper_extract out.xml paragraphs.json

Rules (fitted to the ACL 2020 template, checked against ASSET, 2020.acl-main.424):
- body text is 16 px (11 pt) Times/CM, the abstract 15 px; smaller text is
  footnotes, table cells or footnote markers;
- 15 px text outside the abstract is a caption ("Table 1:", "Figure 1:");
- DejaVuSans is figure text; a line made only of CM math fonts is a display
  equation;
- bold 18 px lines are section headings, bold 16 px lines starting "N.N" are
  subsection headings, and bold text opening a paragraph is a run-in label
  (kept as a label, not as a sentence to simplify);
- a paragraph starts at a heading, a first-line indent, a bullet or a numbered
  item, or a vertical gap; it continues across columns and pages;
- extraction stops at "References".
"""

import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

BODY_SIZES = {16}
ABSTRACT_SIZE = 15
SECTION_SIZE = 18
MATH_FAMILIES = ("CMMI", "CMR", "CMSY", "CMEX")
LINE_TOLERANCE = 4
PARAGRAPH_GAP = 28  # body line pitch is about 20 px
INDENT = (10, 24)  # first-line indent, relative to the column's left edge

# Spacing accents pdftohtml emits before a base letter, as combining marks.
ACCENTS = {"´": "́", "`": "̀", "ˆ": "̂", "¨": "̈", "ˇ": "̌",
           "˜": "̃", "˚": "̊", "¸": "̧"}


def fonts(root):
    return {f.get("id"): {"size": int(f.get("size")), "family": f.get("family").split("+")[-1]}
            for f in root.iter("fontspec")}


def kind(font):
    family = font["family"]
    if family.startswith("DejaVu"):
        return "figure"
    if family.startswith(MATH_FAMILIES):
        return "math"
    if "Medi" in family or "Bold" in family:
        return "bold"
    return "text"


def elements(root):
    spec = fonts(root)
    for page in root.iter("page"):
        width = float(page.get("width"))
        for node in page.iter("text"):
            text = "".join(node.itertext())
            if not text.strip():
                continue
            font = spec[node.get("font")]
            left = float(node.get("left"))
            yield {"page": int(page.get("number")), "col": int(left > width / 2),
                   "top": float(node.get("top")), "left": left, "right": left + float(node.get("width")),
                   "size": font["size"], "kind": kind(font), "text": text}


def lines(root):
    """Group elements into visual lines per (page, column), in reading order."""
    rows = []
    for e in sorted(elements(root), key=lambda e: (e["page"], e["col"], e["top"], e["left"])):
        last = rows[-1] if rows else None
        if last and (last["page"], last["col"]) == (e["page"], e["col"]) and abs(e["top"] - last["top"]) <= LINE_TOLERANCE:
            last["parts"].append(e)
        else:
            rows.append({"page": e["page"], "col": e["col"], "top": e["top"], "parts": [e]})
    for row in rows:
        row["parts"].sort(key=lambda e: e["left"])
        row["left"] = row["parts"][0]["left"]
        sizes = {}
        for e in row["parts"]:
            sizes[e["size"]] = sizes.get(e["size"], 0) + len(e["text"].strip())
        row["size"] = max(sizes, key=sizes.get)
    return rows


def join(parts):
    out = ""
    for k, e in enumerate(parts):
        if k and out and not out.endswith(" ") and not e["text"].startswith(" ") and e["left"] - parts[k - 1]["right"] > 2:
            out += " "
        out += e["text"]
    return out


def fix_accents(text):
    for accent, mark in ACCENTS.items():
        text = re.sub(re.escape(accent) + r"\s?([A-Za-zıȷ])",
                      lambda m: {"ı": "i", "ȷ": "j"}.get(m.group(1), m.group(1)) + mark, text)
    return unicodedata.normalize("NFC", text)


def column_lefts(rows):
    """Left edge of each column: the most common left of body lines."""
    counts = {}
    for r in rows:
        if r["size"] in BODY_SIZES:
            key = (r["col"], round(r["left"]))
            counts[key] = counts.get(key, 0) + 1
    return {col: min((k for k in counts if k[0] == col), key=lambda k: -counts[k])[1] for col in (0, 1)}


def classify(rows):
    """Tag each line: title, heading, abstract, body or drop."""
    lefts = column_lefts(rows)
    state = "front"
    for r in rows:
        kinds = {e["kind"] for e in r["parts"]}
        text = join(r["parts"]).strip()
        bold = kinds == {"bold"}
        if r["page"] == 1 and state == "front" and r["size"] > SECTION_SIZE:
            r["tag"] = "title"
            continue
        if bold and r["size"] == SECTION_SIZE and text == "Abstract":
            state, r["tag"] = "abstract", "drop"
            continue
        if bold and r["size"] == SECTION_SIZE and text in ("References", "Bibliography"):
            state = "end"
        if state in ("front", "end") or "figure" in kinds:
            r["tag"] = "drop"
            continue
        if bold and r["size"] == SECTION_SIZE and state != "end":
            state, r["tag"] = "body", "heading"
            continue
        if bold and r["size"] in BODY_SIZES and re.match(r"\d+\.\d+\b", text):
            r["tag"] = "heading"
            continue
        if state == "abstract":
            r["tag"] = "abstract" if r["size"] == ABSTRACT_SIZE else "drop"
            continue
        if r["size"] not in BODY_SIZES or kinds <= {"math"} or re.fullmatch(r"\d{3,5}", text):
            # Small text, display math, or a page number in the footer.
            r["tag"] = "drop"
            continue
        r["tag"] = "body"
        r["indent"] = r["left"] - lefts[r["col"]]
    return rows


def body_parts(row):
    keep = BODY_SIZES | {ABSTRACT_SIZE} if row["tag"] == "abstract" else BODY_SIZES
    return [e for e in row["parts"] if e["size"] in keep]


def starts_paragraph(row, previous):
    if previous is None or previous["tag"] != row["tag"]:
        return True
    first = body_parts(row)[0]["text"].lstrip()
    if first.startswith("•") or re.match(r"\d\.\s", first) and row["indent"] < INDENT[1]:
        return True
    if INDENT[0] <= row.get("indent", 0) <= INDENT[1]:
        return True
    same_column = (previous["page"], previous["col"]) == (row["page"], row["col"])
    return same_column and row["top"] - previous["top"] > PARAGRAPH_GAP


def paragraphs(rows):
    """Paragraph records: section, label (run-in bold, not simplified), text."""
    out, section, previous, current = [], None, None, None
    title = " ".join(join(r["parts"]).strip() for r in rows if r.get("tag") == "title")
    out.append({"section": "Title", "label": "", "lines": [title]})
    for r in rows:
        tag = r.get("tag")
        if tag == "heading":
            section = " ".join(join(r["parts"]).split())
            previous, current = r, None
            continue
        if tag not in ("abstract", "body"):
            continue
        parts = body_parts(r)
        if not parts:
            continue
        if current is None or starts_paragraph(r, previous):
            current = {"section": "Abstract" if tag == "abstract" else section, "label": "", "lines": [],
                       "labelling": True}
            out.append(current)
        if current.pop("labelling", False) or current.get("label_open"):
            # Leading bold runs (and a bullet glyph) form the paragraph's label.
            k = 0
            while k < len(parts) and (parts[k]["kind"] == "bold" or parts[k]["text"].strip() == "•"):
                k += 1
            label = join(parts[:k]).replace("•", "").strip()
            if label:
                current["label"] = (current["label"] + "\n" + label).strip()
            current["label_open"] = k == len(parts) and bool(label)
            parts = parts[k:]
            if parts and parts[0]["text"].lstrip().startswith(":"):
                current["label"] += ":"
                parts[0] = {**parts[0], "text": parts[0]["text"].lstrip()[1:]}
        if parts:
            current["lines"].append(join(parts))
        previous = r
    return [p for p in out if p["lines"]]


def dehyphenate(lines, vocabulary):
    """Join wrapped lines; drop a line-end hyphen when the joined word is the
    document's (or wordfreq's) spelling, else keep the hyphen."""
    text = ""
    for line in lines:
        line = " ".join(line.split())
        m = re.search(r"(\w+)-$", text)
        if m and line[:1].isalpha():
            nxt = re.match(r"\w+", line).group()
            joined = (m.group(1) + nxt).lower()
            hyphened = (m.group(1) + "-" + nxt).lower()
            if vocabulary(joined, hyphened):
                text = text[:-1] + line
            else:
                text += line
        else:
            text = (text + " " + line).strip() if text else line
    return text


def vocabulary_for(document):
    words = {}
    for w in re.findall(r"[\w-]+", document.lower()):
        words[w] = words.get(w, 0) + 1
    try:
        from wordfreq import zipf_frequency
    except ImportError:  # Document counts alone decide.
        zipf_frequency = None

    def prefer_joined(joined, hyphened):
        if words.get(joined, 0) != words.get(hyphened, 0):
            return words.get(joined, 0) > words.get(hyphened, 0)
        if zipf_frequency is None:
            return True
        if zipf_frequency(joined, "en") > 0:
            return True
        # Unknown either way: keep the hyphen only between two real words
        # ("multi-reference"); a fragment like "-ings" was a line break.
        first, second = hyphened.split("-", 1)
        return not (zipf_frequency(first, "en") >= 3 and zipf_frequency(second, "en") >= 3)
    return prefer_joined


def extract(xml_path):
    root = ET.parse(xml_path).getroot()
    paras = paragraphs(classify(lines(root)))
    # Document vocabulary from unbroken words only (line-final fragments excluded).
    document = " ".join(" ".join(l.split()[:-1]) for p in paras for l in p["lines"])
    prefer = vocabulary_for(fix_accents(document))
    rows = []
    for p in paras:
        text = fix_accents(dehyphenate([fix_accents(l) for l in p["lines"]], prefer))
        section = p["section"]
        slug = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")[:24]
        label = fix_accents(dehyphenate(p["label"].split("\n"), prefer)) if p["label"] else ""
        rows.append({"id": f"p{len(rows):03d}-{slug}", "section": section,
                     "label": label, "original": text})
    return rows


def main(argv=None):
    argv = argv or sys.argv[1:]
    rows = extract(argv[0])
    Path(argv[1]).write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(rows)} paragraphs, {sum(len(r['original'].split()) for r in rows)} words")


if __name__ == "__main__":
    main()
