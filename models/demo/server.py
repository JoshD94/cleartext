"""ClearText demo: simplify a sentence with the default v7 model and show each step.

Run from models/:  .venv/bin/python demo/server.py --host 127.0.0.1 --port 8770
Standard library HTTP server; the model loads once at startup. Requests run one at a time.

The final output always comes from EnsembleClearText.analyze; the trace below re-runs the same library calls
stage by stage so each building block can be shown on its own.
"""

import sys, re, json, argparse, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from nltk.corpus import wordnet as wn
from cleartext.ensemble_pipeline import EnsembleClearText, LATEST
from cleartext.detection import CONTENT
from cleartext.features import nlp
from cleartext.lexical import guardrails, PROTECTED
from cleartext.rewrites import structural, PHRASES

print("Loading model...", flush=True)
PIPE = EnsembleClearText.load(LATEST)
PIPE.analyze(
    "The technician will commence the installation.", structure=False, phrases=False
)  # warm caches
DEFAULTS = {
    "decision": PIPE.decision_threshold,
    "hard": PIPE.detector["threshold"],
    "noun": PIPE.noun_threshold,
}
PAGE = HERE / "index.html"  # read per request so page edits apply without a restart
# Short column label and full description for each safety check, in display order.
CHECK_NAMES = {
    "numbers": ("Numbers", "Numbers are unchanged"),
    "quantities": ("Amounts", "Quantities with units are unchanged"),
    "negation": ("Negation", '"not", "no", "never" are kept'),
    "modality": ("Certainty", '"may", "must", "will" are kept'),
    "entities": ("Names", "Named people, places and groups are kept"),
    "entity_roles": ("Name roles", "Names keep their role in the sentence"),
    "technical_phrases": ("Terms", "Protected technical terms are kept"),
    "argument_roles": ("Who did what", "Subjects and objects keep their roles"),
    "negation_attachment": ("Neg. scope", "Negation still attaches to the same word"),
    "article": ("a/an", 'No "a/an" before a mass noun like "information"'),
}
MEMBER_NAMES = {
    "sense_fit": "Meaning match",
    "round_trip": "Reader gets same meaning",
    "first_sense": "Common meaning",
    "candidate_sense_rank": "Central meaning for new word",
    "swords_checker": "SWORDS context checker",
    "context_vector": "Fits sentence (vectors)",
    "argument_fit": "Fits linked words",
    "bigram": "Word-pair frequency",
    "bert_slot": "BERT: natural in this slot",
    "bert_slot_relative": "BERT: vs original word",
}
print("Ready", flush=True)


def clamp(v, lo, hi, default):
    try:
        return min(hi, max(lo, float(v)))
    except (TypeError, ValueError):
        return default


def skip_reason(t, spans):
    if not t.is_alpha:
        return "punctuation or number"
    if t.ent_type_:
        return "name (" + t.ent_type_.lower() + ")"
    if t.pos_ not in CONTENT:
        return "function word (" + t.pos_.lower() + ")"
    if t.is_stop:
        return "very common word"
    if any(a <= t.idx < b for a, b in spans):
        return "protected technical term"
    return None


def trace(text, opts):
    """Every building block's view of the sentence, in pipeline order."""
    out = {}
    current = text
    # 1. Sentence structure
    if opts["structure"]:
        new, changes = structural(current)
        out["structure"] = {
            "on": True,
            "changed": bool(changes),
            "rule": changes[0]["rule"] if changes else None,
            "before": current,
            "after": new,
        }
        current = new
    else:
        out["structure"] = {"on": False}
    # 2. Phrases: every inventory match, its difficulty before/after, and which one was applied
    matches = []
    for src, tgt in PHRASES:
        for m in re.finditer(r"(?<!\w)" + re.escape(src) + r"(?!\w)", current, re.I):
            a, b = PIPE.complexity(src), PIPE.complexity(tgt)
            matches.append(
                {
                    "phrase": m.group(),
                    "replacement": tgt,
                    "before": a,
                    "after": b,
                    "gain": a - b,
                }
            )
    if opts["phrases"]:
        from cleartext.rewrites import phrase_rewrite

        new, changes = phrase_rewrite(current, PIPE.complexity)
        applied = changes[0]["source"] if changes else None
        for m in matches:
            m["applied"] = m["phrase"] == applied
        out["phrases"] = {"on": True, "matches": matches, "inventory": len(PHRASES)}
        current = new
    else:
        out["phrases"] = {"on": False, "matches": matches, "inventory": len(PHRASES)}
    # 3. Word difficulty and hard-word detection on every token
    doc = nlp()(current)
    original_doc = nlp()(text)
    spans = [
        (m.start(), m.end())
        for p in PROTECTED
        for m in re.finditer(r"(?<!\w)" + re.escape(p) + r"(?!\w)", current, re.I)
    ]
    words = [t for t in doc if skip_reason(t, spans) is None]
    hard = dict(zip([t.i for t in words], PIPE.hardness(doc, words)))
    th = PIPE.detector["threshold"]
    tokens = []
    for t in doc:
        if t.is_space:
            continue
        reason = skip_reason(t, spans)
        h = hard.get(t.i)
        flagged = (
            h is not None and h >= th and (t.pos_ != "NOUN" or h >= PIPE.noun_threshold)
        )
        why = reason or (
            "hard"
            if flagged
            else (
                "noun below noun threshold"
                if h is not None and h >= th
                else "not hard enough"
            )
        )
        tokens.append(
            {
                "text": t.text,
                "ws": t.whitespace_,
                "pos": t.pos_,
                "skip": reason,
                "hardness": None if h is None else float(h),
                "complex": None
                if reason
                else float(PIPE.difficulty.score([t.text])[0]),
                "flagged": flagged,
                "why": why,
            }
        )
    out["detect"] = {
        "threshold": th,
        "noun_threshold": PIPE.noun_threshold,
        "tokens": tokens,
    }
    # 4-9. Per flagged word: meaning, candidates, fit, simplicity, decision, safety
    targets = []
    for t in words:
        h = hard[t.i]
        if not (h >= th and (t.pos_ != "NOUN" or h >= PIPE.noun_threshold)):
            continue
        r = PIPE.rank_word(doc, t, original_doc)
        senses = [
            {
                "sense": n,
                "prob": float(p),
                "definition": wn.synset(n).definition(),
                "lemmas": [l.replace("_", " ") for l in wn.synset(n).lemma_names()][:5],
            }
            for n, p in r["senses"]
        ]
        cands = []
        for c in sorted(r["candidates"], key=lambda c: -c["accept"]):
            passed = c["accept"] >= PIPE.decision_threshold
            checks = None
            if passed:
                g = guardrails(doc.text, c["output"], doc)
                results = {
                    **{k: bool(v) for k, v in g["checks"].items()},
                    "article": "mass noun after a/an" not in c["rejections"],
                }
                checks = [
                    {"key": k, "ok": results[k]} for k in CHECK_NAMES if k in results
                ]
            cands.append(
                {
                    "word": c["word"],
                    "source": c["source"],
                    "fit": c["fit"],
                    "gain": c["gain"],
                    "difficulty": c["target_difficulty"] - c["gain"],
                    "accept": c["accept"],
                    "passed": passed,
                    "checks": checks,
                    "safe": None if checks is None else all(x["ok"] for x in checks),
                    "members": {
                        MEMBER_NAMES.get(k, k): v
                        for k, v in c.get("members", {}).items()
                        if k in MEMBER_NAMES
                    },
                }
            )
        sel = r["selected"]
        targets.append(
            {
                "word": t.text,
                "start": t.idx,
                "hardness": float(h),
                "difficulty": r["difficulty"],
                "senses": senses,
                "candidates": cands,
                "best": sel["word"] if sel else None,
                "best_accept": sel["accept"] if sel else None,
                "score": sel["accept"] * h**PIPE.hardness_alpha if sel else None,
            }
        )
    out["targets"] = targets
    return out


def simplify(req):
    text = str(req.get("text", "")).strip()[:1000]
    if not text:
        return {"error": "Enter a sentence."}
    opts = {
        "max_edits": int(clamp(req.get("max_edits"), 0, 5, 1)),
        "structure": bool(req.get("structure")),
        "phrases": bool(req.get("phrases")),
    }
    PIPE.decision_threshold = clamp(
        req.get("strictness"), 0.05, 0.95, DEFAULTS["decision"]
    )
    PIPE.detector["threshold"] = clamp(req.get("hard"), 0.05, 0.95, DEFAULTS["hard"])
    PIPE.noun_threshold = clamp(req.get("noun"), 0, 1, DEFAULTS["noun"])
    started = time.time()
    try:
        r = PIPE.analyze(
            text,
            structure=opts["structure"],
            phrases=opts["phrases"],
            max_word_edits=opts["max_edits"],
        )
        steps = trace(text, opts)
        steps["decision_threshold"] = PIPE.decision_threshold
    finally:
        PIPE.decision_threshold, PIPE.detector["threshold"], PIPE.noun_threshold = (
            DEFAULTS["decision"],
            DEFAULTS["hard"],
            DEFAULTS["noun"],
        )
    edits = [
        {
            "stage": e["stage"],
            "from": e.get("source") or e.get("before"),
            "to": e.get("replacement") or e.get("after"),
            "rule": e.get("rule"),
        }
        for e in r["edits"]
    ]
    chosen = {e["from"] for e in edits if e["stage"] == "word"}
    ranked = sorted(
        [t for t in steps["targets"] if t["best"]], key=lambda t: -t["score"]
    )
    steps["checks"] = [
        {"key": k, "short": a, "full": b} for k, (a, b) in CHECK_NAMES.items()
    ]
    steps["choice"] = {
        "max_edits": opts["max_edits"],
        "alpha": PIPE.hardness_alpha,
        "ranked": [
            {
                "word": t["word"],
                "best": t["best"],
                "accept": t["best_accept"],
                "hardness": t["hardness"],
                "score": t["score"],
                "chosen": t["word"] in chosen,
            }
            for t in ranked
        ],
    }
    return {
        "original": r["original"],
        "output": r["output"],
        "edits": edits,
        "steps": steps,
        "level_before": r.get("reading_level_before"),
        "level_after": r.get("reading_level_after"),
        "seconds": round(time.time() - started, 2),
    }


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, kind):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        self.send(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/simplify":
            return self.send(404, b"Not found", "text/plain")
        try:
            req = json.loads(
                self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 20000))
                or b"{}"
            )
            out = simplify(req)
        except Exception as e:  # report to the page instead of dropping the connection
            out = {"error": f"{type(e).__name__}: {e}"}
        self.send(200, json.dumps(out, default=float).encode(), "application/json")

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    print(f"Serving http://{a.host}:{a.port}", flush=True)
    HTTPServer((a.host, a.port), Handler).serve_forever()
