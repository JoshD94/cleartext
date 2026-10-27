"""Optional local rule-based input correction, before simplification.

LanguageTool supplies suggestions. This module applies only articles,
apostrophes in negative contractions, and tense-preserving verb agreement.
It does not use LanguageTool's cloud API, n-grams, or neural rules.
"""

import argparse
import json
import re
from urllib.error import URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from lemminflect import getInflection, getLemma

from .features import nlp
from .technical_terms import detect_terms

POLICY_VERSION = "local-grammar-v2"
AGREEMENT_RULES = {
    "HE_VERB_AGR",
    "PERS_PRONOUN_AGREEMENT",
    "AGREEMENT_SENT_START",
    "BASE_FORM",
    "AUXILIARY_DO_WITH_INCORRECT_VERB_FORM",
}
CONTRACTIONS = {
    "dont",
    "doesnt",
    "didnt",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "hasnt",
    "havent",
    "hadnt",
    "couldnt",
    "wouldnt",
    "shouldnt",
    "mustnt",
    "neednt",
}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise URLError("Grammar server redirects are disabled")


def _python_offset(text, units):
    """Java/LanguageTool offsets count UTF-16 units, not Python characters."""
    if not isinstance(units, int) or units < 0:
        raise ValueError("Invalid UTF-16 offset")
    count = 0
    for index, character in enumerate(text):
        if count == units:
            return index
        count += 2 if ord(character) > 0xFFFF else 1
    if count == units:
        return len(text)
    raise ValueError("Offset is out of range or inside a surrogate pair")


def _allowed_form(old, new, rule, token):
    a, b = old.casefold(), new.casefold()
    if rule == "EN_A_VS_AN":
        return {a, b} == {"a", "an"}
    if rule == "EN_CONTRACTION_SPELLING":
        return a in CONTRACTIONS and a != b and b.replace("'", "").replace("’", "") == a
    if rule not in AGREEMENT_RULES or token is None:
        return False
    if token.pos_ not in {"VERB", "AUX"} or not new.isalpha():
        return False
    # These rule families can also propose a different tense. Never choose it.
    if a in {"am", "is", "are", "was", "were"}:
        return a != b and {a, b} <= (
            {"was", "were"} if a in {"was", "were"} else {"am", "is", "are"}
        )
    lemmas = getLemma(a, upos="VERB")
    if len(lemmas) != 1:
        return False
    present = set(getInflection(lemmas[0], tag="VBP")) | set(
        getInflection(lemmas[0], tag="VBZ")
    )
    return a != b and a in present and b in present


class GrammarCorrector:
    def __init__(self, server="http://127.0.0.1:8786", timeout=5, max_edits=8):
        address = urlsplit(server)
        if (
            address.scheme != "http"
            or address.hostname not in {"127.0.0.1", "::1", "localhost"}
            or address.username
            or address.password
            or address.query
            or address.fragment
            or address.path not in {"", "/"}
        ):
            raise ValueError("Use a local loopback LanguageTool HTTP server")
        if timeout <= 0 or max_edits < 0:
            raise ValueError("timeout must be positive and max_edits nonnegative")
        self.server, self.timeout, self.max_edits = (
            server.rstrip("/"),
            timeout,
            max_edits,
        )

    def check(self, text):
        request = Request(
            self.server + "/v2/check",
            data=urlencode({"text": text, "language": "en-US"}).encode(),
        )
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=self.timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict) or not isinstance(result.get("matches"), list):
            raise ValueError("Invalid LanguageTool response")
        return result

    def correct(self, text):
        result = {
            "on": True,
            "backend": "local-languagetool",
            "policy": POLICY_VERSION,
            "original": text,
            "output": text,
            "changed": False,
            "edits": [],
            "suggestions": [],
            "available": False,
            "grammar_certified": False,
        }
        try:
            response = self.check(text)
        except (OSError, ValueError, TimeoutError) as error:
            return {
                **result,
                "reason": f"Grammar service unavailable: {type(error).__name__}",
            }
        result.update(available=True, software=response.get("software", {}))
        doc = nlp()(text)
        protected = [
            (e.start_char, e.end_char)
            for e in doc.ents
            if e.label_ not in {"DATE", "TIME"}
        ]
        date_spans = [
            (e.start_char, e.end_char) for e in doc.ents if e.label_ in {"DATE", "TIME"}
        ]
        protected += [(s.start, s.end) for s in detect_terms(doc) if s.protected]
        protected += [
            (m.start(), m.end()) for m in re.finditer(r"\d+(?:[.,]\d+)*", text)
        ]
        protected += [
            (m.start(), m.end()) for m in re.finditer(r"[\"“][^\"”]+[\"”]", text)
        ]
        protected += [
            (m.start(), m.end())
            for m in re.finditer(r"(?<!\w)'[^'\n]+'(?!\w)|`[^`]+`", text)
        ]
        edits = []
        try:
            for match in response["matches"]:
                start = _python_offset(text, match["offset"])
                end = _python_offset(text, match["offset"] + match["length"])
                if end < start:
                    raise ValueError("Negative match length")
                old = text[start:end]
                rule = match["rule"]["id"]
                choices = list(
                    dict.fromkeys(r["value"] for r in match.get("replacements", []))
                )
                token = next(
                    (t for t in doc if t.idx == start and t.idx + len(t) == end), None
                )
                allowed = [r for r in choices if _allowed_form(old, r, rule, token)]
                reason = None
                # Date detection can include an article, e.g. "A hour". Only
                # a/an may change there; dates, digits and all other names stay.
                blocked = protected + ([] if rule == "EN_A_VS_AN" else date_spans)
                if start == end or any(start < b and end > a for a, b in blocked):
                    reason = "protected_span_or_insertion"
                elif len(allowed) != 1:
                    reason = "unsupported_or_ambiguous_edit"
                elif any(start < e["end"] and end > e["start"] for e in edits):
                    reason = "overlapping_edit"
                elif len(edits) >= self.max_edits:
                    reason = "edit_limit"
                record = {
                    "start": start,
                    "end": end,
                    "source": old,
                    "rule": rule,
                    "suggestions": choices,
                    "applied": reason is None,
                    "reason": reason,
                }
                result["suggestions"].append(record)
                if reason is None:
                    edits.append(
                        {
                            "stage": "grammar",
                            "start": start,
                            "end": end,
                            "source": old,
                            "replacement": allowed[0],
                            "rule": rule,
                            "before": old,
                            "after": allowed[0],
                        }
                    )
        except (KeyError, TypeError, ValueError) as error:
            # A malformed response must not leave a partially corrected input.
            return {
                **result,
                "edits": [],
                "suggestions": [],
                "reason": f"Invalid grammar matches: {type(error).__name__}",
            }
        output = text
        for edit in sorted(edits, key=lambda e: e["start"], reverse=True):
            output = (
                output[: edit["start"]] + edit["replacement"] + output[edit["end"] :]
            )
        result.update(output=output, changed=output != text, edits=edits)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8786")
    args = parser.parse_args()
    print(json.dumps(GrammarCorrector(args.server).correct(args.text), indent=2))


if __name__ == "__main__":
    main()
