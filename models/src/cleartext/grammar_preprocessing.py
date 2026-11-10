"""Optional local rule-based input correction, before simplification.

LanguageTool supplies suggestions. This module applies only articles,
apostrophes in negative contractions, and tense-preserving verb agreement.
It does not use LanguageTool's cloud API, n-grams, or neural rules.
"""

import argparse
from http.client import HTTPException
import json
import re
from urllib.error import URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .protected_literals import literal_spans

POLICY_VERSION = "local-grammar-v3"
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


def nlp():
    # Metadata readers and mocked mechanism tests must not load parser/model
    # dependencies merely by importing the correction policy.
    from .features import nlp as load

    return load()


def detect_terms(doc):
    from .technical_terms import detect_terms as detect

    return detect(doc)


def _present_forms(word):
    from lemminflect import getInflection, getLemma

    lemmas = getLemma(word, upos="VERB")
    if len(lemmas) != 1:
        return set()
    return set(getInflection(lemmas[0], tag="VBP")) | set(
        getInflection(lemmas[0], tag="VBZ")
    )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise URLError("Grammar server redirects are disabled")


def _python_offset(text, units):
    """Java/LanguageTool offsets count UTF-16 units, not Python characters."""
    if type(units) is not int or units < 0:
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
        return a in CONTRACTIONS and b in {
            a[:-1] + "'" + a[-1], a[:-1] + "’" + a[-1]
        }
    if rule not in AGREEMENT_RULES or token is None:
        return False
    if token.pos_ not in {"VERB", "AUX"} or not new.isalpha():
        return False
    # These rule families can also propose a different tense. Never choose it.
    if a in {"am", "is", "are", "was", "were"}:
        return a != b and {a, b} <= (
            {"was", "were"} if a in {"was", "were"} else {"am", "is", "are"}
        )
    present = _present_forms(a)
    return a != b and a in present and b in present


def _validated_matches(response, text):
    """Validate the whole payload before parsing or applying any suggestions."""
    if not isinstance(response, dict) or not isinstance(response.get("matches"), list):
        raise ValueError("Invalid LanguageTool response")
    for match in response["matches"]:
        if not isinstance(match, dict):
            raise ValueError("Invalid grammar match")
        offset, length = match.get("offset"), match.get("length")
        if type(offset) is not int or type(length) is not int or length < 0:
            raise ValueError("Invalid grammar match offsets")
        _python_offset(text, offset)
        _python_offset(text, offset + length)
        rule, choices = match.get("rule"), match.get("replacements", [])
        if (
            not isinstance(rule, dict)
            or not isinstance(rule.get("id"), str)
            or not rule["id"]
            or not isinstance(choices, list)
            or any(not isinstance(r, dict) or not isinstance(r.get("value"), str)
                   for r in choices)
        ):
            raise ValueError("Invalid grammar rule or replacement")
    return response


def _apply(text, edits):
    for edit in sorted(edits, key=lambda e: e["start"], reverse=True):
        text = text[:edit["start"]] + edit["replacement"] + text[edit["end"]:]
    return text


class GrammarCorrector:
    def __init__(self, server="http://127.0.0.1:8786", timeout=5, max_edits=8,
                 contraction_followup=True):
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
        if type(contraction_followup) is not bool:
            raise ValueError("contraction_followup must be boolean")
        self.contraction_followup = contraction_followup

    def check(self, text):
        request = Request(
            self.server + "/v2/check",
            data=urlencode({"text": text, "language": "en-US"}).encode(),
        )
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=self.timeout) as response:
            result = json.load(response)
        return _validated_matches(result, text)

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
            "passes": [],
            "contraction_followup": self.contraction_followup,
        }
        try:
            response = _validated_matches(self.check(text), text)
        except ValueError as error:
            return {**result, "reason": f"Invalid grammar matches: {type(error).__name__}"}
        except (OSError, TimeoutError, HTTPException) as error:
            return {
                **result,
                "reason": f"Grammar service unavailable: {type(error).__name__}",
            }
        result.update(available=True, software=response.get("software", {}))
        try:
            edits, suggestions = self._select(text, response)
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            return {**result, "available": False,
                    "reason": f"Grammar protection unavailable: {type(error).__name__}"}
        result["suggestions"] = suggestions
        output = _apply(text, edits)
        result["passes"].append({"input": text, "output": output,
                                 "edits": [dict(e) for e in edits],
                                 "suggestions": suggestions})
        # Only newly apostrophized contractions can receive another edit.
        # Keep flat edits in original coordinates by composing that edit into
        # its first-pass replacement. Suggestions stay about the ORIGINAL text
        # so grammar_screening never mistakes a later warning for an input one.
        eligible, shift = [], 0
        for edit in sorted(edits, key=lambda e: e["start"]):
            start = edit["start"] + shift
            if edit["rule"] == "EN_CONTRACTION_SPELLING":
                eligible.append((start, start + len(edit["replacement"]), edit))
            shift += len(edit["replacement"]) - (edit["end"] - edit["start"])
        if self.contraction_followup and eligible and len(edits) < self.max_edits:
            try:
                followup = _validated_matches(self.check(output), output)
            except (OSError, ValueError, TimeoutError, HTTPException) as error:
                return {**result, "available": False, "edits": [],
                        "suggestions": [{**s, "applied": False,
                                         "reason": "correction_rolled_back"}
                                        for s in suggestions],
                        "rolled_back": True,
                        "reason": f"Grammar follow-up unavailable: {type(error).__name__}"}
            remaining = self.max_edits - len(edits)
            try:
                repairs, later = self._select(output, followup, eligible, remaining)
            except (OSError, ValueError, RuntimeError, ImportError) as error:
                return {**result, "available": False, "edits": [],
                        "suggestions": [{**s, "applied": False,
                                         "reason": "correction_rolled_back"}
                                        for s in suggestions],
                        "rolled_back": True,
                        "reason": f"Grammar follow-up protection unavailable: {type(error).__name__}"}
            final = _apply(output, repairs)
            result["passes"].append({"input": output, "output": final,
                                     "edits": repairs, "suggestions": later})
            for start, end, original in eligible:
                local = [{**e, "start": e["start"] - start,
                          "end": e["end"] - start}
                         for e in repairs if start <= e["start"] < e["end"] <= end]
                if local:
                    original["first_replacement"] = original["replacement"]
                    original["replacement"] = _apply(original["replacement"], local)
                    original["after"] = original["replacement"]
                    original["followup_rules"] = [e["rule"] for e in local]
            output = final
        result.update(output=output, changed=output != text, edits=edits)
        return result

    def _select(self, text, response, eligible=None, budget=None):
        if not response["matches"]:
            return [], []
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
            for m in re.finditer(r"(?<!\w)'[^'\n]+'(?!\w)", text)
        ]
        protected += [(span.start, span.end) for span in literal_spans(text)]
        edits = []
        suggestions = []
        if budget is None:
            budget = self.max_edits
        for match in response["matches"]:
            start = _python_offset(text, match["offset"])
            end = _python_offset(text, match["offset"] + match["length"])
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
            elif rule in {"EN_A_VS_AN", "EN_CONTRACTION_SPELLING"} and (
                (start and re.match(r"\w", text[start - 1]))
                or (end < len(text) and re.match(r"\w", text[end]))
            ):
                reason = "partial_token"
            elif len(allowed) != 1:
                reason = "unsupported_or_ambiguous_edit"
            elif eligible is not None and (
                rule not in AGREEMENT_RULES
                or not any(a <= start < end <= b for a, b, _ in eligible)
            ):
                reason = "outside_repaired_contraction"
            elif any(start < e["end"] and end > e["start"] for e in edits):
                reason = "overlapping_edit"
            elif len(edits) >= budget:
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
            suggestions.append(record)
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
        return edits, suggestions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8786")
    args = parser.parse_args()
    print(json.dumps(GrammarCorrector(args.server).correct(args.text), indent=2))


if __name__ == "__main__":
    main()
