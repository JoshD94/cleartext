"""Locate Markdown code spans whose bytes must survive simplification.

Unfinished fenced blocks are protected too. An unmatched inline marker is plain
text. Space-padded double-backtick quotations without a literal backtick inside
are ambiguous in tokenized prose and are excluded. This is a conservative code
detector, not a complete Markdown parser or a semantic check.
"""

from collections import Counter
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class LiteralSpan:
    start: int
    end: int
    text: str


FENCE = re.compile(r"(?m)^ {0,3}(`{3,}|~{3,})[^\n]*(?:\n|$)")
BACKTICKS = re.compile(r"`+")


def _escaped(text, start):
    cursor = start - 1
    while cursor >= 0 and text[cursor] == "\\":
        cursor -= 1
    return (start - cursor - 1) % 2 == 1


def literal_spans(text):
    """Return non-overlapping code spans, including delimiters and repeated spans."""
    spans = []
    cursor = 0
    while cursor < len(text):
        fence = FENCE.search(text, cursor)
        tick = BACKTICKS.search(text, cursor)
        if fence is not None and (tick is None or fence.start() <= tick.start()):
            marker = fence.group(1)
            closing = re.compile(
                r"(?m)^ {0,3}" + re.escape(marker[0]) + "{" + str(len(marker))
                + r",}[ \t]*\r?(?:\n|$)"
            ).search(text, fence.end())
            end = closing.end() if closing is not None else len(text)
            spans.append(LiteralSpan(fence.start(), end, text[fence.start():end]))
            cursor = end
        elif tick is not None:
            if _escaped(text, tick.start()):
                cursor = tick.end()
                continue
            # A run closes only an equally long run. Longer runs may contain a
            # literal backtick, as in ``print(`value`)``.
            closing = next(
                (match for match in BACKTICKS.finditer(
                    text, tick.end(), fence.start() if fence is not None else len(text))
                 if len(match.group()) == len(tick.group())),
                None,
            )
            if closing is None:
                cursor = tick.end()
                continue
            else:
                end = closing.end()
                inside = text[tick.end():closing.start()]
                if (len(tick.group()) >= 2 and inside.startswith(" ")
                        and inside.endswith(" ") and "`" not in inside):
                    cursor = end
                    continue
            spans.append(LiteralSpan(tick.start(), end, text[tick.start():end]))
            cursor = end
        else:
            break
    return spans


def preserved_literals(original, output):
    """Require the same code spans, byte for byte and with the same multiplicity."""
    before = Counter(span.text for span in literal_spans(original))
    after = Counter(span.text for span in literal_spans(output))
    return {
        "pass": before == after,
        "missing": list((before - after).elements()),
        "added": list((after - before).elements()),
        "semantic_certified": False,
    }
