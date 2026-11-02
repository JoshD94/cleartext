"""Code preservation is byte-level behavior, independent of semantic labels."""

import pytest

from cleartext.protected_literals import literal_spans, preserved_literals


@pytest.mark.parametrize("text,expected", [
    ("Use `commence()` here.", ["`commence()`"]),
    ("Use ``print(`value`)`` here.", ["``print(`value`)``"]),
    ("Use `x` and `x`.", ["`x`", "`x`"]),
    ("A sentence isn't code.", []),
    (r"An escaped \` mark.", []),
    ("They called it `` difficult '' but it worked.", []),
    ("This is ` quoted text ' in a tokenized corpus.", []),
    ("`` first '' and `` second '' are quoted words.", []),
    ("Use `` spaced code `` here.", []),
    ("They called it `` unusual `` but it worked.", []),
    ("Use `` `value` `` here.", ["`` `value` ``"]),
    ("Use ` x ` with ' word ' as a quote.", ["` x `"]),
    ("Use ` value = 'x' ` here.", ["` value = 'x' `"]),
    ("Before.\n```python\nprint(`x`)\n```\nAfter.", ["```python\nprint(`x`)\n```\n"]),
    ("~~~sh\necho hello\n~~~~\n", ["~~~sh\necho hello\n~~~~\n"]),
    ("Before `unfinished\nAfter.", []),
    ("```python\nunfinished", ["```python\nunfinished"]),
    ("Use `unfinished\n```py\n`x`\n```\n", ["```py\n`x`\n```\n"]),
])
def test_span_boundaries_preserve_complete_or_unfinished_code(text, expected):
    spans = literal_spans(text)
    assert [span.text for span in spans] == expected
    assert all(text[span.start:span.end] == span.text for span in spans)


def test_surrounding_prose_can_change():
    assert preserved_literals("Please commence `execute()` now.",
                              "Please start `execute()` now.")["pass"]


@pytest.mark.parametrize("output", [
    "Use `start()`.", "Use commence().", "Use `Commence()`.",
    "Use `commence( )`.", "Use `commence()` and `commence()`.",
])
def test_content_delimiters_case_spacing_and_multiplicity_must_survive(output):
    assert not preserved_literals("Use `commence()`.", output)["pass"]


def test_removing_one_repeated_occurrence_fails():
    assert not preserved_literals("Use `x` and `x`.", "Use `x`.")["pass"]


def test_introducing_code_is_also_a_change():
    assert not preserved_literals("Use x.", "Use `x`.")["pass"]


@pytest.mark.parametrize("marker", ["```", "~~~"])
def test_crlf_fence_does_not_capture_following_prose(marker):
    code = marker + "py\r\nx=1\r\n" + marker + "\r\n"
    source = code + "Please commence now."
    assert [s.text for s in literal_spans(source)] == [code]
    assert preserved_literals(source, source.replace("commence", "start"))["pass"]


@pytest.mark.parametrize("output", ["Use ` changed ` with ' word ' as a quote.",
                                   "Use  x  with ' word ' as a quote."])
def test_later_apostrophe_cannot_hide_code_damage(output):
    assert not preserved_literals("Use ` x ` with ' word ' as a quote.", output)["pass"]
