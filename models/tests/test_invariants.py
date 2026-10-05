import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cleartext.features import nlp, target_token
from cleartext.lexical import guardrails, generate, LexicalPipeline


class Fixed:
    def predict(self, rows):
        return np.full(len(rows), 0.3)


def test_exact_offset_repeated_word():
    doc = nlp()("The bank faced the river bank.")
    assert target_token(doc, "bank", 25).idx == 25


def test_keep_is_byte_preserving():
    text = "  The result may change.\n"
    doc = nlp()(text)
    t = next(t for t in doc if t.text == "result")
    out = LexicalPipeline(Fixed()).rank(doc, t, [])
    assert out["output"] == text and not out["changed"]


def test_guardrail_changes():
    assert not guardrails("The cost is 5 dollars.", "The cost is 50 dollars.")["pass"]
    assert not guardrails("The firm may expand.", "The firm will expand.")["pass"]
    assert not guardrails(
        "The result is not significant.", "The result is significant."
    )["pass"]
    assert not guardrails("The neural network learns.", "The nerve network learns.")[
        "pass"
    ]
    assert not guardrails("Alice defeated Bob.", "Bob defeated Alice.")["pass"]


def test_identity_passes():
    s = "The bank may approve a loan of 50 dollars."
    assert guardrails(s, s)["pass"]


def test_candidates_preserve_inflection():
    doc = nlp()("They purchased the equipment.")
    token = next(t for t in doc if t.text == "purchased")
    candidates = generate(doc, token)
    assert all(c["word"] not in {"purchase", "buy"} for c in candidates)


def test_passive_to_adjective_subject_is_same_role():
    assert guardrails(
        "Although this sentence is convoluted, the idea is simple.",
        "Although this sentence is complex, the idea is simple.",
    )["checks"]["argument_roles"]
    assert not guardrails("Alice defeated Bob.", "Bob defeated Alice.")["pass"]
