"""Authored dependency fixtures test mechanisms, not simplification accuracy.

This file deliberately imports no parser, pretrained model, torch or numpy.
"""

import sys
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.structure.split_selection import (
    anchored_words,
    event_signatures,
    select_split,
    split_checks,
)


class Morph:
    def __init__(self, tense):
        self.tense = tense

    def get(self, key):
        return {"Tense": [self.tense], "VerbForm": ["Fin"], "Mood": ["Ind"]}.get(key, []) if self.tense else []


class Token:
    def __init__(self, i, word, lemma, pos, dep, head, tense=""):
        self.i, self.text, self.lemma_, self.pos_, self.dep_ = i, word, lemma, pos, dep
        self.head_index, self.morph = head, Morph(tense)
        self.lower_ = word.lower()
        self.is_punct, self.is_space = pos == "PUNCT", False
        self.is_alpha, self.is_quote = word.isalpha(), word in {'"', "'"}
        self.tag_ = ""
        self.children = []

    @property
    def subtree(self):
        yield self
        for child in self.children:
            yield from child.subtree


class Doc(list):
    def __init__(self, specs, boundaries=()):
        super().__init__(Token(i, *spec) for i, spec in enumerate(specs))
        for token in self:
            token.head = self[token.head_index]
            if token.head != token:
                token.head.children.append(token)
        edges = [0, *boundaries, len(self)]
        self.sents = [self[a:b] for a, b in zip(edges, edges[1:])]


def paired(*, split=False, swap_objects=False, swap_tense=False):
    # Alice praised Bob and Carol praised Dave. Repeated verb lemmas expose the
    # information lost when actor, object and tense are separate sets.
    objects = ["Dave", "Bob"] if swap_objects else ["Bob", "Dave"]
    tenses = ["Pres", "Past"] if swap_tense else ["Past", "Pres"]
    return Doc([
        ("Alice", "Alice", "PROPN", "nsubj", 1),
        ("praised", "praise", "VERB", "ROOT", 1, tenses[0]),
        (objects[0], objects[0], "PROPN", "dobj", 1),
        (("." if split else "and"), ("." if split else "and"),
         ("PUNCT" if split else "CCONJ"), ("punct" if split else "cc"), 1),
        ("Carol", "Carol", "PROPN", "nsubj", 5),
        ("praised", "praise", "VERB", ("ROOT" if split else "conj"),
         (5 if split else 1), tenses[1]),
        (objects[1], objects[1], "PROPN", "dobj", 5),
        (".", ".", "PUNCT", "punct", 5),
    ], (4,) if split else ())


def checked(_source, _text, _old, terminated=True):
    return {"pass": terminated, "checks": {"generation_terminated": terminated},
            "failed": [] if terminated else ["generation_terminated"]}


class SelectionTests(unittest.TestCase):
    def test_repeated_predicate_keeps_joint_actor_patient_association(self):
        self.assertEqual(event_signatures(paired()), event_signatures(paired(split=True)))
        self.assertNotEqual(event_signatures(paired()),
                            event_signatures(paired(split=True, swap_objects=True)))

    def test_tense_cannot_move_to_another_occurrence_of_same_verb(self):
        self.assertNotEqual(event_signatures(paired()),
                            event_signatures(paired(split=True, swap_tense=True)))

    def test_duplicate_event_multiplicity_is_not_a_set(self):
        original = paired()
        signatures = event_signatures(original)
        self.assertEqual(sum(signatures.values()), 2)
        self.assertNotEqual(signatures, signatures + signatures)

    def test_boundary_and_removal_preserves_source_order(self):
        self.assertTrue(anchored_words(paired(), paired(split=True)))
        self.assertFalse(anchored_words(paired(), paired(split=True, swap_objects=True)))

    def test_shared_subject_may_repeat_at_coordinated_predicate_only(self):
        old = Doc([
            ("Alice", "Alice", "PROPN", "nsubj", 1),
            ("worked", "work", "VERB", "ROOT", 1, "Past"),
            ("and", "and", "CCONJ", "cc", 1),
            ("rested", "rest", "VERB", "conj", 1, "Past"),
        ])
        new = Doc([
            ("Alice", "Alice", "PROPN", "nsubj", 1),
            ("worked", "work", "VERB", "ROOT", 1, "Past"),
            ("Alice", "Alice", "PROPN", "nsubj", 3),
            ("rested", "rest", "VERB", "ROOT", 3, "Past"),
        ], (2,))
        self.assertTrue(all(split_checks(old, new).values()))
        new[2].lower_ = "bob"
        self.assertFalse(anchored_words(old, new))

    def test_function_word_omission_is_rejected(self):
        old = Doc([
            ("Alice", "Alice", "PROPN", "nsubj", 1),
            ("thought", "think", "VERB", "ROOT", 1, "Past"),
            ("about", "about", "ADP", "prep", 1),
            ("it", "it", "PRON", "pobj", 2),
        ])
        new = Doc([
            ("Alice", "Alice", "PROPN", "nsubj", 1),
            ("thought", "think", "VERB", "ROOT", 1, "Past"),
            ("about", "about", "ADP", "prep", 1),
        ])
        self.assertFalse(anchored_words(old, new))

    def test_detached_attribution_or_relative_scope_is_rejected(self):
        for dependency in ["ccomp", "xcomp", "relcl", "advcl"]:
            old = paired()
            old[5].dep_ = dependency
            self.assertNotEqual(event_signatures(old), event_signatures(paired(split=True)))

    def test_copy_is_not_a_shorter_split(self):
        self.assertFalse(split_checks(paired(), paired())["shorter_split"])

    def test_shared_auxiliary_can_repeat_without_changing_tense(self):
        old = Doc([
            ("Alice", "Alice", "PROPN", "nsubj", 2),
            ("has", "have", "AUX", "aux", 2, "Pres"),
            ("worked", "work", "VERB", "ROOT", 2),
            ("and", "and", "CCONJ", "cc", 2),
            ("rested", "rest", "VERB", "conj", 2),
        ])
        new = Doc([
            ("Alice", "Alice", "PROPN", "nsubj", 2),
            ("has", "have", "AUX", "aux", 2, "Pres"),
            ("worked", "work", "VERB", "ROOT", 2),
            ("Alice", "Alice", "PROPN", "nsubj", 5),
            ("has", "have", "AUX", "aux", 5, "Pres"),
            ("rested", "rest", "VERB", "ROOT", 5),
        ], (3,))
        self.assertTrue(all(split_checks(old, new).values()))

    def test_modal_scope_is_bound_to_event(self):
        old, new = paired(), paired(split=True)
        # A modal transfer to a different occurrence is visible even when the
        # global modal count and predicate-lemma set remain identical.
        old[3].dep_, old[3].tag_, old[3].lemma_ = "aux", "MD", "may"
        new[3].dep_, new[3].tag_, new[3].lemma_ = "aux", "MD", "may"
        new[1].children.remove(new[3])
        new[5].children.append(new[3])
        self.assertNotEqual(event_signatures(old), event_signatures(new))

    def test_quoted_or_bracketed_input_abstains(self):
        for punctuation in ['"', "(", "`", "[", "{"]:
            old = paired()
            old[3].text = punctuation
            old[3].is_quote = punctuation == '"'
            self.assertFalse(split_checks(old, paired(split=True))["unquoted_split"])

    def test_preferred_copy_does_not_hide_constrained_split(self):
        result = select_split("copy", [{"text": "copy", "terminated": True},
                                      {"text": "split", "terminated": True}], paired(),
                              parse=lambda text: paired(split=text == "split"), check=checked)
        self.assertEqual(result["output"], "split")
        self.assertEqual(result["selected_beam"], 1)
        self.assertEqual(len(result["candidates"]), 2)

    def test_new_policy_cannot_override_existing_guard_failure(self):
        result = select_split("copy", [{"text": "split", "terminated": False}], paired(),
                              parse=lambda _: paired(split=True), check=checked)
        self.assertEqual(result["output"], "copy")
        self.assertIn("generation_terminated", result["candidates"][0]["checks"]["failed"])

    def test_model_order_breaks_equal_shortening_ties(self):
        result = select_split("copy", [{"text": "first", "terminated": True},
                                      {"text": "second", "terminated": True}], paired(),
                              parse=lambda _: paired(split=True), check=checked)
        self.assertEqual(result["selected_beam"], 0)

    def test_shorter_longest_sentence_beats_higher_model_rank(self):
        def three(split_points=()):
            return Doc([
                ("Alice", "Alice", "PROPN", "nsubj", 1),
                ("worked", "work", "VERB", "ROOT", 1, "Past"),
                (("." if 3 in split_points else "and"), "and",
                 ("PUNCT" if 3 in split_points else "CCONJ"),
                 ("punct" if 3 in split_points else "cc"), 1),
                ("Bob", "Bob", "PROPN", "nsubj", 4),
                ("rested", "rest", "VERB", ("ROOT" if 3 in split_points else "conj"),
                 (4 if 3 in split_points else 1), "Past"),
                (("." if 6 in split_points else "and"), "and",
                 ("PUNCT" if 6 in split_points else "CCONJ"),
                 ("punct" if 6 in split_points else "cc"), 4),
                ("Carol", "Carol", "PROPN", "nsubj", 7),
                ("slept", "sleep", "VERB", ("ROOT" if 6 in split_points else "conj"),
                 (7 if 6 in split_points else 4), "Past"),
            ], split_points)
        result = select_split("copy", [{"text": "two", "terminated": True},
                                      {"text": "three", "terminated": True}], three(),
                              parse=lambda text: three((3,) if text == "two" else (3, 6)),
                              check=checked)
        self.assertEqual(result["selected_beam"], 1)

    def test_empty_pool_returns_copy(self):
        result = select_split("copy", [], paired(), parse=lambda _: paired(), check=checked)
        self.assertEqual(result["selected_origin"], "identity_fallback")


class ReplayTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "scripts/structure_sprint_replay.py"
        spec = importlib.util.spec_from_file_location("structure_sprint_replay", path)
        self.replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.replay)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "saved.jsonl"
        self.input.write_text(json.dumps({"id": "authored", "source": "copy",
            "search_details": {"model_candidates": [{"text": "copy", "terminated": True}]}}) + "\n")
        self.run = self.root / "frozen"

    def test_freeze_copies_exact_raw_bytes_and_refuses_overwrite(self):
        manifest = self.replay.freeze([self.input], self.run)
        self.assertEqual(manifest["inputs"][0]["rows"], 1)
        self.assertEqual((self.run / "input-0.jsonl").read_bytes(), self.input.read_bytes())
        with self.assertRaises(FileExistsError):
            self.replay.freeze([self.input], self.run)

    def test_modified_raw_snapshot_fails_before_parser_import(self):
        self.replay.freeze([self.input], self.run)
        (self.run / "input-0.jsonl").write_text("{}\n")
        # Other tests may already have imported spaCy in this pytest process.
        # Reject an attempted import here rather than depending on test order.
        import builtins
        original_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name in {"spacy", "cleartext.features"}:
                raise AssertionError("parser import before snapshot validation")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", checked_import):
            with self.assertRaisesRegex(ValueError, "raw candidates changed"):
                self.replay.replay(self.run)

    def test_modified_code_fails_before_parser_import(self):
        self.replay.freeze([self.input], self.run)
        self.replay.code_hashes = lambda: {}
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.replay.replay(self.run)

    def test_changed_parser_package_fails_before_model_import(self):
        self.replay.freeze([self.input], self.run)
        self.replay.parser_environment = lambda: {"spacy": "changed"}
        with self.assertRaisesRegex(ValueError, "Parser package versions changed"):
            self.replay.replay(self.run)

    def test_missing_eos_provenance_cannot_be_replayed(self):
        self.input.write_text(json.dumps({"search_details": {
            "model_candidates": [{"text": "copy"}]}}) + "\n")
        with self.assertRaisesRegex(ValueError, "termination/text"):
            self.replay.freeze([self.input], self.run)


if __name__ == "__main__":
    unittest.main()
