"""Native parser regressions. Parent schedules this file, estimated 1 GiB RSS.

Authored cases verify gates and selection behavior, not semantic accuracy.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.features import nlp
from cleartext.structure.search import select_candidate
from cleartext.structure.seq2seq import generation_checks
from cleartext.structure.split_selection import event_signatures, split_checks


SOURCE = "Alice reviewed the report and Bob checked the figures."
SPLIT = "Alice reviewed the report. Bob checked the figures."


def proposal(text):
    return {"text": text, "terminated": True}


class NativeTests(unittest.TestCase):
    def test_structure_checks_preserve_code_delimiters(self):
        source = "Alice ran `print(1)` and Bob checked the result."
        output = "Alice ran print(1). Bob checked the result."
        self.assertFalse(generation_checks(source, output)["checks"]["code_literals"])

    def test_default_copy_policy_is_unchanged(self):
        result = select_candidate(SOURCE, [proposal(SOURCE), proposal(SPLIT)])
        self.assertEqual(result["output"], SOURCE)

    def test_opt_in_can_select_hidden_split(self):
        result = select_candidate(SOURCE, [proposal(SOURCE), proposal(SPLIT)],
                                  policy="source_anchored_split")
        self.assertEqual(result["output"], SPLIT)
        self.assertEqual(result["selected_beam"], 1)

    def test_opt_in_retains_existing_punctuation_repair(self):
        source = "Alice reviewed the report, Bob checked the figures."
        baseline = select_candidate(source, [proposal(source)])
        self.assertNotEqual(baseline["output"], source)
        candidate = select_candidate(source, [proposal(source)], policy="source_anchored_split")
        self.assertEqual(candidate["output"], baseline["output"])
        self.assertEqual(candidate["selection_route"], "retained_baseline_edit")

    def test_no_extra_split_keeps_baseline_spacing_repair(self):
        source, cleaned = "Alice  worked.", "Alice worked."
        result = select_candidate(source, [proposal(cleaned)], policy="source_anchored_split")
        self.assertEqual(result["output"], cleaned)
        self.assertEqual(result["selection_route"], "retained_baseline_copy")

    def test_participants_stay_paired_with_same_predicate(self):
        old = nlp()("The teacher praised the student and the coach praised the player.")
        good = nlp()("The teacher praised the student. The coach praised the player.")
        bad = nlp()("The teacher praised the player. The coach praised the student.")
        self.assertEqual(event_signatures(old), event_signatures(good))
        self.assertNotEqual(event_signatures(old), event_signatures(bad))

    def test_parser_attachment_disagreement_conservatively_abstains(self):
        source = "Alice praised Bob and Carol praised Dave."
        split = "Alice praised Bob. Carol praised Dave."
        self.assertNotEqual(event_signatures(nlp()(source)), event_signatures(nlp()(split)))
        result = select_candidate(source, [proposal(split)], policy="source_anchored_split")
        self.assertEqual(result["output"], source)

    def test_tense_stays_with_participant(self):
        old = nlp()("Alice worked and Bob works.")
        bad = nlp()("Alice works. Bob worked.")
        self.assertNotEqual(event_signatures(old), event_signatures(bad))

    def test_dropped_preposition_object_cannot_pass_word_alignment(self):
        old = nlp()("This is a topic that I had never thought about in this way.")
        bad = nlp()("This is a topic. I had never thought about in this way.")
        self.assertFalse(split_checks(old, bad)["source_anchored_words"])

    def test_attribution_condition_causality_and_negation_abstain(self):
        cases = [
            ("Alice said that Bob left and Carol stayed.", "Alice said that Bob left. Carol stayed."),
            ("If Alice leaves, Bob stays and Carol waits.", "If Alice leaves, Bob stays. Carol waits."),
            ("Alice worked so Bob rested.", "Alice worked. Bob rested."),
            ("Alice did not work and rest.", "Alice did not work. Alice rested."),
        ]
        for source, bad in cases:
            with self.subTest(source=source):
                result = select_candidate(source, [proposal(source), proposal(bad)], policy="source_anchored_split")
                self.assertEqual(result["output"], source)

    def test_unknown_policy_fails_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            select_candidate(SOURCE, [], policy="unsupported")


if __name__ == "__main__":
    unittest.main()
