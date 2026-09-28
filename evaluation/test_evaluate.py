import unittest

from evaluate import evaluate, evaluate_record, readability, word_complexity


class EvaluateTest(unittest.TestCase):
    def test_word_complexity_is_bounded(self):
        score = word_complexity("photosynthesis", ["requires", "energy"])
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_readability_exposes_proposal_features(self):
        metrics = readability("Short words help.")
        self.assertEqual(
            set(metrics), {"flesch", "grade", "fog", "smog", "sentence_length", "word_length"}
        )

    def test_record_contains_formula_outputs_and_guardrail(self):
        result = evaluate_record(
            {"original": "The algorithm has high latency.", "simplified": "The method is slow."}
        )
        self.assertIn("S_lex", result)
        self.assertIn("loss_info", result)
        self.assertIn("U", result)
        self.assertFalse(result["accepted"])

    def test_evaluate_aggregates_records(self):
        metrics = evaluate(
            [
                {"original": "A short sentence.", "simplified": "A short sentence."},
                {"original": "A long sentence.", "simplified": "A long sentence."},
            ]
        )
        self.assertEqual(metrics["records"], 2)
        self.assertIn("accepted_rate", metrics)

    def test_empty_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate([])


if __name__ == "__main__":
    unittest.main()
