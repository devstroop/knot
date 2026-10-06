import unittest

from training.eval_metrics import ECE_BINS, _record, ece, summarize


class EceTests(unittest.TestCase):
    def test_perfectly_calibrated_bins_score_zero(self):
        confidences = [0.9] * 10
        hits = [True] * 9 + [False]
        self.assertAlmostEqual(ece(confidences, hits), 0.0)

    def test_confident_and_wrong_scores_one(self):
        self.assertAlmostEqual(ece([1.0, 1.0], [False, False]), 1.0)

    def test_empty_input_is_none(self):
        self.assertIsNone(ece([], []))

    def test_rejects_mismatched_or_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            ece([0.5], [])
        with self.assertRaisesRegex(ValueError, "positive"):
            ece([0.5], [True], bins=0)
        with self.assertRaisesRegex(ValueError, "probability"):
            ece([1.5], [True])
        with self.assertRaisesRegex(ValueError, "probability"):
            ece([float("nan")], [True])

    def test_single_bin_edges_land_in_the_last_bucket(self):
        self.assertAlmostEqual(ece([1.0], [False]), 1.0)
        self.assertAlmostEqual(ece([0.0], [True]), 1.0)

    def test_default_bin_count_is_ten(self):
        self.assertEqual(ECE_BINS, 10)


class SummarizeTests(unittest.TestCase):
    def records(self):
        return [
            _record("AmazonScience/massive", "choice", True, 0.8, 0.1),
            _record("AmazonScience/massive", "choice", False, 0.6, 0.5),
            _record("PolyAI/banking77", "score", True, 0.9, 0.2, 0),
            _record("PolyAI/banking77", "score", False, 0.7, 0.3, 2),
            _record("AmazonScience/massive", "noul", True, 0.95, 0.05),
        ]

    def test_overall_and_per_primitive_metrics(self):
        report = summarize(self.records())
        self.assertEqual(report["case_count"], 5)
        self.assertAlmostEqual(report["accuracy"], 0.6)
        self.assertEqual(report["by_primitive"]["choice"]["count"], 2)
        self.assertAlmostEqual(report["by_primitive"]["choice"]["accuracy"], 0.5)
        self.assertAlmostEqual(report["by_primitive"]["score"]["mae"], 1.0)
        self.assertAlmostEqual(report["by_primitive"]["noul"]["brier"], 0.05)
        self.assertIn("ece", report["by_primitive"]["choice"])

    def test_per_source_breakdown(self):
        report = summarize(self.records())
        self.assertEqual(set(report["by_source"]), {
            "AmazonScience/massive", "PolyAI/banking77",
        })
        massive = report["by_source"]["AmazonScience/massive"]
        self.assertEqual(massive["count"], 3)
        self.assertAlmostEqual(massive["accuracy"], 2 / 3)
        self.assertEqual(massive["by_primitive"]["choice"]["count"], 2)
        self.assertEqual(massive["by_primitive"]["noul"]["count"], 1)
        self.assertNotIn("score", massive["by_primitive"])

    def test_rejects_empty_and_malformed_records(self):
        with self.assertRaisesRegex(ValueError, "no evaluation records"):
            summarize([])
        with self.assertRaisesRegex(ValueError, "malformed"):
            summarize([{"primitive": "choice"}])
        bad = _record("AmazonScience/massive", "choice", True, 0.5, 0.1)
        bad["primitive"] = "rating"
        with self.assertRaisesRegex(ValueError, "unknown primitive"):
            summarize([bad])

    def test_record_helper_rejects_unknown_primitive(self):
        with self.assertRaisesRegex(ValueError, "unknown primitive"):
            _record("AmazonScience/massive", "rating", True, 0.5, 0.1)


if __name__ == "__main__":
    unittest.main()
