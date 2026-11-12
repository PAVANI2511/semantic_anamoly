"""
Unit tests for Span Localization Evaluation.
Verifies:
- Token-level precision, recall, F1
- Exact span match rate
- Partial span match rate
- 1D Intersection-over-Union (IoU) calculation
"""

import unittest
from evaluation.span_evaluator import SpanEvaluator, ComprehensiveSpanMetrics


class TestSpanEvaluation(unittest.TestCase):
    """Verifies span localization evaluation metrics and mathematical correctness."""

    def test_compute_iou(self):
        # Case A: Identical spans
        iou_exact = SpanEvaluator.compute_iou((10, 20), (10, 20))
        self.assertEqual(iou_exact, 1.0)

        # Case B: Disjoint spans
        iou_disjoint = SpanEvaluator.compute_iou((0, 10), (15, 25))
        self.assertEqual(iou_disjoint, 0.0)

        # Case C: 50% overlap: (10, 20) and (15, 25) -> intersection = 5, union = 15 -> 1/3
        iou_overlap = SpanEvaluator.compute_iou((10, 20), (15, 25))
        self.assertAlmostEqual(iou_overlap, 5.0 / 15.0, places=4)

    def test_evaluate_single_document_spans(self):
        gold = [
            {"start_char": 10, "end_char": 20, "category": "topic_deviation", "text": "chocolate"}
        ]
        pred = [
            {"start_char": 10, "end_char": 20, "category": "topic_deviation", "text": "chocolate"}
        ]
        doc_len = 100

        res = SpanEvaluator.evaluate_spans(gold, pred, doc_len)
        self.assertEqual(res["exact_matches"], 1)
        self.assertEqual(res["partial_matches"], 1)
        self.assertEqual(res["char_precision"], 1.0)
        self.assertEqual(res["char_recall"], 1.0)
        self.assertEqual(res["char_f1"], 1.0)
        self.assertAlmostEqual(res["ious"][0], 1.0)

    def test_evaluate_benchmark_collection(self):
        records = [
            {
                "doc_id": "D1",
                "doc_text": "This is a document about machine learning and baking cakes in the oven.",
                "gold_spans": [
                    {"start_char": 45, "end_char": 51, "category": "topic_deviation", "text": "baking"},
                    {"start_char": 66, "end_char": 70, "category": "topic_deviation", "text": "oven"}
                ],
                "pred_spans": [
                    {"start_char": 45, "end_char": 51, "category": "topic_deviation", "text": "baking"}
                ]
            }
        ]

        metrics = SpanEvaluator.evaluate_benchmark(records)
        self.assertIsInstance(metrics, ComprehensiveSpanMetrics)
        self.assertEqual(metrics.total_ground_truth_spans, 2)
        self.assertEqual(metrics.total_predicted_spans, 1)
        self.assertEqual(metrics.exact_match_count, 1)
        self.assertAlmostEqual(metrics.exact_match_rate, 0.50)
        self.assertEqual(metrics.partial_match_count, 1)
        self.assertGreater(metrics.token_precision, 0.0)

    def test_exact_character_boundary_verification(self):
        """Verifies that localized spans strictly match raw sentence slices."""
        from models.span_localizer import SpanLocalizer
        from preprocessing.sentence_segmenter import SentenceSegmenter
        from config.settings import AnomalyCategories

        segmenter = SentenceSegmenter(use_nltk=False)
        localizer = SpanLocalizer()

        doc = "The executive leadership announced total financial failure with overall enterprise revenues that decreased drastically."
        sentences = segmenter.segment(doc)
        s = sentences[0]

        res = localizer.localize_spans(
            sentence=s,
            anomaly_category=AnomalyCategories.CONTRADICTION,
            anomaly_score=0.88,
            is_anomalous=True
        )
        self.assertTrue(res.has_token_spans)
        for sp in res.spans:
            rel_s = sp.start_char - s.start_char
            rel_e = sp.end_char - s.start_char
            # Raw text slice must equal span.text exactly
            self.assertEqual(s.text[rel_s:rel_e], sp.text)
            self.assertEqual(doc[sp.start_char:sp.end_char], sp.text)


if __name__ == "__main__":
    unittest.main()

