"""
Unit tests for Evaluation Metrics, Dataset Splitting, and Baselines (Section 5 & 6).
"""

import unittest
from pathlib import Path
import numpy as np

from evaluation.metrics import compute_sentence_metrics, compute_token_span_metrics
from data.dataset_loader import DatasetLoader, AnnotatedDocument, AnnotatedSentence
from evaluation.baselines import BaselineRunner


class TestEvaluationMetrics(unittest.TestCase):
    """Tests research evaluation metrics."""

    def test_perfect_classification_metrics(self):
        y_true = [0, 0, 1, 1, 0]
        y_pred = [0, 0, 1, 1, 0]
        y_scores = [0.1, 0.2, 0.8, 0.9, 0.3]

        metrics = compute_sentence_metrics(y_true, y_pred, y_scores)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f1, 1.0)
        self.assertEqual(metrics.tp, 2)
        self.assertEqual(metrics.tn, 3)
        self.assertEqual(metrics.fp, 0)
        self.assertEqual(metrics.fn, 0)
        self.assertEqual(metrics.roc_auc, 1.0)

    def test_token_span_iou_metrics(self):
        # Case A: Exact overlap
        true_spans = [[(10, 20)]]
        pred_spans = [[(10, 20)]]
        doc_lens = [100]
        res = compute_token_span_metrics(true_spans, pred_spans, doc_lens)
        self.assertEqual(res.mean_span_iou, 1.0)
        self.assertEqual(res.token_f1, 1.0)

        # Case B: Partial overlap (10 to 15 vs 10 to 20 -> intersection=5, union=10 -> IoU=0.5)
        pred_partial = [[(10, 15)]]
        res_part = compute_token_span_metrics(true_spans, pred_partial, doc_lens)
        self.assertAlmostEqual(res_part.mean_span_iou, 0.5, places=2)


class TestDatasetLoader(unittest.TestCase):
    """Verifies dataset loading and absence of document leakage."""

    def test_load_synthetic_benchmark(self):
        json_path = Path("data/datasets/synthetic_benchmark.json")
        self.assertTrue(json_path.exists())

        docs = DatasetLoader.load_from_json(json_path)
        self.assertGreaterEqual(len(docs), 5)
        
        # Verify first document
        doc0 = docs[0]
        self.assertEqual(doc0.doc_id, "SYN_01")
        self.assertTrue(doc0.is_synthetic)
        self.assertGreater(len(doc0.sentences), 0)

    def test_zero_document_leakage_split(self):
        docs = DatasetLoader.load_from_json("data/datasets/synthetic_benchmark.json")
        train_docs, test_docs = DatasetLoader.split_train_test(docs, test_ratio=0.40, random_seed=42)

        # Verify sets of document IDs are strictly disjoint
        train_ids = {d.doc_id for d in train_docs}
        test_ids = {d.doc_id for d in test_docs}
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(train_ids) + len(test_ids), len(docs))


class TestBaselines(unittest.TestCase):
    """Verifies baseline anomaly detection models."""

    def test_tfidf_isolation_forest_baseline(self):
        texts = [
            "Deep neural network architecture for NLP.",
            "Transformer attention weights in language models.",
            "Convolutional neural networks for image understanding.",
            "Recipes for baking sourdough bread with flour.",
            "Backpropagation through time in recurrent networks."
        ]
        res = BaselineRunner.run_tfidf_isolation_forest(texts)
        self.assertIn("scores", res)
        self.assertIn("preds", res)
        self.assertEqual(len(res["scores"]), len(texts))
        self.assertGreater(res["time_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()

