"""
Unit tests for Benchmark Zero Data Leakage.
Verifies:
- 3-way Document-level split (Train, Val, Test)
- Total absence of document ID overlap across splits
- Total absence of sentence text overlap between train, val, and test
"""

import unittest
from pathlib import Path

from data.dataset_loader import DatasetLoader


class TestBenchmarkLeakage(unittest.TestCase):
    """Verifies strict isolation between training, validation, and testing partitions."""

    def setUp(self):
        json_path = Path("data/datasets/research_benchmark.json")
        if not json_path.exists():
            json_path = Path("data/datasets/synthetic_benchmark.json")
        self.docs = DatasetLoader.load_from_json(json_path)

    def test_three_way_split_zero_document_overlap(self):
        train_docs, val_docs, test_docs = DatasetLoader.split_train_val_test(
            self.docs, train_ratio=0.50, val_ratio=0.25, test_ratio=0.25, random_seed=42
        )

        train_ids = {d.doc_id for d in train_docs}
        val_ids = {d.doc_id for d in val_docs}
        test_ids = {d.doc_id for d in test_docs}

        # Verify no intersections
        self.assertEqual(len(train_ids.intersection(val_ids)), 0, "Train and Val share documents!")
        self.assertEqual(len(train_ids.intersection(test_ids)), 0, "Train and Test share documents!")
        self.assertEqual(len(val_ids.intersection(test_ids)), 0, "Val and Test share documents!")

        # Verify all documents accounted for
        self.assertEqual(len(train_ids) + len(val_ids) + len(test_ids), len(self.docs))

    def test_three_way_split_zero_sentence_leakage(self):
        train_docs, val_docs, test_docs = DatasetLoader.split_train_val_test(
            self.docs, train_ratio=0.50, val_ratio=0.25, test_ratio=0.25, random_seed=42
        )

        train_sentences = {s.text.strip() for d in train_docs for s in d.sentences}
        val_sentences = {s.text.strip() for d in val_docs for s in d.sentences}
        test_sentences = {s.text.strip() for d in test_docs for s in d.sentences}

        # Verify no sentence overlap between partitions
        overlap_tv = train_sentences.intersection(val_sentences)
        overlap_tt = train_sentences.intersection(test_sentences)
        overlap_vt = val_sentences.intersection(test_sentences)

        self.assertEqual(len(overlap_tv), 0, f"Sentence leakage between Train and Val: {overlap_tv}")
        self.assertEqual(len(overlap_tt), 0, f"Sentence leakage between Train and Test: {overlap_tt}")
        self.assertEqual(len(overlap_vt), 0, f"Sentence leakage between Val and Test: {overlap_vt}")


if __name__ == "__main__":
    unittest.main()
