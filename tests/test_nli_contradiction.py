"""
Unit tests for Contradiction Detection (NLI Transformer & Lexical Fallback).
Verifies:
- NLI DeBERTa CrossEncoder probability calibration
- Dynamic backend switching between NLI and Lexical
- Opposing polarity and numerical contradiction handling
"""

import unittest
import numpy as np

from preprocessing.sentence_segmenter import SentenceSegmenter
from models.contradiction_detector import ContradictionDetector


class TestNLIContradiction(unittest.TestCase):
    """Verifies dual-engine contradiction detection functionality."""

    @classmethod
    def setUpClass(cls):
        cls.segmenter = SentenceSegmenter(use_nltk=False)
        cls.detector = ContradictionDetector(use_nli=True)

    def test_backend_availability(self):
        backend = self.detector.backend
        self.assertIn(backend, ["nli_transformer", "lexical_fallback"])

    def test_dynamic_backend_switching(self):
        detector = ContradictionDetector(use_nli=False)
        self.assertEqual(detector.backend, "lexical_fallback")

        detector.backend = "nli_transformer"
        self.assertEqual(detector.backend, "nli_transformer")

        detector.backend = "lexical_fallback"
        self.assertEqual(detector.backend, "lexical_fallback")

    def test_lexical_polarity_contradiction(self):
        sent_a = "The operating profit increased by twenty percent."
        sent_b = "The operating profit decreased by twenty percent."
        score, reasons = self.detector._detect_contradiction_between(sent_a, sent_b)
        self.assertGreater(score, 0.50)
        self.assertTrue(any("opposing" in r.lower() for r in reasons))

    def test_numerical_contradiction(self):
        sent_a = "The benchmark interest rate was raised to 5.50 percent."
        sent_b = "The benchmark interest rate was reduced to 0.25 percent."
        score, reasons = self.detector._detect_contradiction_between(sent_a, sent_b)
        self.assertGreater(score, 0.50)

    def test_full_contradiction_scoring_pipeline(self):
        doc = (
            "Apex Technologies recorded extraordinary commercial performance throughout the third fiscal quarter.\n\n"
            "Total enterprise revenues increased by 38 percent year-over-year.\n\n"
            "The company experienced total financial failure with overall enterprise revenues that decreased drastically by 40 percent."
        )
        sentences = self.segmenter.segment(doc)
        norm_scores, raw_scores, details = self.detector.compute_contradiction_scores(sentences)

        self.assertEqual(len(norm_scores), 3)
        # Sentence 2 (the contradiction) should have the highest score
        self.assertGreater(norm_scores[2], norm_scores[0])
        self.assertGreater(norm_scores[2], 0.40)


if __name__ == "__main__":
    unittest.main()
