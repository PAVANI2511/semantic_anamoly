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
        self.assertFalse(detector.use_nli)

        # Request NLI backend
        detector.backend = "nli_transformer"
        self.assertTrue(detector.use_nli)
        # If sentence_transformers is installed and model loads, backend is 'nli_transformer';
        # otherwise, detector correctly remains on 'lexical_fallback'
        expected_backend = "nli_transformer" if detector.is_nli_active else "lexical_fallback"
        self.assertEqual(detector.backend, expected_backend)

        # When an active model instance is present, backend resolves to 'nli_transformer'
        detector._nli_model = object()
        detector.is_nli_active = True
        detector.backend = "nli_transformer"
        self.assertEqual(detector.backend, "nli_transformer")

        # Explicitly switch back to lexical fallback
        detector.backend = "lexical_fallback"
        self.assertEqual(detector.backend, "lexical_fallback")
        self.assertFalse(detector.use_nli)
        self.assertFalse(detector.is_nli_active)

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

    def test_mocked_nli_loading_success(self):
        """Verifies that when CrossEncoder loads cleanly, backend is 'nli_transformer'."""
        from unittest.mock import MagicMock, patch
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[5.0, -2.0, -1.0]])  # High contradiction logit

        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            # Bypass class cache for isolated testing
            old_cache = ContradictionDetector._cached_nli_model
            old_name = ContradictionDetector._cached_model_name
            ContradictionDetector._cached_nli_model = None
            ContradictionDetector._cached_model_name = None
            try:
                det = ContradictionDetector(use_nli=True, model_name="mock-cross-encoder")
                self.assertTrue(det.is_nli_active)
                self.assertEqual(det.backend, "nli_transformer")
                self.assertIsNotNone(det._nli_model)
            finally:
                ContradictionDetector._cached_nli_model = old_cache
                ContradictionDetector._cached_model_name = old_name

    def test_mocked_nli_loading_failure_falls_back_to_lexical(self):
        """Verifies graceful fallback to lexical when CrossEncoder raises ImportError or Exception."""
        from unittest.mock import patch

        with patch("sentence_transformers.CrossEncoder", side_effect=ImportError("sentence_transformers missing")):
            old_cache = ContradictionDetector._cached_nli_model
            old_name = ContradictionDetector._cached_model_name
            ContradictionDetector._cached_nli_model = None
            ContradictionDetector._cached_model_name = None
            try:
                det = ContradictionDetector(use_nli=True, model_name="unreachable-model")
                self.assertFalse(det.is_nli_active)
                self.assertEqual(det.backend, "lexical_fallback")
                self.assertIsNone(det._nli_model)
            finally:
                ContradictionDetector._cached_nli_model = old_cache
                ContradictionDetector._cached_model_name = old_name

    def test_avoid_redundant_model_loading(self):
        """Verifies that model instance is cached and not reloaded repeatedly."""
        from unittest.mock import MagicMock, patch
        mock_factory = MagicMock(return_value=MagicMock())

        with patch("sentence_transformers.CrossEncoder", mock_factory):
            old_cache = ContradictionDetector._cached_nli_model
            old_name = ContradictionDetector._cached_model_name
            ContradictionDetector._cached_nli_model = None
            ContradictionDetector._cached_model_name = None
            try:
                det1 = ContradictionDetector(use_nli=True, model_name="test-singleton")
                det2 = ContradictionDetector(use_nli=True, model_name="test-singleton")
                # Factory should have been invoked only once
                self.assertEqual(mock_factory.call_count, 1)
                self.assertIs(det1._nli_model, det2._nli_model)
            finally:
                ContradictionDetector._cached_nli_model = old_cache
                ContradictionDetector._cached_model_name = old_name


if __name__ == "__main__":
    unittest.main()

