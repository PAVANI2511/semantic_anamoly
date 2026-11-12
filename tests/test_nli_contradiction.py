"""
Unit tests for Contradiction Detection (NLI Transformer & Lexical Fallback).
Verifies:
- NLI DeBERTa CrossEncoder probability calibration
- Safe lazy loading and dependency isolation
- Graceful fallback when sentence_transformers is uninstalled
- Graceful fallback when model loading or download fails
- Graceful fallback when NLI inference raises an exception
- Dynamic backend switching between NLI and Lexical
- Opposing polarity and numerical contradiction handling
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from preprocessing.sentence_segmenter import SentenceSegmenter
from models.contradiction_detector import ContradictionDetector


class TestNLIContradiction(unittest.TestCase):
    """Verifies dual-engine contradiction detection functionality and test reliability."""

    @classmethod
    def setUpClass(cls):
        cls.segmenter = SentenceSegmenter(use_nltk=False)
        cls.detector = ContradictionDetector(use_nli=True)

    def test_nli_dependency_check(self):
        """Verifies that dependency checking executes safely without crashing."""
        avail = ContradictionDetector.is_nli_dependency_available()
        self.assertIsInstance(avail, bool)

    def test_backend_availability(self):
        """Backend must accurately report the active backend name."""
        backend = self.detector.backend
        self.assertIn(backend, ["nli_transformer", "lexical_fallback"])

    def test_dynamic_backend_switching(self):
        """Verifies clean switching between lexical and transformer backends."""
        detector = ContradictionDetector(use_nli=False)
        self.assertEqual(detector.backend, "lexical_fallback")
        self.assertFalse(detector.use_nli)

        # Request NLI backend
        detector.backend = "nli_transformer"
        self.assertTrue(detector.use_nli)
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
        """Verifies rule-based detection on opposing antonym pairs."""
        sent_a = "The operating profit increased by twenty percent."
        sent_b = "The operating profit decreased by twenty percent."
        score, reasons = self.detector._detect_lexical_contradiction(sent_a, sent_b)
        self.assertGreater(score, 0.50)
        self.assertTrue(any("opposing" in r.lower() for r in reasons))

    def test_numerical_contradiction(self):
        """Verifies numerical discrepancy detection on metric parameters."""
        sent_a = "The benchmark interest rate was raised to 5.50 percent."
        sent_b = "The benchmark interest rate was reduced to 0.25 percent."
        score, reasons = self.detector._detect_lexical_contradiction(sent_a, sent_b)
        self.assertGreater(score, 0.50)

    def test_full_contradiction_scoring_pipeline(self):
        """Verifies end-to-end contradiction scoring pipeline across multi-sentence documents."""
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
        """
        Verifies that when CrossEncoder loads cleanly, backend is 'nli_transformer'.
        Patches at the class import boundary to guarantee execution even without sentence_transformers.
        """
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[5.0, -2.0, -1.0]])  # High contradiction logit
        mock_cross_encoder_cls = MagicMock(return_value=mock_model)

        with patch.object(ContradictionDetector, "_import_cross_encoder", return_value=mock_cross_encoder_cls):
            old_cache = ContradictionDetector._cached_nli_model
            old_name = ContradictionDetector._cached_model_name
            ContradictionDetector._cached_nli_model = None
            ContradictionDetector._cached_model_name = None
            try:
                det = ContradictionDetector(use_nli=True, model_name="mock-cross-encoder")
                self.assertTrue(det.is_nli_active)
                self.assertEqual(det.backend, "nli_transformer")
                self.assertIsNotNone(det._nli_model)

                # Test prediction inference
                probs = det._predict_pairs_nli([("Sentence A", "Sentence B")])
                self.assertIsNotNone(probs)
                self.assertGreater(probs[0, 0], 0.90)  # Contradiction probability high
            finally:
                ContradictionDetector._cached_nli_model = old_cache
                ContradictionDetector._cached_model_name = old_name

    def test_mocked_nli_dependency_unavailable(self):
        """
        Verifies graceful fallback to lexical when sentence_transformers is NOT installed.
        Patches _import_cross_encoder to raise ImportError without depending on the installed package.
        """
        with patch.object(
            ContradictionDetector,
            "_import_cross_encoder",
            side_effect=ImportError("No module named 'sentence_transformers'")
        ):
            old_cache = ContradictionDetector._cached_nli_model
            old_name = ContradictionDetector._cached_model_name
            ContradictionDetector._cached_nli_model = None
            ContradictionDetector._cached_model_name = None
            try:
                det = ContradictionDetector(use_nli=True, model_name="test-uninstalled-dep")
                self.assertFalse(det.is_nli_active)
                self.assertEqual(det.backend, "lexical_fallback")
                self.assertIsNone(det._nli_model)

                # Detection must still succeed using lexical fallback
                s_a = "Revenues increased by 20 percent."
                s_b = "Revenues decreased by 20 percent."
                score, reasons = det._detect_contradiction_between(s_a, s_b)
                self.assertGreater(score, 0.50)
            finally:
                ContradictionDetector._cached_nli_model = old_cache
                ContradictionDetector._cached_model_name = old_name

    def test_mocked_nli_loading_failure_falls_back_to_lexical(self):
        """Verifies graceful fallback to lexical when model weights cannot be downloaded or loaded."""
        mock_cls = MagicMock(side_effect=OSError("Model checkpoint not found or corrupted"))
        with patch.object(ContradictionDetector, "_import_cross_encoder", return_value=mock_cls):
            old_cache = ContradictionDetector._cached_nli_model
            old_name = ContradictionDetector._cached_model_name
            ContradictionDetector._cached_nli_model = None
            ContradictionDetector._cached_model_name = None
            try:
                det = ContradictionDetector(use_nli=True, model_name="unreachable-remote-model")
                self.assertFalse(det.is_nli_active)
                self.assertEqual(det.backend, "lexical_fallback")
                self.assertIsNone(det._nli_model)
            finally:
                ContradictionDetector._cached_nli_model = old_cache
                ContradictionDetector._cached_model_name = old_name

    def test_mocked_nli_inference_failure_falls_back_to_lexical(self):
        """
        Verifies that if CrossEncoder.predict raises an unexpected RuntimeError (e.g. CUDA OOM),
        the pipeline does not crash and seamlessly falls back to lexical contradiction scoring.
        """
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CUDA out of memory during batch forward pass")
        mock_cross_encoder_cls = MagicMock(return_value=mock_model)

        with patch.object(ContradictionDetector, "_import_cross_encoder", return_value=mock_cross_encoder_cls):
            old_cache = ContradictionDetector._cached_nli_model
            old_name = ContradictionDetector._cached_model_name
            ContradictionDetector._cached_nli_model = None
            ContradictionDetector._cached_model_name = None
            try:
                det = ContradictionDetector(use_nli=True, model_name="mock-oom-model")
                self.assertTrue(det.is_nli_active)

                # Predict pairs should catch the exception and return None
                probs = det._predict_pairs_nli([("Sentence 1", "Sentence 2")])
                self.assertIsNone(probs)

                # End-to-end compute_contradiction_scores must not crash and must detect via lexical fallback
                s_a = "Revenues increased significantly."
                s_b = "Revenues decreased significantly."
                sentences = self.segmenter.segment(f"{s_a}\n\n{s_b}")
                norm_scores, _, details = det.compute_contradiction_scores(sentences)
                self.assertGreater(norm_scores[1], 0.50)
                self.assertEqual(details[1]["backend"], "lexical_fallback")
            finally:
                ContradictionDetector._cached_nli_model = old_cache
                ContradictionDetector._cached_model_name = old_name

    def test_avoid_redundant_model_loading(self):
        """Verifies that model instance is cached and not reloaded repeatedly."""
        mock_model = MagicMock()
        mock_factory = MagicMock(return_value=mock_model)

        with patch.object(ContradictionDetector, "_import_cross_encoder", return_value=mock_factory):
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
