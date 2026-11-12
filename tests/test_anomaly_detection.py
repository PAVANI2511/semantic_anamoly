"""
Unit tests for DGSA Anomaly Detection and Scoring (Modules 4, 5, 6, 7).
"""

import unittest
import numpy as np

from preprocessing.sentence_segmenter import SentenceSegmenter
from models.embedding_model import TFIDFEmbeddingModel
from analysis.global_context import GlobalContextAnalyzer
from analysis.local_context import LocalContextAnalyzer
from models.contradiction_detector import ContradictionDetector
from analysis.anomaly_scoring import DGSAScorer
from models.span_localizer import SpanLocalizer
from models.dgsa_model import DGSAModel
from config.settings import AnomalyCategories


class TestAnomalyDetection(unittest.TestCase):
    """Tests mathematical formulation and scoring behavior of DGSA."""

    def setUp(self):
        self.segmenter = SentenceSegmenter(use_nltk=False)
        self.embedding_model = TFIDFEmbeddingModel(max_features=128)
        self.dgsa_model = DGSAModel(embedding_model=self.embedding_model)

    def test_dgsa_scoring_formulation(self):
        scorer = DGSAScorer(alpha=0.4, beta=0.35, gamma=0.25, threshold=0.5)
        g = np.array([0.8, 0.2])
        l = np.array([0.7, 0.1])
        c = np.array([0.9, 0.0])

        comp, eff_a, eff_b, eff_c = scorer.compute_composite_scores(g, l, c)
        
        # Verify weights sum to 1.0
        self.assertAlmostEqual(eff_a + eff_b + eff_c, 1.0, places=5)

        # Expected value for first item: 0.4*0.8 + 0.35*0.7 + 0.25*0.9 = 0.32 + 0.245 + 0.225 = 0.79
        expected_0 = 0.4 * 0.8 + 0.35 * 0.7 + 0.25 * 0.9
        self.assertAlmostEqual(comp[0], expected_0, places=4)

    def test_ablation_weights_normalization(self):
        scorer = DGSAScorer(alpha=0.4, beta=0.35, gamma=0.25)
        g = np.array([0.5])
        l = np.array([0.5])
        c = np.array([0.5])

        # Disable contradiction
        comp, eff_a, eff_b, eff_c = scorer.compute_composite_scores(
            g, l, c, enable_global=True, enable_local=True, enable_contradiction=False
        )
        self.assertAlmostEqual(eff_c, 0.0)
        self.assertAlmostEqual(eff_a + eff_b, 1.0, places=5)
        self.assertAlmostEqual(comp[0], 0.5, places=4)

    def test_contradiction_detection_rule(self):
        detector = ContradictionDetector()
        sent1 = "The corporate profit increased by fifty percent this quarter."
        sent2 = "The corporate profit decreased by fifty percent this quarter."
        score, reasons = detector._detect_contradiction_between(sent1, sent2)
        self.assertGreater(score, 0.5)
        self.assertTrue(any("opposing" in r.lower() for r in reasons))

    def test_span_localizer_distinction(self):
        localizer = SpanLocalizer()
        sample_doc = "Quantum computing relies on superposition. Bake chocolate cake in the hot oven."
        sentences = self.segmenter.segment(sample_doc)
        s_cake = sentences[1]

        # Case A: Low anomaly score -> should return no token spans (sentence-level only)
        res_low = localizer.localize_spans(s_cake, AnomalyCategories.TOPIC_DEVIATION, anomaly_score=0.1)
        self.assertFalse(res_low.has_token_spans)
        self.assertEqual(res_low.method, "none")

        # Case B: High anomaly score with topic deviation
        res_high = localizer.localize_spans(
            s_cake,
            AnomalyCategories.TOPIC_DEVIATION,
            anomaly_score=0.85,
            global_context_tokens={"quantum", "computing", "superposition"}
        )
        self.assertTrue(res_high.has_token_spans)
        self.assertGreater(len(res_high.spans), 0)
        # Verify span offsets point to correct substring
        for sp in res_high.spans:
            self.assertEqual(sample_doc[sp.start_char:sp.end_char], sp.text)

    def test_end_to_end_dgsa_pipeline(self):
        doc = (
            "Neural networks and deep learning models process natural language representations.\n\n"
            "Deep learning language models achieve high accuracy on semantic benchmark tasks.\n\n"
            "To make strawberry ice cream whisk cream and frozen strawberries together in a bowl.\n\n"
            "Neural network models and deep learning architectures learn dense language representations."
        )
        result = self.dgsa_model.analyze_document(doc, custom_threshold=0.40)
        self.assertEqual(result.total_sentences, 4)
        self.assertEqual(len(result.anomaly_scores), 4)
        self.assertEqual(len(result.explanations), 4)

        # The third sentence (strawberry ice cream) should have the highest anomaly score
        scores = result.anomaly_scores
        self.assertEqual(np.argmax(scores), 2)
        self.assertTrue(result.is_anomaly[2])

    def test_empty_document_handling(self):
        """Verifies that empty string or whitespace does not crash the pipeline."""
        res_empty = self.dgsa_model.analyze_document("")
        self.assertEqual(res_empty.total_sentences, 0)
        self.assertEqual(res_empty.num_anomalous, 0)
        self.assertEqual(len(res_empty.anomaly_scores), 0)

        res_ws = self.dgsa_model.analyze_document("   \n\n\t  ")
        self.assertEqual(res_ws.total_sentences, 0)

    def test_single_sentence_document_handling(self):
        """Single sentence document has G(i)=0 and no neighbors; verify zero anomalies."""
        text = "Transformer self-attention models scale compute efficiently."
        res = self.dgsa_model.analyze_document(text)
        self.assertEqual(res.total_sentences, 1)
        self.assertEqual(len(res.anomaly_scores), 1)
        self.assertAlmostEqual(res.anomaly_scores[0], 0.0, places=3)
        self.assertFalse(res.is_anomaly[0])

    def test_explanation_engine_direct_model_evidence(self):
        """Verifies that explanations contain direct_model_evidence and heuristic_indicators."""
        doc = (
            "Machine learning systems learn representations from data.\n\n"
            "Supervised neural networks minimize empirical training risk.\n\n"
            "Astronauts landed on the surface of Mars in a titanium capsule.\n\n"
            "Deep neural network models optimize objective functions through gradient descent."
        )
        res = self.dgsa_model.analyze_document(doc)
        exp = res.explanations[2]
        self.assertTrue(hasattr(exp, "direct_model_evidence"))
        self.assertTrue(hasattr(exp, "heuristic_indicators"))
        self.assertIn("active_contradiction_backend", exp.direct_model_evidence)
        self.assertIn("neural_confidence", exp.direct_model_evidence)


if __name__ == "__main__":
    unittest.main()
