"""
Regression test suite for the 38-sentence AI Research Project Report.
Verifies fix for false positive explosion, markdown heading handling,
technical terminology preservation, and sensitivity to genuine injected anomalies.
"""

import unittest
import numpy as np

from models.dgsa_model import DGSAModel
from config.settings import AnomalyCategories


class TestAIReportRegression(unittest.TestCase):
    """
    Validates that the 38-sentence AI Research Project Report is accurately analyzed:
    - 0 false positives on the clean document
    - Headings and list numbers preserved as structure and never flagged as contradictions
    - Technical terminology preserved without false entity or contextual anomalies
    - Deliberate anomalies across categories are reliably caught
    """

    @classmethod
    def setUpClass(cls):
        cls.sentences_38 = [
            "# Artificial Intelligence Research Project Report",
            "## 1.",
            "Introduction",
            "Artificial Intelligence (AI) is transforming modern industries by enabling machines to perform tasks that traditionally require human intelligence.",
            "Machine learning, deep learning, and natural language processing are widely used in healthcare, education, finance, and transportation.",
            "Machine learning systems learn patterns from historical data and use those patterns to make predictions.",
            "The quality of the training data plays an important role in determining model performance.",
            "## 2.",
            "Project Overview",
            "The proposed project focuses on developing an intelligent document analysis system that identifies semantic inconsistencies in text documents.",
            "The system analyzes document context, sentence relationships, and semantic contradictions.",
            "The application uses Python for backend processing and Streamlit for its user interface.",
            "Sentence Transformers are used to generate semantic embeddings.",
            "## 3.",
            "Methodology",
            "The system follows three major stages:",
            "1.",
            "Global Context Analysis: Understands the overall topic and meaning of the document.",
            "2.",
            "Local Context Analysis: Examines relationships between neighboring sentences.",
            "3.",
            "Contradiction Detection: Identifies statements that conflict with previously established information.",
            "The three components are combined to calculate an anomaly score for each sentence.",
            "## 4.",
            "Experimental Results",
            "The system was evaluated using a dataset containing 500 documents.",
            "The dataset consisted of academic reports, business documents, and technical articles.",
            "The proposed model achieved an accuracy of 94% and an F1-score of 91%.",
            "The evaluation was conducted using a separate test dataset that was not used during model training.",
            "## 5.",
            "Technical Implementation",
            "Python was selected because of its extensive machine learning ecosystem.",
            "PyTorch supports model development, while Hugging Face Transformers provides access to pretrained language models.",
            "The application stores uploaded documents and generates downloadable reports.",
            "## 6.",
            "Conclusion",
            "The proposed system demonstrates how semantic analysis can help identify inconsistencies in documents.",
            "Future improvements may include multilingual document analysis, larger evaluation datasets, and improved anomaly localization."
        ]
        cls.raw_document = "\n\n".join(cls.sentences_38)
        cls.model = DGSAModel()
        cls.clean_result = cls.model.analyze_document(cls.raw_document)

    def test_clean_document_zero_false_positives(self):
        """Verifies that the clean 38-sentence report produces 0 anomalies (down from 19)."""
        self.assertEqual(self.clean_result.total_sentences, 38)
        self.assertEqual(self.clean_result.num_anomalous, 0)
        self.assertEqual(self.clean_result.anomaly_percentage, 0.0)
        for i, is_anom in enumerate(self.clean_result.is_anomaly):
            self.assertFalse(
                is_anom,
                f"Sentence #{i+1} ('{self.clean_result.sentences[i].text}') falsely flagged as anomaly "
                f"with score {self.clean_result.anomaly_scores[i]:.3f} (category: {self.clean_result.categories[i]})"
            )

    def test_markdown_headings_and_structure_preservation(self):
        """Verifies that markdown headings and list markers are tagged as structural and have 0.0 anomaly score."""
        structural_indices = [0, 1, 2, 7, 8, 13, 14, 16, 18, 20, 23, 24, 29, 30, 34, 35]
        for idx in structural_indices:
            sent = self.clean_result.sentences[idx]
            is_struct = getattr(sent, "is_structural", False) or getattr(sent, "is_heading", False)
            self.assertTrue(
                is_struct,
                f"Sentence #{idx+1} ('{sent.text}') should be recognized as structural or heading"
            )
            self.assertEqual(self.clean_result.anomaly_scores[idx], 0.0)
            self.assertEqual(self.clean_result.categories[idx], AnomalyCategories.NORMAL)

    def test_technical_sentences_not_flagged(self):
        """Verifies that technical implementation statements are safely below threshold."""
        # Sentences #12, #13, #18, #20, #32, #33
        tech_indices = [11, 12, 17, 19, 31, 32]
        for idx in tech_indices:
            score = self.clean_result.anomaly_scores[idx]
            sent_text = self.clean_result.sentences[idx].text
            self.assertLess(
                score, 0.45,
                f"Technical sentence #{idx+1} ('{sent_text[:40]}...') exceeded threshold with score {score:.3f}"
            )

    def test_injected_semantic_contradiction_detected(self):
        """Verifies that inserting a direct contradiction into the report is caught reliably."""
        corrupted_sentences = list(self.sentences_38)
        # Follow sentence 28 (94% accuracy) with a direct opposing contradiction
        corrupted_sentences.insert(
            28,
            "The proposed model completely failed to function and enterprise accuracy decreased drastically to 12 percent."
        )
        corrupted_doc = "\n\n".join(corrupted_sentences)
        res = self.model.analyze_document(corrupted_doc)

        self.assertGreater(res.num_anomalous, 0)
        # Verify the injected sentence (index 28) is flagged
        self.assertTrue(res.is_anomaly[28])
        self.assertEqual(res.categories[28], AnomalyCategories.CONTRADICTION)

    def test_injected_topic_deviation_detected(self):
        """Verifies that inserting an alien out-of-domain sentence is caught as Topic Deviation."""
        corrupted_sentences = list(self.sentences_38)
        corrupted_sentences.insert(
            12,
            "The medieval blacksmith forged heavy iron swords and protective armor for the royal cavalry before sunset."
        )
        corrupted_doc = "\n\n".join(corrupted_sentences)
        res = self.model.analyze_document(corrupted_doc)

        self.assertGreater(res.num_anomalous, 0)
        self.assertTrue(res.is_anomaly[12])
        self.assertIn(
            res.categories[12],
            {AnomalyCategories.TOPIC_DEVIATION, AnomalyCategories.CONTEXTUAL_INCONSISTENCY}
        )

    def test_injected_numerical_inconsistency_detected(self):
        """Verifies that conflicting numerical claims on the evaluation dataset are caught."""
        corrupted_sentences = list(self.sentences_38)
        # Sentence 26 states "evaluated using a dataset containing 500 documents"
        # Insert a sentence asserting "evaluated on a tiny dataset of only 3 documents"
        corrupted_sentences.insert(
            27,
            "The system was evaluated using a dataset containing only 3 documents."
        )
        corrupted_doc = "\n\n".join(corrupted_sentences)
        res = self.model.analyze_document(corrupted_doc)

        self.assertGreater(res.num_anomalous, 0)
        self.assertTrue(res.is_anomaly[27])

    def test_injected_entity_inconsistency_detected(self):
        """Verifies that inserting an alien entity into the document is detected."""
        corrupted_sentences = list(self.sentences_38)
        corrupted_sentences.insert(
            11,
            "Count Dracula summoned his vampire legions from Castle Bran to besiege the mountain fortress."
        )
        corrupted_doc = "\n\n".join(corrupted_sentences)
        res = self.model.analyze_document(corrupted_doc)

        self.assertGreater(res.num_anomalous, 0)
        self.assertTrue(res.is_anomaly[11])

    def test_span_localization_no_generic_words(self):
        """Verifies that localized spans never attribute anomalies to generic verbs or nouns."""
        corrupted_sentences = list(self.sentences_38)
        corrupted_sentences.insert(
            10,
            "The company experienced total financial failure with overall enterprise revenues that decreased drastically."
        )
        corrupted_doc = "\n\n".join(corrupted_sentences)
        res = self.model.analyze_document(corrupted_doc)

        self.assertTrue(res.is_anomaly[10])
        exp = res.explanations[10]
        spans = exp.localized_spans
        generic_banned = {"uses", "backend", "user", "development", "while", "overview", "implementation"}
        for sp in spans:
            for word in sp["text"].lower().split():
                self.assertNotIn(
                    word, generic_banned,
                    f"Span '{sp['text']}' contains banned generic word '{word}'"
                )

    def test_anomaly_explanations_multi_factor(self):
        """Verifies that explanations contain required multi-factor details without tautology."""
        corrupted_sentences = list(self.sentences_38)
        corrupted_sentences.insert(
            28,
            "The proposed model completely failed to function and enterprise accuracy decreased drastically to 12 percent."
        )
        corrupted_doc = "\n\n".join(corrupted_sentences)
        res = self.model.analyze_document(corrupted_doc)

        exp = res.explanations[28]
        # Must answer: Why flagged? What is evidence? Contributing signals? Confidence?
        self.assertIn("Flagged as", exp.contextual_explanation)
        self.assertIn("dominant signal", exp.contextual_explanation)
        self.assertIn("Certainty:", exp.contextual_explanation)
        self.assertTrue(len(exp.direct_model_evidence) > 0)
        self.assertIn("active_contradiction_backend", exp.direct_model_evidence)


if __name__ == "__main__":
    unittest.main()
