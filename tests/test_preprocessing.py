"""
Unit tests for Document Extraction and Preprocessing (Modules 1 & 2).
"""

import unittest
from pathlib import Path

from preprocessing.document_extractor import DocumentExtractor, ExtractedDocument
from preprocessing.sentence_segmenter import SentenceSegmenter, SentenceItem
from preprocessing.text_cleaner import TextCleaner


class TestDocumentExtractor(unittest.TestCase):
    """Verifies file and text extraction robustness."""

    def test_extract_from_valid_text(self):
        sample = "First paragraph here.\n\nSecond paragraph here."
        doc = DocumentExtractor.extract_from_text(sample, "test_doc")
        self.assertTrue(doc.is_valid)
        self.assertEqual(len(doc.paragraphs), 2)
        self.assertEqual(doc.paragraphs[0], "First paragraph here.")
        self.assertEqual(doc.paragraphs[1], "Second paragraph here.")

    def test_extract_from_empty_text(self):
        doc = DocumentExtractor.extract_from_text("   \n\n  ")
        self.assertFalse(doc.is_valid)
        self.assertIn("empty", doc.error_message.lower())

    def test_extract_unsupported_file_type(self):
        doc = DocumentExtractor.extract_from_file("unsupported.xyz")
        self.assertFalse(doc.is_valid)
        self.assertIn("unsupported", doc.error_message.lower())


class TestSentenceSegmenter(unittest.TestCase):
    """Verifies segmentation, tokenization, and exact character offset mapping."""

    def setUp(self):
        self.segmenter = SentenceSegmenter(use_nltk=False)

    def test_exact_character_offsets(self):
        raw_text = "Machine learning is powerful. It detects anomalies in complex documents."
        sentences = self.segmenter.segment(raw_text)

        self.assertEqual(len(sentences), 2)
        
        # Verify first sentence
        s1 = sentences[0]
        self.assertEqual(s1.sentence_id, 0)
        self.assertEqual(raw_text[s1.start_char:s1.end_char], s1.text)

        # Verify second sentence
        s2 = sentences[1]
        self.assertEqual(s2.sentence_id, 1)
        self.assertEqual(raw_text[s2.start_char:s2.end_char], s2.text)

        # Ensure no negative offsets or inverted boundaries
        self.assertLess(s1.start_char, s1.end_char)
        self.assertLess(s2.start_char, s2.end_char)
        self.assertGreaterEqual(s2.start_char, s1.end_char)

    def test_token_spans_alignment(self):
        raw_text = "Artificial intelligence processes semantic text."
        sentences = self.segmenter.segment(raw_text)
        self.assertEqual(len(sentences), 1)

        sent = sentences[0]
        self.assertTrue(len(sent.token_spans) > 0)
        for token_span in sent.token_spans:
            # Check character slice in raw_text matches token text
            slice_text = raw_text[token_span.start_char:token_span.end_char]
            self.assertEqual(slice_text, token_span.text)

    def test_paragraph_indexing(self):
        p1 = "Sentence one. Sentence two."
        p2 = "Sentence three in second paragraph."
        full = f"{p1}\n\n{p2}"

        sentences = self.segmenter.segment(full, paragraphs=[p1, p2])
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0].paragraph_id, 0)
        self.assertEqual(sentences[1].paragraph_id, 0)
        self.assertEqual(sentences[2].paragraph_id, 1)


if __name__ == "__main__":
    unittest.main()

