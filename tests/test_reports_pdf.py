"""
Unit tests for Report Generation and PDF/HTML Exporting.
Verifies:
- ReportLab authentic PDF generation with valid PDF header (%PDF-)
- Secure HTML escaping against XSS injection
- JSON and CSV output integrity
"""

import unittest
from pathlib import Path
import os

from preprocessing.sentence_segmenter import SentenceSegmenter
from models.embedding_model import TFIDFEmbeddingModel
from models.dgsa_model import DGSAModel
from reports.export_results import ExportManager


class TestReportExports(unittest.TestCase):
    """Verifies that PDF, HTML, JSON, and CSV exports are authentic and secure."""

    @classmethod
    def setUpClass(cls):
        cls.segmenter = SentenceSegmenter(use_nltk=False)
        cls.embedding_model = TFIDFEmbeddingModel(max_features=64)
        cls.model = DGSAModel(embedding_model=cls.embedding_model)

        test_doc = (
            "Climate science monitors greenhouse emissions worldwide.\n\n"
            "Rising temperatures impact polar ice sheets significantly.\n\n"
            "<script>alert('xss')</script> Bake chocolate cake in a convection oven."
        )
        cls.result = cls.model.analyze_document(test_doc)
        cls.out_dir = Path("outputs/test_exports")
        cls.out_dir.mkdir(parents=True, exist_ok=True)

    def test_pdf_export_generation(self):
        pdf_path = self.out_dir / "test_audit_report.pdf"
        out_file = ExportManager.export_to_pdf(self.result, pdf_path)
        self.assertTrue(Path(out_file).exists())
        self.assertGreater(os.path.getsize(out_file), 1000)

        # Verify real PDF magic bytes
        with open(out_file, "rb") as f:
            header = f.read(5)
            self.assertEqual(header, b"%PDF-")

    def test_html_xss_escaping(self):
        html_path = self.out_dir / "test_audit_report.html"
        out_file = ExportManager.export_to_html_report(self.result, html_path)
        self.assertTrue(Path(out_file).exists())

        with open(out_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Raw unescaped script tag must NOT exist in HTML
            self.assertNotIn("<script>alert('xss')</script>", content)
            # Escaped tag must be present
            self.assertIn("&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;", content)

    def test_json_and_csv_export(self):
        json_path = self.out_dir / "test_audit_report.json"
        csv_path = self.out_dir / "test_audit_report.csv"

        out_json = ExportManager.export_to_json(self.result, json_path)
        out_csv = ExportManager.export_to_csv(self.result, csv_path)

        self.assertTrue(Path(out_json).exists())
        self.assertTrue(Path(out_csv).exists())
        self.assertGreater(os.path.getsize(out_json), 100)
        self.assertGreater(os.path.getsize(out_csv), 100)


if __name__ == "__main__":
    unittest.main()
