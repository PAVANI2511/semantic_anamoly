"""
Module 1: Document Input and Extraction.
Handles PDF, DOCX, TXT files, and raw pasted text with graceful error recovery.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import io

from utils.logging_config import logger


@dataclass
class ExtractedDocument:
    """Represents the raw extracted document content and metadata."""
    raw_text: str
    source_name: str
    file_type: str
    paragraphs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    error_message: Optional[str] = None


class DocumentExtractor:
    """Robust document extractor supporting PDF, DOCX, TXT, and raw text."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    @classmethod
    def extract_from_text(cls, text: str, source_name: str = "Pasted Text") -> ExtractedDocument:
        """Extracts content from pasted or direct text."""
        if not text or not text.strip():
            return ExtractedDocument(
                raw_text="",
                source_name=source_name,
                file_type="text",
                is_valid=False,
                error_message="Document is empty."
            )

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        return ExtractedDocument(
            raw_text=text.strip(),
            source_name=source_name,
            file_type="text",
            paragraphs=paragraphs,
            metadata={"paragraph_count": len(paragraphs), "character_count": len(text)}
        )

    @classmethod
    def extract_from_file(cls, file_path: str | Path) -> ExtractedDocument:
        """Extracts content from a file path."""
        path = Path(file_path)
        if not path.exists():
            return ExtractedDocument(
                raw_text="",
                source_name=path.name,
                file_type=path.suffix.lower(),
                is_valid=False,
                error_message=f"File not found: {path}"
            )

        suffix = path.suffix.lower()
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            return ExtractedDocument(
                raw_text="",
                source_name=path.name,
                file_type=suffix,
                is_valid=False,
                error_message=f"Unsupported format '{suffix}'. Supported: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )

        try:
            with open(path, "rb") as f:
                content_bytes = f.read()
            return cls.extract_from_bytes(content_bytes, path.name)
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            return ExtractedDocument(
                raw_text="",
                source_name=path.name,
                file_type=suffix,
                is_valid=False,
                error_message=f"Error reading file: {str(e)}"
            )

    @classmethod
    def extract_from_bytes(cls, content: bytes, filename: str) -> ExtractedDocument:
        """Extracts text and paragraphs from file binary bytes."""
        path = Path(filename)
        suffix = path.suffix.lower()

        if not content:
            return ExtractedDocument(
                raw_text="",
                source_name=filename,
                file_type=suffix,
                is_valid=False,
                error_message="File content is empty."
            )

        if suffix == ".txt":
            return cls._extract_txt(content, filename)
        elif suffix == ".pdf":
            return cls._extract_pdf(content, filename)
        elif suffix == ".docx":
            return cls._extract_docx(content, filename)
        else:
            return ExtractedDocument(
                raw_text="",
                source_name=filename,
                file_type=suffix,
                is_valid=False,
                error_message=f"Unsupported file format: {suffix}"
            )

    @classmethod
    def _extract_txt(cls, content: bytes, filename: str) -> ExtractedDocument:
        """Extracts text from TXT with multiple encoding attempts."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "ascii"]
        text = None
        used_enc = None

        for enc in encodings:
            try:
                text = content.decode(enc)
                used_enc = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if text is None:
            return ExtractedDocument(
                raw_text="",
                source_name=filename,
                file_type=".txt",
                is_valid=False,
                error_message="Failed to decode text file. Unsupported character encoding."
            )

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs and text.strip():
            paragraphs = [text.strip()]

        return ExtractedDocument(
            raw_text=text.strip(),
            source_name=filename,
            file_type=".txt",
            paragraphs=paragraphs,
            metadata={"encoding": used_enc, "paragraph_count": len(paragraphs)}
        )

    @classmethod
    def _extract_pdf(cls, content: bytes, filename: str) -> ExtractedDocument:
        """Extracts text from PDF bytes using pypdf or PyMuPDF if available."""
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            pages_text = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text.strip())

            full_text = "\n\n".join(pages_text).strip()
            if not full_text:
                return ExtractedDocument(
                    raw_text="",
                    source_name=filename,
                    file_type=".pdf",
                    is_valid=False,
                    error_message="PDF contains no extractable text (it might be scanned or image-only)."
                )

            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            return ExtractedDocument(
                raw_text=full_text,
                source_name=filename,
                file_type=".pdf",
                paragraphs=paragraphs,
                metadata={"page_count": len(reader.pages), "paragraph_count": len(paragraphs)}
            )
        except Exception as e:
            logger.error(f"PDF extraction error for {filename}: {e}")
            return ExtractedDocument(
                raw_text="",
                source_name=filename,
                file_type=".pdf",
                is_valid=False,
                error_message=f"PDF extraction failed: {str(e)}"
            )

    @classmethod
    def _extract_docx(cls, content: bytes, filename: str) -> ExtractedDocument:
        """Extracts text from DOCX bytes using python-docx."""
        try:
            import docx
            doc = docx.Document(io.BytesIO(content))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs).strip()

            if not full_text:
                return ExtractedDocument(
                    raw_text="",
                    source_name=filename,
                    file_type=".docx",
                    is_valid=False,
                    error_message="DOCX document contains no text."
                )

            return ExtractedDocument(
                raw_text=full_text,
                source_name=filename,
                file_type=".docx",
                paragraphs=paragraphs,
                metadata={"paragraph_count": len(paragraphs)}
            )
        except Exception as e:
            logger.error(f"DOCX extraction error for {filename}: {e}")
            return ExtractedDocument(
                raw_text="",
                source_name=filename,
                file_type=".docx",
                is_valid=False,
                error_message=f"DOCX extraction failed: {str(e)}"
            )

