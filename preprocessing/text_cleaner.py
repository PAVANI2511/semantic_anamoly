"""
Text cleaning and normalization utilities.
Preserves character offset mapping whenever necessary.
"""

import re
import unicodedata
from typing import Tuple


class TextCleaner:
    """Provides safe text normalization and whitespace normalization."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Standardizes unicode characters and normalized whitespace without destroying structure."""
        if not text:
            return ""
        # Normalize unicode (NFKC)
        normalized = unicodedata.normalize("NFKC", text)
        # Standardize curly quotes and dashes
        normalized = normalized.replace("“", '"').replace("”", '"')
        normalized = normalized.replace("‘", "'").replace("’", "'")
        normalized = normalized.replace("—", " - ").replace("–", " - ")
        return normalized

    @staticmethod
    def clean_for_embedding(text: str) -> str:
        """Removes excessive internal whitespace and line breaks for transformer embedding."""
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

