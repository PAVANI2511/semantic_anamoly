"""
Module 2: Sentence Segmentation and Tokenization.
Preserves exact character offsets and paragraph associations.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from utils.logging_config import logger
from preprocessing.text_cleaner import TextCleaner


@dataclass
class TokenSpan:
    """Represents a token and its character offset."""
    text: str
    start_char: int
    end_char: int


@dataclass
class SentenceItem:
    """Structured representation of an extracted sentence."""
    sentence_id: int
    text: str
    cleaned_text: str
    paragraph_id: int
    start_char: int
    end_char: int
    tokens: List[str] = field(default_factory=list)
    token_spans: List[TokenSpan] = field(default_factory=list)


class SentenceSegmenter:
    """
    Robust sentence segmenter with character offset tracking and paragraph indexing.
    Supports tokenization and punctuation-safe splitting.
    """

    # Sentence boundary regex handling abbreviations, quotes, etc.
    _SPLIT_REGEX = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[])|(?<=[.!?]["\'\)])\s+(?=[A-Z0-9])'
    )
    _TOKEN_REGEX = re.compile(r'\b\w+\b|[^\w\s]')

    def __init__(self, use_nltk: bool = True):
        self.use_nltk = use_nltk
        self._nltk_available = False
        if use_nltk:
            try:
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                    self._nltk_available = True
                except (LookupError, AttributeError):
                    try:
                        nltk.download('punkt', quiet=True)
                        self._nltk_available = True
                    except Exception:
                        self._nltk_available = False
            except ImportError:
                self._nltk_available = False

    def segment(self, raw_text: str, paragraphs: Optional[List[str]] = None) -> List[SentenceItem]:
        """
        Segments document text into sentences while tracking start/end character offsets
        and mapping each sentence to its corresponding paragraph.
        """
        if not raw_text or not raw_text.strip():
            return []

        sentences: List[SentenceItem] = []
        sentence_id = 0
        current_search_idx = 0

        # If paragraphs are not provided, deduce them from text
        if not paragraphs:
            paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [raw_text.strip()]

        for p_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue

            # Find paragraph start in raw_text to ensure exact offsets
            p_start = raw_text.find(paragraph, current_search_idx)
            if p_start == -1:
                p_start = current_search_idx
            current_search_idx = p_start + len(paragraph)

            raw_sentences = self._split_paragraph_into_sentences(paragraph)
            local_p_offset = 0

            for s_text in raw_sentences:
                s_text_stripped = s_text.strip()
                if not s_text_stripped:
                    continue

                # Locate sentence within the paragraph
                s_local_start = paragraph.find(s_text_stripped, local_p_offset)
                if s_local_start == -1:
                    s_local_start = local_p_offset

                s_start_char = p_start + s_local_start
                s_end_char = s_start_char + len(s_text_stripped)
                local_p_offset = s_local_start + len(s_text_stripped)

                # Tokenize sentence
                tokens, token_spans = self._tokenize(s_text_stripped, s_start_char)

                cleaned_text = TextCleaner.clean_for_embedding(s_text_stripped)

                sentence_item = SentenceItem(
                    sentence_id=sentence_id,
                    text=s_text_stripped,
                    cleaned_text=cleaned_text,
                    paragraph_id=p_idx,
                    start_char=s_start_char,
                    end_char=s_end_char,
                    tokens=tokens,
                    token_spans=token_spans
                )
                sentences.append(sentence_item)
                sentence_id += 1

        return sentences

    def _split_paragraph_into_sentences(self, paragraph: str) -> List[str]:
        """Splits a paragraph string into sentence strings."""
        if self._nltk_available:
            try:
                import nltk
                return nltk.sent_tokenize(paragraph)
            except Exception:
                pass

        # Regex fallback
        parts = self._SPLIT_REGEX.split(paragraph)
        results = []
        for part in parts:
            part = part.strip()
            if part:
                results.append(part)
        return results if results else [paragraph]

    def _tokenize(self, sentence_text: str, sentence_start_offset: int) -> Tuple[List[str], List[TokenSpan]]:
        """Extracts word/punctuation tokens with document-level character offsets."""
        tokens: List[str] = []
        spans: List[TokenSpan] = []

        for match in self._TOKEN_REGEX.finditer(sentence_text):
            tok = match.group()
            start = sentence_start_offset + match.start()
            end = sentence_start_offset + match.end()
            tokens.append(tok)
            spans.append(TokenSpan(text=tok, start_char=start, end_char=end))

        return tokens, spans
