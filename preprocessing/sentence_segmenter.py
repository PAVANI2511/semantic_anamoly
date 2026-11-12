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
    is_heading: bool = False
    is_structural: bool = False
    section_title: str = ""
    section_id: int = 0
    heading_level: int = 0


class SentenceSegmenter:
    """
    Robust sentence segmenter with character offset tracking and paragraph indexing.
    Supports tokenization, punctuation-safe splitting, and document structure awareness
    (headings, sections, numbered lists).
    """

    # Sentence boundary regex handling abbreviations, quotes, etc.
    _SPLIT_REGEX = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[])|(?<=[.!?]["\'\)])\s+(?=[A-Z0-9])'
    )
    _TOKEN_REGEX = re.compile(r'\b\w+\b|[^\w\s]')

    SECTION_NAMES = {
        "introduction", "project overview", "overview", "methodology",
        "experimental results", "results", "technical implementation",
        "implementation", "conclusion", "conclusions", "abstract",
        "background", "discussion", "related work", "future work",
        "evaluation", "system architecture"
    }

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

    @classmethod
    def is_heading_or_structural(cls, text: str, prev_was_marker: bool = False) -> Tuple[bool, bool, str, int]:
        """
        Detects whether a sentence or fragment represents document structure (heading, section title, list marker).
        Returns:
            (is_structural: bool, is_heading: bool, section_title: str, heading_level: int)
        """
        s = text.strip()
        if not s:
            return True, False, "", 0

        # 1. Explicit Markdown headings (e.g. "# Title", "## 1.", "### Section")
        m_h = re.match(r'^(#{1,6})\s*(.*)$', s)
        if m_h:
            hashes, rest = m_h.groups()
            level = len(hashes)
            m_num = re.match(r'^\d+\.?\s*(.*)$', rest)
            title = m_num.group(1).strip() if m_num else rest.strip()
            return True, True, title, level

        # 2. Standalone list markers or numbers (e.g. "1.", "2.", "(a)", "-", "*")
        if re.match(r'^(?:\d{1,3}\.|\([0-9a-zA-Z]+\)|[a-zA-Z]\.|\*|-|\+)\s*$', s):
            return True, False, "", 0

        # 3. Known section names or title-cased headings following a heading marker
        low = s.lower().rstrip(':')
        if low in cls.SECTION_NAMES:
            return True, True, s.rstrip(':'), 2

        # 4. Short title-cased phrase without terminal punctuation (e.g. "Project Overview", "Technical Implementation")
        words = s.split()
        if (prev_was_marker or len(words) <= 4) and not s.endswith(('.', '!', '?', ';')) and (s.istitle() or s.isupper()):
            return True, True, s, 2

        return False, False, "", 0

    def segment(self, raw_text: str, paragraphs: Optional[List[str]] = None) -> List[SentenceItem]:
        """
        Segments document text into sentences while tracking start/end character offsets,
        mapping each sentence to its corresponding paragraph and section context.
        """
        if not raw_text or not raw_text.strip():
            return []

        sentences: List[SentenceItem] = []
        sentence_id = 0
        current_search_idx = 0

        current_section_title = "Document"
        current_section_id = 0
        prev_was_marker = False

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

                # Structure & Section Analysis
                is_struct, is_head, s_title, h_lvl = self.is_heading_or_structural(
                    s_text_stripped, prev_was_marker=prev_was_marker
                )

                if is_head and s_title:
                    current_section_title = s_title
                    current_section_id += 1
                elif is_head or is_struct:
                    current_section_id += 1

                prev_was_marker = is_struct and (not s_title)

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
                    token_spans=token_spans,
                    is_heading=is_head,
                    is_structural=is_struct,
                    section_title=current_section_title,
                    section_id=current_section_id,
                    heading_level=h_lvl
                )
                sentences.append(sentence_item)
                sentence_id += 1

        return sentences

    def _split_paragraph_into_sentences(self, paragraph: str) -> List[str]:
        """Splits a paragraph string into sentence strings while preserving headings and list markers."""
        raw_parts: List[str] = []
        if self._nltk_available:
            try:
                import nltk
                raw_parts = nltk.sent_tokenize(paragraph)
            except Exception:
                raw_parts = []

        if not raw_parts:
            parts = self._SPLIT_REGEX.split(paragraph)
            raw_parts = [p.strip() for p in parts if p.strip()]

        # Re-attach detached heading numbers or list markers:
        # e.g., ["## 1.", "Introduction"] -> ["## 1. Introduction"]
        # e.g., ["1.", "Global Context Analysis..."] -> ["1. Global Context Analysis..."]
        merged: List[str] = []
        idx = 0
        while idx < len(raw_parts):
            part = raw_parts[idx].strip()
            if not part:
                idx += 1
                continue
            is_detached_marker = bool(
                re.match(r'^(?:#{1,6}\s*\d+\.?|\d{1,3}\.|\([0-9a-zA-Z]+\)|[a-zA-Z]\.|\*|-|\+)\s*$', part)
            )
            if is_detached_marker and idx + 1 < len(raw_parts):
                next_part = raw_parts[idx + 1].strip()
                merged.append(f"{part} {next_part}")
                idx += 2
            else:
                merged.append(part)
                idx += 1

        return merged if merged else [paragraph]

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

