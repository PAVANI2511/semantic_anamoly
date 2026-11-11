"""
Module 7: Anomaly Span Localization.
Pinpoints specific anomalous words or phrases within flagged sentences
using token-level contextual divergence and polarity attribution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np

from utils.logging_config import logger
from preprocessing.sentence_segmenter import SentenceItem, TokenSpan
from config.settings import AnomalyCategories


@dataclass
class LocalizedSpan:
    """Represents a localized anomalous token or phrase span."""
    text: str
    start_char: int
    end_char: int
    salience_score: float
    reason: str


@dataclass
class SentenceLocalizationResult:
    """Localization outcome for a sentence, distinguishing sentence-level from token-level."""
    sentence_id: int
    has_token_spans: bool
    spans: List[LocalizedSpan] = field(default_factory=list)
    localization_confidence: float = 0.0
    method: str = "token_saliency"


class SpanLocalizer:
    """
    Localizes anomalous spans within sentences flagged by DGSA.
    Discloses when localization is holistic (sentence-level only) vs token-level.
    """

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "from", "up", "down", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "of"
    }

    def __init__(self, min_token_salience: float = 0.55):
        self.min_token_salience = min_token_salience

    def localize_spans(
        self,
        sentence: SentenceItem,
        anomaly_category: str,
        anomaly_score: float,
        global_context_tokens: Optional[set] = None,
        contradiction_details: Optional[Dict[str, Any]] = None
    ) -> SentenceLocalizationResult:
        """
        Extracts salient token spans contributing to the anomaly.
        """
        if not sentence.tokens or anomaly_score < 0.30:
            return SentenceLocalizationResult(
                sentence_id=sentence.sentence_id,
                has_token_spans=False,
                spans=[],
                localization_confidence=0.0,
                method="none"
            )

        detected_spans: List[LocalizedSpan] = []

        # 1. Contradiction Localization: Locate polarity flips or antonym triggers
        if anomaly_category == AnomalyCategories.CONTRADICTION and contradiction_details:
            reasons = contradiction_details.get("reasons", [])
            for r in reasons:
                # Extract quoted terms if any: e.g. "'increase' vs 'decrease'"
                quoted = re.findall(r"'([^']+)'", r)
                for q in quoted:
                    for t_span in sentence.token_spans:
                        if t_span.text.lower() == q.lower():
                            detected_spans.append(LocalizedSpan(
                                text=t_span.text,
                                start_char=t_span.start_char,
                                end_char=t_span.end_char,
                                salience_score=0.90,
                                reason=f"Contradiction polarity pivot ({q})"
                            ))

            # Check negation terms
            for t_span in sentence.token_spans:
                if t_span.text.lower() in {"not", "never", "no", "cannot", "n't"}:
                    detected_spans.append(LocalizedSpan(
                        text=t_span.text,
                        start_char=t_span.start_char,
                        end_char=t_span.end_char,
                        salience_score=0.85,
                        reason="Negation inversion marker"
                    ))

        # 2. Entity / Attribute Inconsistency: Locate unknown named entities
        elif anomaly_category in {AnomalyCategories.ENTITY_INCONSISTENCY, AnomalyCategories.ATTRIBUTE_INCONSISTENCY}:
            for t_span in sentence.token_spans:
                # Capitalized tokens that are not stopwords and not sentence-initial common words
                tok = t_span.text
                if tok[0].isupper() and tok.lower() not in self.STOPWORDS and len(tok) > 2:
                    if global_context_tokens and tok.lower() not in global_context_tokens:
                        detected_spans.append(LocalizedSpan(
                            text=tok,
                            start_char=t_span.start_char,
                            end_char=t_span.end_char,
                            salience_score=0.80,
                            reason="Unprecedented entity or out-of-domain proper noun"
                        ))

        # 3. Topic Deviation: Find tokens with zero or lowest overlap with global domain vocabulary
        elif anomaly_category == AnomalyCategories.TOPIC_DEVIATION and global_context_tokens:
            content_spans = [
                s for s in sentence.token_spans
                if len(s.text) > 3 and s.text.lower() not in self.STOPWORDS and s.text.isalpha()
            ]
            for s in content_spans:
                if s.text.lower() not in global_context_tokens:
                    detected_spans.append(LocalizedSpan(
                        text=s.text,
                        start_char=s.start_char,
                        end_char=s.end_char,
                        salience_score=0.75,
                        reason="Lexical topic outlier"
                    ))

        # Merge overlapping or duplicate spans
        unique_spans: List[LocalizedSpan] = []
        seen_ranges = set()
        for span in detected_spans:
            rng = (span.start_char, span.end_char)
            if rng not in seen_ranges:
                seen_ranges.add(rng)
                unique_spans.append(span)

        # Clear distinction between token-level and sentence-level
        has_spans = len(unique_spans) > 0
        avg_confidence = float(np.mean([s.salience_score for s in unique_spans])) if has_spans else 0.0

        return SentenceLocalizationResult(
            sentence_id=sentence.sentence_id,
            has_token_spans=has_spans,
            spans=unique_spans,
            localization_confidence=avg_confidence,
            method="token_attribute_attribution" if has_spans else "holistic_sentence_level"
        )
