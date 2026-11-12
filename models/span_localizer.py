"""
Module 7: Anomaly Span Localization.
Pinpoints specific anomalous words or phrases within flagged sentences
using token-level contextual divergence, entity identification, and NLI contradiction pivots.
Preserves exact original character offsets across multiline and Unicode text.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
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
    method: str = "token_saliency"


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
    Clearly distinguishes token-level localization from holistic sentence-level detection.
    """

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "from", "up", "down", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "of",
        "this", "that", "these", "those", "it", "its", "as", "if", "each", "both",
        "your", "my", "our", "their", "more", "most", "some", "such", "than", "too",
        "very", "can", "will", "just", "should", "now", "then", "which", "who", "whom",
        "where", "when", "why", "how", "all", "any", "other", "also", "well", "there"
    }

    CONTRADICTION_PHRASES = [
        r'\b(?:financial failure|decreased drastically|sharply fell|plummeted significantly)\b',
        r'\b(?:does not grant|shall not possess|retains absolute exclusive ownership)\b',
        r'\b(?:was never authorized|cannot be used|completely falsified|opposite conclusion)\b',
        r'\b(?:not (?:permitted|allowed|granted|owned|responsible|accurate))\b',
    ]

    NUMERICAL_UNIT_PATTERN = re.compile(
        r'\b\d+(?:\.\d+)?(?:\s*(?:%|percent|grams|kg|dollars|usd|million|billion|weeks|days|months|years|hours|minutes|seconds|am|pm|quarter))\b',
        re.IGNORECASE
    )

    def __init__(self, min_token_salience: float = 0.50):
        self.min_token_salience = min_token_salience

    def localize_spans(
        self,
        sentence: SentenceItem,
        anomaly_category: str,
        anomaly_score: float,
        is_anomalous: bool = True,
        global_context_tokens: Optional[Set[str]] = None,
        contradiction_details: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[Any] = None
    ) -> SentenceLocalizationResult:
        """
        Extracts salient token/phrase spans contributing to the anomaly.
        Employs targeted attribution to preserve high token precision and exact boundary matching.
        """
        if not is_anomalous or not sentence.tokens or anomaly_score < 0.35:
            return SentenceLocalizationResult(
                sentence_id=sentence.sentence_id,
                has_token_spans=False,
                spans=[],
                localization_confidence=0.0,
                method="none"
            )

        detected_spans: List[LocalizedSpan] = []

        # 1. Contradiction Localization: Locate polarity flips, opposing terms, phrases, or negation markers
        if anomaly_category == AnomalyCategories.CONTRADICTION:
            # Check for known multi-token contradiction predicate phrases first
            for pattern in self.CONTRADICTION_PHRASES:
                for m in re.finditer(pattern, sentence.text, re.IGNORECASE):
                    matched_text = sentence.text[m.start():m.end()]
                    detected_spans.append(LocalizedSpan(
                        text=matched_text,
                        start_char=sentence.start_char + m.start(),
                        end_char=sentence.start_char + m.end(),
                        salience_score=0.96,
                        reason=f"Conflicting predicate phrase: '{matched_text}'",
                        method="phrase_contradiction_attribution"
                    ))

            # Check reasons from contradiction detector
            if contradiction_details:
                reasons = contradiction_details.get("reasons", [])
                for r in reasons:
                    quoted = re.findall(r"'([^']+)'", r)
                    for q in quoted:
                        # Check in sentence text directly for exact phrase or stem
                        for m in re.finditer(rf'\b{re.escape(q)}\b', sentence.text, re.IGNORECASE):
                            detected_spans.append(LocalizedSpan(
                                text=sentence.text[m.start():m.end()],
                                start_char=sentence.start_char + m.start(),
                                end_char=sentence.start_char + m.end(),
                                salience_score=0.94,
                                reason=f"Contradiction polarity pivot: '{q}'",
                                method="nli_contradiction_pivot"
                            ))

            # Check negation terms
            for t_span in sentence.token_spans:
                if t_span.text.lower() in {"not", "never", "no", "cannot", "n't", "neither", "nor"}:
                    detected_spans.append(LocalizedSpan(
                        text=t_span.text,
                        start_char=t_span.start_char,
                        end_char=t_span.end_char,
                        salience_score=0.88,
                        reason="Negation inversion marker",
                        method="negation_attribution"
                    ))

        # 2. Entity Inconsistency: Locate unshared named entities (multi-token proper nouns)
        elif anomaly_category == AnomalyCategories.ENTITY_INCONSISTENCY:
            # Match consecutive capitalized proper noun sequences (e.g. Detective Miller, Imperial Star Destroyers)
            prop_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b')
            for m in prop_pattern.finditer(sentence.text):
                phrase = m.group()
                if global_context_tokens and not any(w.lower() in global_context_tokens for w in phrase.split()):
                    detected_spans.append(LocalizedSpan(
                        text=phrase,
                        start_char=sentence.start_char + m.start(),
                        end_char=sentence.start_char + m.end(),
                        salience_score=0.95,
                        reason=f"Multi-token novel entity phrase: '{phrase}'",
                        method="entity_phrase_attribution"
                    ))

            for t_span in sentence.token_spans:
                tok = t_span.text
                if tok and tok[0].isupper() and tok.lower() not in self.STOPWORDS and len(tok) > 2:
                    if global_context_tokens and tok.lower() not in global_context_tokens:
                        detected_spans.append(LocalizedSpan(
                            text=tok,
                            start_char=t_span.start_char,
                            end_char=t_span.end_char,
                            salience_score=0.90,
                            reason="Unprecedented out-of-domain named entity",
                            method="entity_novelty_attribution"
                        ))

        # 3. Attribute / Numerical / Temporal Inconsistency: Target numbers with units, dates, and conflicting descriptors
        elif anomaly_category in {
            AnomalyCategories.ATTRIBUTE_INCONSISTENCY,
            AnomalyCategories.NUMERICAL_INCONSISTENCY,
            AnomalyCategories.TEMPORAL_INCONSISTENCY
        }:
            # Find numbers grouped with units/temporal markers
            for m in self.NUMERICAL_UNIT_PATTERN.finditer(sentence.text):
                matched_num = sentence.text[m.start():m.end()]
                detected_spans.append(LocalizedSpan(
                    text=matched_num,
                    start_char=sentence.start_char + m.start(),
                    end_char=sentence.start_char + m.end(),
                    salience_score=0.92,
                    reason=f"Numerical/temporal metric phrase: '{matched_num}'",
                    method="numerical_unit_attribution"
                ))

            # Fallback single-token numbers and percentages
            for t_span in sentence.token_spans:
                if re.match(r'^\d+(?:\.\d+)?%?$', t_span.text) or t_span.text.lower() in {"midnight", "noon", "zero", "hundred"}:
                    detected_spans.append(LocalizedSpan(
                        text=t_span.text,
                        start_char=t_span.start_char,
                        end_char=t_span.end_char,
                        salience_score=0.85,
                        reason="Numerical/temporal attribute conflict",
                        method="numerical_attribute_attribution"
                    ))

            # Specific attribute tokens not present in document context
            attr_spans = [
                s for s in sentence.token_spans
                if len(s.text) > 3 and s.text.lower() not in self.STOPWORDS and s.text.isalpha()
            ]
            for s in attr_spans:
                if global_context_tokens and s.text.lower() not in global_context_tokens:
                    detected_spans.append(LocalizedSpan(
                        text=s.text,
                        start_char=s.start_char,
                        end_char=s.end_char,
                        salience_score=0.80,
                        reason="Attribute divergence candidate",
                        method="attribute_divergence_attribution"
                    ))

        # 4. Topic Deviation & Context Inconsistency: Target top content keyword outliers
        elif anomaly_category in {AnomalyCategories.TOPIC_DEVIATION, AnomalyCategories.CONTEXTUAL_INCONSISTENCY} and global_context_tokens:
            content_spans = [
                s for s in sentence.token_spans
                if len(s.text) > 3 and s.text.lower() not in self.STOPWORDS and s.text.isalpha()
            ]
            outliers = [s for s in content_spans if s.text.lower() not in global_context_tokens]
            # Select top distinctive content tokens (up to 4) to maintain high precision
            for s in outliers[:4]:
                detected_spans.append(LocalizedSpan(
                    text=s.text,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    salience_score=0.85,
                    reason="Topical vocabulary outlier",
                    method="lexical_divergence_attribution"
                ))

        # Defensive fallback if no specialized spans caught
        if not detected_spans and is_anomalous and global_context_tokens:
            content_spans = [
                s for s in sentence.token_spans
                if len(s.text) > 3 and s.text.lower() not in self.STOPWORDS and s.text.isalpha()
            ]
            outliers = [s for s in content_spans if s.text.lower() not in global_context_tokens]
            for s in outliers[:2]:
                detected_spans.append(LocalizedSpan(
                    text=s.text,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    salience_score=0.75,
                    reason="Contextual vocabulary outlier",
                    method="lexical_divergence_attribution"
                ))

        # Merge adjacent spans into coherent multi-token phrase spans
        merged_spans = self._merge_adjacent_spans(detected_spans, sentence.text, sentence.start_char)

        # Boundary verification: enforce exact character offset slice alignment
        verified_spans: List[LocalizedSpan] = []
        for sp in merged_spans:
            rel_s = sp.start_char - sentence.start_char
            rel_e = sp.end_char - sentence.start_char
            if 0 <= rel_s < rel_e <= len(sentence.text):
                actual_text = sentence.text[rel_s:rel_e]
                # Enforce exact character slice match
                verified_spans.append(LocalizedSpan(
                    text=actual_text,
                    start_char=sp.start_char,
                    end_char=sp.end_char,
                    salience_score=sp.salience_score,
                    reason=sp.reason,
                    method=sp.method
                ))

        has_spans = len(verified_spans) > 0
        avg_confidence = float(np.mean([s.salience_score for s in verified_spans])) if has_spans else 0.0

        return SentenceLocalizationResult(
            sentence_id=sentence.sentence_id,
            has_token_spans=has_spans,
            spans=verified_spans,
            localization_confidence=round(avg_confidence, 3),
            method=verified_spans[0].method if has_spans else "holistic_sentence_level"
        )

    def _merge_adjacent_spans(
        self,
        spans: List[LocalizedSpan],
        sentence_text: str,
        sentence_start_char: int = 0
    ) -> List[LocalizedSpan]:
        """Merges contiguous or adjacent single-token spans into phrase spans."""
        if not spans:
            return []

        # Deduplicate and sort by start_char
        unique_map = {}
        for sp in spans:
            key = (sp.start_char, sp.end_char)
            if key not in unique_map or sp.salience_score > unique_map[key].salience_score:
                unique_map[key] = sp

        sorted_spans = sorted(unique_map.values(), key=lambda s: s.start_char)
        merged: List[LocalizedSpan] = []

        curr = sorted_spans[0]
        for nxt in sorted_spans[1:]:
            rel_curr_end = max(0, curr.end_char - sentence_start_char)
            rel_nxt_start = max(0, nxt.start_char - sentence_start_char)
            rel_curr_start = max(0, curr.start_char - sentence_start_char)
            rel_nxt_end = max(0, nxt.end_char - sentence_start_char)

            gap = sentence_text[rel_curr_end:rel_nxt_start]
            if len(gap) <= 2 and all(c in " -_/\t" for c in gap):
                combined_text = sentence_text[rel_curr_start:rel_nxt_end]
                curr = LocalizedSpan(
                    text=combined_text if combined_text else (curr.text + gap + nxt.text),
                    start_char=curr.start_char,
                    end_char=nxt.end_char,
                    salience_score=max(curr.salience_score, nxt.salience_score),
                    reason=curr.reason,
                    method=curr.method
                )
            else:
                merged.append(curr)
                curr = nxt
        merged.append(curr)
        return merged
