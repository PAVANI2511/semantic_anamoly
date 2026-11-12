"""
Module 8: Explainable AI Engine.
Generates evidence-grounded, transparent explanations for flagged anomalies,
distinguishing continuous neural model evidence from rule-based/heuristic indicators,
detailing contributing mathematical signals, context sentences, and localized spans.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from utils.logging_config import logger
from preprocessing.sentence_segmenter import SentenceItem
from models.span_localizer import LocalizedSpan
from config.settings import AnomalyCategories


@dataclass
class AnomalyExplanation:
    """Detailed explanation metadata for an anomalous sentence."""
    sentence_id: int
    sentence_text: str
    category: str
    anomaly_score: float
    is_anomalous: bool
    primary_factor: str
    signals_breakdown: Dict[str, float]
    surrounding_context: List[Dict[str, Any]]
    contextual_explanation: str
    localized_spans: List[Dict[str, Any]] = field(default_factory=list)
    has_token_localization: bool = False
    evidence_certainty: str = "moderate"
    contradiction_backend: str = "unknown"
    direct_model_evidence: Dict[str, Any] = field(default_factory=dict)
    heuristic_indicators: List[str] = field(default_factory=list)


class ExplanationEngine:
    """
    Constructs explainable reports directly from DGSA mathematical decomposition:
    G(i) [Global], L(i) [Local], and C(i) [Contradiction].
    Explicitly separates direct neural model evidence from heuristic indicators.
    """

    @classmethod
    def generate_explanation(
        cls,
        sentence: SentenceItem,
        all_sentences: List[SentenceItem],
        category: str,
        anomaly_score: float,
        is_anomalous: bool,
        global_score: float,
        local_score: float,
        contradiction_score: float,
        alpha: float,
        beta: float,
        gamma: float,
        local_details: Dict[str, Any],
        contradiction_details: Dict[str, Any],
        localized_spans: List[LocalizedSpan],
        contradiction_backend: str = "unknown"
    ) -> AnomalyExplanation:
        """Constructs an evidence-backed explanation for a sentence."""
        # Calculate component contributions
        contrib_g = alpha * global_score
        contrib_l = beta * local_score
        contrib_c = gamma * contradiction_score

        signals = {
            "global_deviation_G": round(float(global_score), 4),
            "local_inconsistency_L": round(float(local_score), 4),
            "contradiction_score_C": round(float(contradiction_score), 4),
            "effective_global_contrib": round(float(contrib_g), 4),
            "effective_local_contrib": round(float(contrib_l), 4),
            "effective_contradiction_contrib": round(float(contrib_c), 4),
            "composite_anomaly_score_A": round(float(anomaly_score), 4)
        }

        # Gather surrounding sentences
        s_id = sentence.sentence_id
        context_items = []
        for offset in [-2, -1, 1, 2]:
            target_idx = s_id + offset
            if 0 <= target_idx < len(all_sentences):
                context_items.append({
                    "relative_pos": f"{offset:+d}",
                    "sentence_id": target_idx,
                    "text": all_sentences[target_idx].text
                })

        # Determine primary contributing factor
        factors = [
            ("Global topic divergence", contrib_g),
            ("Local contextual incoherence", contrib_l),
            ("Semantic contradiction", contrib_c)
        ]
        factors.sort(key=lambda x: x[1], reverse=True)
        primary_factor = factors[0][0]

        # Separate continuous neural model evidence from discrete heuristic indicators
        direct_model_evidence = {
            "global_centroid_divergence_G": round(float(global_score), 4),
            "local_neighborhood_inconsistency_L": round(float(local_score), 4),
            "contradiction_signal_C": round(float(contradiction_score), 4),
            "active_contradiction_backend": contradiction_backend,
            "neural_confidence": round(float(anomaly_score), 4)
        }

        heuristic_indicators: List[str] = []
        if local_details.get("is_bridge_anomaly"):
            heuristic_indicators.append("Bridge discontinuity detected across flanking sentences")
        if local_details.get("entities"):
            novel_ents = local_details["entities"][:3]
            heuristic_indicators.append(f"Unprecedented named entity: {', '.join(novel_ents)}")

        # Formulate contextual narrative
        reasons_text = []
        if category == AnomalyCategories.CONTRADICTION:
            reasons = contradiction_details.get("reasons", [])
            conflicting_idx = contradiction_details.get("conflicting_sentence_idx")
            conflict_ref = f"sentence #{conflicting_idx + 1}" if conflicting_idx is not None else "preceding context"
            if reasons:
                reasons_text.append(f"Model detected semantic contradiction with {conflict_ref}: {'; '.join(reasons)}.")
                for r in reasons:
                    heuristic_indicators.append(f"Contradiction clue: {r}")
            else:
                reasons_text.append(
                    f"Direct semantic contradiction detected against {conflict_ref} "
                    f"using {contradiction_backend}."
                )
        elif category == AnomalyCategories.TOPIC_DEVIATION:
            reasons_text.append(
                f"Global semantic deviation is exceptionally high (G={global_score:.2f}). "
                f"The semantic embedding diverges from the document centroid."
            )
            heuristic_indicators.append("Global embedding cosine distance above threshold")
        elif category in {AnomalyCategories.ENTITY_INCONSISTENCY, AnomalyCategories.ATTRIBUTE_INCONSISTENCY}:
            if local_details.get("is_bridge_anomaly"):
                reasons_text.append(
                    "Bridge anomaly detected: Flanking sentences share semantic continuity, "
                    "but this sentence abruptly breaks narrative progression."
                )
            if local_details.get("entities"):
                reasons_text.append(f"Contains novel or unlinked entities: {', '.join(local_details['entities'][:3])}.")
        else:
            reasons_text.append(
                f"Local contextual inconsistency is prominent (L={local_score:.2f}), "
                f"indicating low semantic similarity to immediate neighboring sentences."
            )

        if not is_anomalous:
            explanation_summary = f"Sentence conforms to document semantics (Score: {anomaly_score:.2f})."
        else:
            explanation_summary = " ".join(reasons_text)

        # Certainty assessment based on signal magnitude
        certainty = "high" if anomaly_score > 0.70 else ("moderate" if anomaly_score > 0.45 else "low")

        spans_json = [
            {
                "text": s.text,
                "start_char": s.start_char,
                "end_char": s.end_char,
                "salience": s.salience_score,
                "reason": s.reason
            }
            for s in localized_spans
        ]

        return AnomalyExplanation(
            sentence_id=s_id,
            sentence_text=sentence.text,
            category=category,
            anomaly_score=anomaly_score,
            is_anomalous=is_anomalous,
            primary_factor=primary_factor,
            signals_breakdown=signals,
            surrounding_context=context_items,
            contextual_explanation=explanation_summary,
            localized_spans=spans_json,
            has_token_localization=len(spans_json) > 0,
            evidence_certainty=certainty,
            contradiction_backend=contradiction_backend,
            direct_model_evidence=direct_model_evidence,
            heuristic_indicators=heuristic_indicators
        )
