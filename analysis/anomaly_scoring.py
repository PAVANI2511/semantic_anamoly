"""
Module 6: Dual-Level Anomaly Scoring and Category Classifier.
Implements the core DGSA formulation:
    A(i) = alpha * G(i) + beta * L(i) + gamma * C(i)
Includes ablation support to disable individual signals for experimental analysis,
evidence-grounded category inference, and confidence estimation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from utils.logging_config import logger
from config.settings import AnomalyCategories


@dataclass
class CategoryInferenceResult:
    """Represents the outcome of anomaly category inference."""
    primary_category: str
    secondary_categories: List[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence_signals: List[str] = field(default_factory=list)
    is_uncertain: bool = False


class DGSAScorer:
    """
    Computes composite anomaly scores A(i) from Global, Local, and Contradiction signals,
    and infers evidence-based semantic anomaly categories.
    """

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.35,
        gamma: float = 0.25,
        threshold: float = 0.50
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.threshold = float(threshold)
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Validates that weights are non-negative and normalizes them."""
        if self.alpha < 0.0 or self.beta < 0.0 or self.gamma < 0.0:
            raise ValueError(f"Weights must be non-negative: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")
        total = self.alpha + self.beta + self.gamma
        if total <= 0.0:
            raise ValueError("Sum of weights must be strictly positive.")
        self.alpha /= total
        self.beta /= total
        self.gamma /= total

    def compute_composite_scores(
        self,
        global_scores: np.ndarray,
        local_scores: np.ndarray,
        contradiction_scores: np.ndarray,
        enable_global: bool = True,
        enable_local: bool = True,
        enable_contradiction: bool = True
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Combines signals with dynamic weight re-normalization (ablation support).
        
        Returns:
            - composite_scores: A(i) [N] in [0, 1]
            - effective_alpha, effective_beta, effective_gamma
        """
        num_sentences = len(global_scores)
        if num_sentences == 0:
            return np.array([], dtype=np.float32), 0.0, 0.0, 0.0

        # Handle missing or mismatched arrays safely
        g_arr = np.asarray(global_scores, dtype=np.float32) if len(global_scores) == num_sentences else np.zeros(num_sentences, dtype=np.float32)
        l_arr = np.asarray(local_scores, dtype=np.float32) if len(local_scores) == num_sentences else np.zeros(num_sentences, dtype=np.float32)
        c_arr = np.asarray(contradiction_scores, dtype=np.float32) if len(contradiction_scores) == num_sentences else np.zeros(num_sentences, dtype=np.float32)

        # Dynamic ablation weights
        w_g = self.alpha if enable_global else 0.0
        w_l = self.beta if enable_local else 0.0
        w_c = self.gamma if enable_contradiction else 0.0

        total_weight = w_g + w_l + w_c
        if total_weight > 0:
            norm_alpha = float(w_g / total_weight)
            norm_beta = float(w_l / total_weight)
            norm_gamma = float(w_c / total_weight)
        else:
            norm_alpha = norm_beta = norm_gamma = 0.0

        # Compute composite anomaly score A(i)
        composite = (
            norm_alpha * g_arr +
            norm_beta * l_arr +
            norm_gamma * c_arr
        )

        composite = np.clip(composite, 0.0, 1.0).astype(np.float32)
        return composite, norm_alpha, norm_beta, norm_gamma

    def classify_category(
        self,
        is_anomalous: bool,
        global_score: float,
        local_score: float,
        contradiction_score: float,
        local_detail: Dict[str, Any],
        contradiction_detail: Dict[str, Any]
    ) -> CategoryInferenceResult:
        """
        Infers evidence-backed anomaly category.
        A normal sentence receives 'normal' category.
        """
        if not is_anomalous:
            return CategoryInferenceResult(
                primary_category=AnomalyCategories.NORMAL,
                secondary_categories=[],
                confidence=1.0 - max(global_score, local_score, contradiction_score),
                evidence_signals=["Sentence conforms to expected contextual distribution."],
                is_uncertain=False
            )

        evidence: List[str] = []
        secondary: List[str] = []
        candidate_scores: Dict[str, float] = {}

        # 1. Contradiction Signal
        if contradiction_score >= 0.40 or (contradiction_detail and contradiction_detail.get("reasons")):
            backend = contradiction_detail.get("backend", "heuristic")
            evidence.append(f"Contradiction detected via {backend} (Score: {contradiction_score:.2f})")
            candidate_scores[AnomalyCategories.CONTRADICTION] = contradiction_score

        # 2. Entity Inconsistency Signal
        unshared = local_detail.get("unshared_entities", [])
        if unshared or local_detail.get("entity_novelty", 0) > 0.50:
            evidence.append(f"Unprecedented named entity detected: {', '.join(unshared[:3]) if unshared else 'novel proper noun'}")
            candidate_scores[AnomalyCategories.ENTITY_INCONSISTENCY] = local_detail.get("entity_novelty", 0.6)

        # 3. Topic Deviation Signal
        if global_score >= 0.50 and (global_score - local_score > 0.10 or global_score > 0.70):
            evidence.append(f"Global topic divergence exceeds centroid threshold (G={global_score:.2f})")
            candidate_scores[AnomalyCategories.TOPIC_DEVIATION] = global_score

        # 4. Contextual Inconsistency (Bridge Anomaly)
        if local_detail.get("is_bridge_anomaly"):
            evidence.append("Bridge anomaly: Sentence alienates otherwise continuous preceding and following narrative")
            candidate_scores[AnomalyCategories.CONTEXTUAL_INCONSISTENCY] = local_score + 0.20
        elif local_score >= 0.50:
            evidence.append(f"High local neighbor inconsistency (L={local_score:.2f})")
            candidate_scores[AnomalyCategories.CONTEXTUAL_INCONSISTENCY] = local_score

        # 5. Attribute Inconsistency Signal
        if local_detail.get("numbers") or "attribute" in " ".join(evidence).lower():
            # Check for conflicting numbers or attributes
            candidate_scores[AnomalyCategories.ATTRIBUTE_INCONSISTENCY] = 0.55

        # Decision
        if not candidate_scores:
            # Fallback when score exceeded threshold via combined modest signals
            if global_score >= local_score and global_score >= contradiction_score:
                primary = AnomalyCategories.TOPIC_DEVIATION
            elif local_score >= contradiction_score:
                primary = AnomalyCategories.CONTEXTUAL_INCONSISTENCY
            else:
                primary = AnomalyCategories.CONTRADICTION
            confidence = 0.45
            is_uncertain = True
            evidence.append("Marginal composite signal above threshold; category assignment has lower certainty.")
        else:
            sorted_cats = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            primary = sorted_cats[0][0]
            confidence = float(min(1.0, sorted_cats[0][1]))
            secondary = [c for c, _ in sorted_cats[1:] if c != primary]
            is_uncertain = confidence < 0.50

        return CategoryInferenceResult(
            primary_category=primary,
            secondary_categories=secondary,
            confidence=round(confidence, 3),
            evidence_signals=evidence,
            is_uncertain=is_uncertain
        )
