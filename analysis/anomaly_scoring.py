"""
Module 6: Dual-Level Anomaly Scoring and Category Classifier.
Implements the core DGSA formulation:
    A(i) = alpha * G(i) + beta * L(i) + gamma * C(i)
Includes ablation support to disable individual signals for experimental analysis.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from utils.logging_config import logger
from utils.helpers import min_max_normalize
from config.settings import AnomalyCategories


class DGSAScorer:
    """
    Computes composite anomaly scores A(i) from Global, Local, and Contradiction signals,
    and infers the specific semantic anomaly category.
    """

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.35,
        gamma: float = 0.25,
        threshold: float = 0.50
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold

    def compute_composite_scores(
        self,
        global_scores: np.ndarray,
        local_scores: np.ndarray,
        contradiction_scores: np.ndarray,
        enable_global: bool = True,
        enable_local: bool = True,
        enable_contradiction: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Combines signals with dynamic weight re-normalization (ablation support).
        
        Returns:
            - composite_scores: A(i) [N] in [0, 1]
            - effective_alpha, effective_beta, effective_gamma
        """
        num_sentences = len(global_scores)
        if num_sentences == 0:
            return np.array([]), 0.0, 0.0, 0.0

        # Adjust weights based on active ablation switches
        w_g = self.alpha if enable_global else 0.0
        w_l = self.beta if enable_local else 0.0
        w_c = self.gamma if enable_contradiction else 0.0

        total_weight = w_g + w_l + w_c
        if total_weight > 0:
            norm_alpha = w_g / total_weight
            norm_beta = w_l / total_weight
            norm_gamma = w_c / total_weight
        else:
            norm_alpha = norm_beta = norm_gamma = 0.0

        # Compute composite anomaly score A(i)
        composite = (
            norm_alpha * global_scores +
            norm_beta * local_scores +
            norm_gamma * contradiction_scores
        )

        return composite, norm_alpha, norm_beta, norm_gamma

    def classify_category(
        self,
        global_score: float,
        local_score: float,
        contradiction_score: float,
        local_detail: Dict[str, Any],
        contradiction_detail: Dict[str, Any]
    ) -> str:
        """
        Infers the anomaly category from component signals and indicators.
        """
        # 1. Contradiction has highest semantic priority if detected
        if contradiction_score > 0.45 or (contradiction_detail and contradiction_detail.get("reasons")):
            return AnomalyCategories.CONTRADICTION

        # 2. Bridge or Novel Entity indicators
        if local_detail.get("is_bridge_anomaly") or local_detail.get("entity_novelty", 0) > 0.65:
            # If entities are involved
            if local_detail.get("entities"):
                return AnomalyCategories.ENTITY_INCONSISTENCY
            return AnomalyCategories.CONTEXTUAL_INCONSISTENCY

        # 3. Disproportionate Global deviation vs Local
        if global_score > 0.60 and (global_score - local_score > 0.20):
            return AnomalyCategories.TOPIC_DEVIATION

        # 4. Local inconsistency dominant
        if local_score > global_score:
            return AnomalyCategories.CONTEXTUAL_INCONSISTENCY

        # Default fallback
        return AnomalyCategories.TOPIC_DEVIATION

