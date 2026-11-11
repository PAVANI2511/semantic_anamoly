"""
Traditional and baseline anomaly detection algorithms.
Implements Isolation Forest, One-Class SVM, and Distance-based baselines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from utils.logging_config import logger
from utils.helpers import min_max_normalize


class BaseBaselineDetector(ABC):
    """Abstract interface for baseline anomaly detectors."""

    @abstractmethod
    def fit_predict_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Takes sentence embeddings [N, D] and returns anomaly scores [N] normalized to [0, 1].
        Higher score means more anomalous.
        """
        pass


class IsolationForestDetector(BaseBaselineDetector):
    """Isolation Forest anomaly detection on sentence embeddings."""

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )

    def fit_predict_scores(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) < 3:
            return np.zeros(len(embeddings), dtype=np.float32)

        # IsolationForest decision_function outputs negative scores for anomalies (lower = more abnormal)
        # Score inversion: lower decision function -> higher anomaly score
        self.model.fit(embeddings)
        decision_scores = self.model.decision_function(embeddings)
        raw_anomaly = -decision_scores
        return min_max_normalize(raw_anomaly)


class OneClassSVMDetector(BaseBaselineDetector):
    """One-Class SVM anomaly detection on sentence embeddings."""

    def __init__(self, nu: float = 0.1, kernel: str = "rbf", gamma: str = "scale"):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit_predict_scores(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) < 3:
            return np.zeros(len(embeddings), dtype=np.float32)

        self.model.fit(embeddings)
        # Lower decision function implies higher anomaly
        decision_scores = self.model.decision_function(embeddings)
        raw_anomaly = -decision_scores
        return min_max_normalize(raw_anomaly)


class DistanceCentroidDetector(BaseBaselineDetector):
    """Distance-to-centroid anomaly detection."""

    def fit_predict_scores(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            return np.array([])
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        # Cosine distance = 1 - cosine similarity
        norm_emb = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        norm_cent = centroid / np.maximum(np.linalg.norm(centroid), 1e-12)
        cos_sim = np.dot(norm_emb, norm_cent.T).squeeze(axis=1)
        raw_dist = 1.0 - cos_sim
        return min_max_normalize(raw_dist)

