"""
Baseline anomaly detection models for scientific benchmarking.
Implements:
1. TF-IDF + Isolation Forest
2. Sentence-BERT + Isolation Forest
3. Sentence-BERT + One-Class SVM
4. Supervised Linear Classifier
5. Proposed DGSA Framework
"""

from typing import Dict, Any, List, Optional
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from utils.logging_config import logger
from utils.helpers import min_max_normalize
from models.dgsa_model import DGSAModel


class BaselineRunner:
    """Orchestrates execution of all comparison baselines."""

    @staticmethod
    def run_tfidf_isolation_forest(texts: List[str], contamination: float = 0.1) -> Dict[str, Any]:
        """Baseline 1: TF-IDF + Isolation Forest."""
        start = time.perf_counter()
        if len(texts) < 3:
            return {"scores": np.zeros(len(texts)), "preds": np.zeros(len(texts), dtype=int), "time_ms": 0.0}

        vectorizer = TfidfVectorizer(max_features=512, stop_words="english")
        tfidf_mat = vectorizer.fit_transform(texts).toarray()
        
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(tfidf_mat)
        dec_scores = iso.decision_function(tfidf_mat)
        norm_scores = min_max_normalize(-dec_scores)
        preds = (norm_scores >= 0.50).astype(int)
        
        elapsed = (time.perf_counter() - start) * 1000
        return {"scores": norm_scores, "preds": preds, "time_ms": elapsed}

    @staticmethod
    def run_sbert_isolation_forest(embeddings: np.ndarray, contamination: float = 0.1) -> Dict[str, Any]:
        """Baseline 2: Sentence-BERT + Isolation Forest."""
        start = time.perf_counter()
        if len(embeddings) < 3:
            return {"scores": np.zeros(len(embeddings)), "preds": np.zeros(len(embeddings), dtype=int), "time_ms": 0.0}

        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(embeddings)
        dec_scores = iso.decision_function(embeddings)
        norm_scores = min_max_normalize(-dec_scores)
        preds = (norm_scores >= 0.50).astype(int)
        
        elapsed = (time.perf_counter() - start) * 1000
        return {"scores": norm_scores, "preds": preds, "time_ms": elapsed}

    @staticmethod
    def run_sbert_oneclass_svm(embeddings: np.ndarray, nu: float = 0.1) -> Dict[str, Any]:
        """Baseline 3: Sentence-BERT + One-Class SVM."""
        start = time.perf_counter()
        if len(embeddings) < 3:
            return {"scores": np.zeros(len(embeddings)), "preds": np.zeros(len(embeddings), dtype=int), "time_ms": 0.0}

        svm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        svm.fit(embeddings)
        dec_scores = svm.decision_function(embeddings)
        norm_scores = min_max_normalize(-dec_scores)
        preds = (norm_scores >= 0.50).astype(int)
        
        elapsed = (time.perf_counter() - start) * 1000
        return {"scores": norm_scores, "preds": preds, "time_ms": elapsed}

    @staticmethod
    def run_supervised_classifier(
        train_emb: np.ndarray,
        train_labels: np.ndarray,
        test_emb: np.ndarray
    ) -> Dict[str, Any]:
        """Baseline 4: Supervised Logistic Classifier on embeddings."""
        start = time.perf_counter()
        clf = LogisticRegression(class_weight="balanced", random_state=42)
        clf.fit(train_emb, train_labels)
        
        probs = clf.predict_proba(test_emb)[:, 1]
        preds = (probs >= 0.50).astype(int)
        
        elapsed = (time.perf_counter() - start) * 1000
        return {"scores": probs, "preds": preds, "time_ms": elapsed}
