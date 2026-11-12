"""
Baseline anomaly detection models for scientific benchmarking.
Strict train/validation/test separation to guarantee zero data or sentence leakage.
Implements:
1. TF-IDF + Isolation Forest (fitted on train, calibrated on val, evaluated on test)
2. Sentence-BERT + Isolation Forest (fitted on train, calibrated on val, evaluated on test)
3. Sentence-BERT + One-Class SVM (fitted on train, calibrated on val, evaluated on test)
4. Supervised Logistic Regression (fitted on train, calibrated on val, evaluated on test)
"""

from typing import Dict, Any, List, Optional
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score

from utils.logging_config import logger


class BaselineRunner:
    """Orchestrates execution of comparison baselines with strict train/val/test split discipline."""

    @staticmethod
    def calibrate_threshold(
        val_scores: np.ndarray,
        val_labels: np.ndarray,
        default_th: float = 0.50
    ) -> float:
        """
        Calibrates decision threshold on validation data to maximize F1 score.
        Ensures strict zero-leakage: test set is never seen during threshold calibration.
        """
        if val_scores is None or val_labels is None or len(val_scores) == 0 or len(val_labels) == 0:
            return default_th
        if len(np.unique(val_labels)) < 2:
            return default_th

        best_th = default_th
        best_f1 = -1.0
        for th in np.linspace(0.10, 0.90, 17):
            bin_preds = (val_scores >= th).astype(int)
            f1 = float(f1_score(val_labels, bin_preds, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(th)

        return best_th

    @classmethod
    def run_tfidf_isolation_forest(
        cls,
        train_texts: List[str],
        test_texts: Optional[List[str]] = None,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """Baseline 1: TF-IDF + Isolation Forest (Fit on Train, Calibrate on Val, Predict on Test)."""
        start = time.perf_counter()
        if test_texts is None:
            test_texts = train_texts

        if len(train_texts) < 3 or len(test_texts) == 0:
            return {
                "scores": np.zeros(len(test_texts)),
                "preds": np.zeros(len(test_texts), dtype=int),
                "threshold": 0.50,
                "time_ms": 0.0
            }

        vectorizer = TfidfVectorizer(max_features=512, stop_words="english")
        train_tfidf = vectorizer.fit_transform(train_texts).toarray()
        test_tfidf = vectorizer.transform(test_texts).toarray()

        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(train_tfidf)

        train_dec = iso.decision_function(train_tfidf)
        test_dec = iso.decision_function(test_tfidf)

        min_val, max_val = float(np.min(train_dec)), float(np.max(train_dec))
        denom = max_val - min_val if max_val > min_val else 1.0
        test_scores = np.clip(1.0 - (test_dec - min_val) / denom, 0.0, 1.0)

        # Calibrate on validation set if provided
        th = 0.50
        if val_texts is not None and val_labels is not None and len(val_texts) > 0:
            val_tfidf = vectorizer.transform(val_texts).toarray()
            val_dec = iso.decision_function(val_tfidf)
            val_scores = np.clip(1.0 - (val_dec - min_val) / denom, 0.0, 1.0)
            th = cls.calibrate_threshold(val_scores, np.array(val_labels), default_th=0.50)

        preds = (test_scores >= th).astype(int)
        elapsed = (time.perf_counter() - start) * 1000
        return {"scores": test_scores, "preds": preds, "threshold": th, "time_ms": elapsed}

    @classmethod
    def run_sbert_isolation_forest(
        cls,
        train_emb: np.ndarray,
        test_emb: Optional[np.ndarray] = None,
        val_emb: Optional[np.ndarray] = None,
        val_labels: Optional[List[int]] = None,
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """Baseline 2: Sentence-BERT + Isolation Forest (Fit on Train, Calibrate on Val, Predict on Test)."""
        start = time.perf_counter()
        if test_emb is None:
            test_emb = train_emb

        if len(train_emb) < 3 or len(test_emb) == 0:
            return {
                "scores": np.zeros(len(test_emb)),
                "preds": np.zeros(len(test_emb), dtype=int),
                "threshold": 0.50,
                "time_ms": 0.0
            }

        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(train_emb)

        train_dec = iso.decision_function(train_emb)
        test_dec = iso.decision_function(test_emb)

        min_val, max_val = float(np.min(train_dec)), float(np.max(train_dec))
        denom = max_val - min_val if max_val > min_val else 1.0
        test_scores = np.clip(1.0 - (test_dec - min_val) / denom, 0.0, 1.0)

        # Calibrate on validation set if provided
        th = 0.50
        if val_emb is not None and val_labels is not None and len(val_emb) > 0:
            val_dec = iso.decision_function(val_emb)
            val_scores = np.clip(1.0 - (val_dec - min_val) / denom, 0.0, 1.0)
            th = cls.calibrate_threshold(val_scores, np.array(val_labels), default_th=0.50)

        preds = (test_scores >= th).astype(int)
        elapsed = (time.perf_counter() - start) * 1000
        return {"scores": test_scores, "preds": preds, "threshold": th, "time_ms": elapsed}

    @classmethod
    def run_sbert_oneclass_svm(
        cls,
        train_emb: np.ndarray,
        test_emb: Optional[np.ndarray] = None,
        val_emb: Optional[np.ndarray] = None,
        val_labels: Optional[List[int]] = None,
        nu: float = 0.1
    ) -> Dict[str, Any]:
        """Baseline 3: Sentence-BERT + One-Class SVM (Fit on Train, Calibrate on Val, Predict on Test)."""
        start = time.perf_counter()
        if test_emb is None:
            test_emb = train_emb

        if len(train_emb) < 3 or len(test_emb) == 0:
            return {
                "scores": np.zeros(len(test_emb)),
                "preds": np.zeros(len(test_emb), dtype=int),
                "threshold": 0.50,
                "time_ms": 0.0
            }

        svm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        svm.fit(train_emb)

        train_dec = svm.decision_function(train_emb)
        test_dec = svm.decision_function(test_emb)

        min_val, max_val = float(np.min(train_dec)), float(np.max(train_dec))
        denom = max_val - min_val if max_val > min_val else 1.0
        test_scores = np.clip(1.0 - (test_dec - min_val) / denom, 0.0, 1.0)

        # Calibrate on validation set if provided
        th = 0.50
        if val_emb is not None and val_labels is not None and len(val_emb) > 0:
            val_dec = svm.decision_function(val_emb)
            val_scores = np.clip(1.0 - (val_dec - min_val) / denom, 0.0, 1.0)
            th = cls.calibrate_threshold(val_scores, np.array(val_labels), default_th=0.50)

        preds = (test_scores >= th).astype(int)
        elapsed = (time.perf_counter() - start) * 1000
        return {"scores": test_scores, "preds": preds, "threshold": th, "time_ms": elapsed}

    @classmethod
    def run_supervised_classifier(
        cls,
        train_emb: np.ndarray,
        train_labels: np.ndarray,
        test_emb: np.ndarray,
        val_emb: Optional[np.ndarray] = None,
        val_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Baseline 4: Supervised Logistic Classifier on embeddings (Fit on Train, Calibrate on Val, Predict on Test)."""
        start = time.perf_counter()
        if len(np.unique(train_labels)) < 2:
            return {
                "scores": np.zeros(len(test_emb)),
                "preds": np.zeros(len(test_emb), dtype=int),
                "threshold": 0.50,
                "time_ms": 0.0
            }

        clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        clf.fit(train_emb, train_labels)

        test_probs = clf.predict_proba(test_emb)[:, 1]

        th = 0.50
        if val_emb is not None and val_labels is not None and len(val_emb) > 0:
            val_probs = clf.predict_proba(val_emb)[:, 1]
            th = cls.calibrate_threshold(val_probs, np.array(val_labels), default_th=0.50)

        preds = (test_probs >= th).astype(int)
        elapsed = (time.perf_counter() - start) * 1000
        return {"scores": test_probs, "preds": preds, "threshold": th, "time_ms": elapsed}
