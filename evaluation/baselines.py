"""
Baseline anomaly detection models for scientific benchmarking.
Strict train/test separation to guarantee zero data or sentence leakage.
Implements:
1. TF-IDF + Isolation Forest (fitted on train, evaluated on test)
2. Sentence-BERT + Isolation Forest (fitted on train, evaluated on test)
3. Sentence-BERT + One-Class SVM (fitted on train, evaluated on test)
4. Supervised Logistic Regression (fitted on train, evaluated on test)
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


class BaselineRunner:
    """Orchestrates execution of comparison baselines with strict train/test split discipline."""

    @staticmethod
    def run_tfidf_isolation_forest(
        train_texts: List[str],
        test_texts: Optional[List[str]] = None,
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """Baseline 1: TF-IDF + Isolation Forest (Fit on Train, Predict on Test)."""
        start = time.perf_counter()
        if test_texts is None:
            test_texts = train_texts

        if len(train_texts) < 3 or len(test_texts) == 0:
            return {"scores": np.zeros(len(test_texts)), "preds": np.zeros(len(test_texts), dtype=int), "time_ms": 0.0}

        vectorizer = TfidfVectorizer(max_features=512, stop_words="english")
        train_tfidf = vectorizer.fit_transform(train_texts).toarray()
        test_tfidf = vectorizer.transform(test_texts).toarray()

        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(train_tfidf)

        # Anomaly score: negative decision function
        train_dec = iso.decision_function(train_tfidf)
        test_dec = iso.decision_function(test_tfidf)

        # Calibrate score range relative to training distribution
        min_val, max_val = float(np.min(train_dec)), float(np.max(train_dec))
        denom = max_val - min_val if max_val > min_val else 1.0
        norm_scores = np.clip(1.0 - (test_dec - min_val) / denom, 0.0, 1.0)
        preds = (norm_scores >= 0.50).astype(int)

        elapsed = (time.perf_counter() - start) * 1000
        return {"scores": norm_scores, "preds": preds, "time_ms": elapsed}

    @staticmethod
    def run_sbert_isolation_forest(
        train_emb: np.ndarray,
        test_emb: Optional[np.ndarray] = None,
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """Baseline 2: Sentence-BERT + Isolation Forest (Fit on Train, Predict on Test)."""
        start = time.perf_counter()
        if test_emb is None:
            test_emb = train_emb

        if len(train_emb) < 3 or len(test_emb) == 0:
            return {"scores": np.zeros(len(test_emb)), "preds": np.zeros(len(test_emb), dtype=int), "time_ms": 0.0}

        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(train_emb)

        train_dec = iso.decision_function(train_emb)
        test_dec = iso.decision_function(test_emb)

        min_val, max_val = float(np.min(train_dec)), float(np.max(train_dec))
        denom = max_val - min_val if max_val > min_val else 1.0
        norm_scores = np.clip(1.0 - (test_dec - min_val) / denom, 0.0, 1.0)
        preds = (norm_scores >= 0.50).astype(int)

        elapsed = (time.perf_counter() - start) * 1000
        return {"scores": norm_scores, "preds": preds, "time_ms": elapsed}

    @staticmethod
    def run_sbert_oneclass_svm(
        train_emb: np.ndarray,
        test_emb: Optional[np.ndarray] = None,
        nu: float = 0.1
    ) -> Dict[str, Any]:
        """Baseline 3: Sentence-BERT + One-Class SVM (Fit on Train, Predict on Test)."""
        start = time.perf_counter()
        if test_emb is None:
            test_emb = train_emb

        if len(train_emb) < 3 or len(test_emb) == 0:
            return {"scores": np.zeros(len(test_emb)), "preds": np.zeros(len(test_emb), dtype=int), "time_ms": 0.0}

        svm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        svm.fit(train_emb)

        train_dec = svm.decision_function(train_emb)
        test_dec = svm.decision_function(test_emb)

        min_val, max_val = float(np.min(train_dec)), float(np.max(train_dec))
        denom = max_val - min_val if max_val > min_val else 1.0
        norm_scores = np.clip(1.0 - (test_dec - min_val) / denom, 0.0, 1.0)
        preds = (norm_scores >= 0.50).astype(int)

        elapsed = (time.perf_counter() - start) * 1000
        return {"scores": norm_scores, "preds": preds, "time_ms": elapsed}

        elapsed = (time.perf_counter() - start) * 1000
        return {"scores": norm_scores, "preds": preds, "time_ms": elapsed}

    @staticmethod
    def run_supervised_classifier(
        train_emb: np.ndarray,
        train_labels: np.ndarray,
        test_emb: np.ndarray
    ) -> Dict[str, Any]:
        """Baseline 4: Supervised Logistic Classifier on embeddings (Fit on Train, Predict on Test)."""
        start = time.perf_counter()
        if len(np.unique(train_labels)) < 2:
            return {"scores": np.zeros(len(test_emb)), "preds": np.zeros(len(test_emb), dtype=int), "time_ms": 0.0}

        clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        clf.fit(train_emb, train_labels)

        probs = clf.predict_proba(test_emb)[:, 1]
        preds = (probs >= 0.50).astype(int)

        elapsed = (time.perf_counter() - start) * 1000
        return {"scores": probs, "preds": preds, "time_ms": elapsed}
