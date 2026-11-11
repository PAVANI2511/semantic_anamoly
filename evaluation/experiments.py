"""
Research experimentation harness:
- Baseline comparative benchmark
- Module ablation experiments
- Threshold calibration sweep
- Category-wise error analysis
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from utils.logging_config import logger
from models.dgsa_model import DGSAModel
from evaluation.baselines import BaselineRunner
from evaluation.metrics import compute_sentence_metrics, ClassificationMetrics


class ExperimentHarness:
    """Executes reproducible experiments and benchmarks on annotated datasets."""

    def __init__(self, dgsa_model: DGSAModel):
        self.dgsa_model = dgsa_model

    def run_comparative_benchmark(
        self,
        sentences_texts: List[str],
        y_true: List[int],
        embeddings: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Runs and compares all 5 models on the given test data.
        Returns a structured DataFrame of results.
        """
        if embeddings is None:
            embeddings = self.dgsa_model.embedding_model.embed_sentences(sentences_texts)

        results = []

        # 1. TF-IDF + Isolation Forest
        res_tfidf = BaselineRunner.run_tfidf_isolation_forest(sentences_texts)
        m_tfidf = compute_sentence_metrics(y_true, res_tfidf["preds"], res_tfidf["scores"])
        results.append(self._format_row("TF-IDF + Isolation Forest", m_tfidf, res_tfidf["time_ms"], len(sentences_texts)))

        # 2. Sentence-BERT + Isolation Forest
        res_sbert_if = BaselineRunner.run_sbert_isolation_forest(embeddings)
        m_sbert_if = compute_sentence_metrics(y_true, res_sbert_if["preds"], res_sbert_if["scores"])
        results.append(self._format_row("S-BERT + Isolation Forest", m_sbert_if, res_sbert_if["time_ms"], len(sentences_texts)))

        # 3. Sentence-BERT + One-Class SVM
        res_sbert_svm = BaselineRunner.run_sbert_oneclass_svm(embeddings)
        m_sbert_svm = compute_sentence_metrics(y_true, res_sbert_svm["preds"], res_sbert_svm["scores"])
        results.append(self._format_row("S-BERT + One-Class SVM", m_sbert_svm, res_sbert_svm["time_ms"], len(sentences_texts)))

        # 4. Supervised Linear Baseline (50/50 split if enough samples)
        if len(sentences_texts) >= 8 and len(np.unique(y_true)) > 1:
            split = len(sentences_texts) // 2
            res_sup = BaselineRunner.run_supervised_classifier(
                embeddings[:split], np.array(y_true[:split]), embeddings[split:]
            )
            m_sup = compute_sentence_metrics(y_true[split:], res_sup["preds"], res_sup["scores"])
            results.append(self._format_row("Supervised Linear (50% train)", m_sup, res_sup["time_ms"], len(sentences_texts) - split))

        # 5. Proposed DGSA Framework
        full_text = " ".join(sentences_texts)
        dgsa_res = self.dgsa_model.analyze_document(full_text)
        dgsa_preds = [1 if a else 0 for a in dgsa_res.is_anomaly]
        m_dgsa = compute_sentence_metrics(y_true, dgsa_preds, dgsa_res.anomaly_scores)
        results.append(self._format_row("DGSA Framework (Proposed)", m_dgsa, dgsa_res.inference_time_ms, len(sentences_texts)))

        return pd.DataFrame(results)

    def run_ablation_study(
        self,
        full_text: str,
        y_true: List[int]
    ) -> pd.DataFrame:
        """
        Runs ablation experiments evaluating the individual impact of G(i), L(i), and C(i).
        """
        configs = [
            ("DGSA Full (Global + Local + Contradiction)", True, True, True),
            ("DGSA w/o Global Context (Local + Contra only)", False, True, True),
            ("DGSA w/o Local Context (Global + Contra only)", True, False, True),
            ("DGSA w/o Contradiction (Global + Local only)", True, True, False),
            ("Global Context Only", True, False, False),
            ("Local Context Only", False, True, False),
        ]

        rows = []
        for name, en_g, en_l, en_c in configs:
            res = self.dgsa_model.analyze_document(
                text=full_text,
                enable_global=en_g,
                enable_local=en_l,
                enable_contradiction=en_c
            )
            preds = [1 if a else 0 for a in res.is_anomaly]
            metrics = compute_sentence_metrics(y_true, preds, res.anomaly_scores)
            rows.append(self._format_row(name, metrics, res.inference_time_ms, len(res.sentences)))

        return pd.DataFrame(rows)

    def calibrate_threshold(
        self,
        anomaly_scores: List[float],
        y_true: List[int],
        steps: int = 17
    ) -> Tuple[float, pd.DataFrame]:
        """
        Sweeps threshold from 0.10 to 0.90 to find the optimal decision boundary.
        """
        thresholds = np.linspace(0.10, 0.90, steps)
        records = []
        best_th = 0.50
        best_f1 = -1.0

        for th in thresholds:
            preds = [1 if s >= th else 0 for s in anomaly_scores]
            m = compute_sentence_metrics(y_true, preds, anomaly_scores)
            if m.f1 > best_f1:
                best_f1 = m.f1
                best_th = float(th)

            records.append({
                "Threshold": round(float(th), 2),
                "F1 Score": m.f1,
                "Precision": m.precision,
                "Recall": m.recall,
                "Accuracy": m.accuracy
            })

        return best_th, pd.DataFrame(records)

    def _format_row(self, model_name: str, m: ClassificationMetrics, time_ms: float, count: int) -> Dict[str, Any]:
        return {
            "Model": model_name,
            "Accuracy": m.accuracy,
            "Precision": m.precision,
            "Recall": m.recall,
            "F1-Score": m.f1,
            "ROC-AUC": m.roc_auc if m.roc_auc is not None else 0.0,
            "PR-AUC": m.pr_auc if m.pr_auc is not None else 0.0,
            "Latency (ms)": round(time_ms, 2),
            "Latency/Sent (ms)": round(time_ms / max(count, 1), 3),
            "TP": m.tp,
            "FP": m.fp,
            "TN": m.tn,
            "FN": m.fn
        }
