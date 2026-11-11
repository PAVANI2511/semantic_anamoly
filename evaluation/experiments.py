"""
Research experimentation harness:
- Rigorous zero-leakage baseline comparative benchmark (Train / Val / Test)
- Module ablation experiments (7 configurations)
- Validation threshold calibration sweep
- Category-wise error analysis
- Span localization evaluation
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np

from utils.logging_config import logger
from models.dgsa_model import DGSAModel, DGSAResult
from evaluation.baselines import BaselineRunner
from evaluation.metrics import compute_sentence_metrics, ClassificationMetrics
from evaluation.span_evaluator import SpanEvaluator, ComprehensiveSpanMetrics
from data.dataset_loader import AnnotatedDocument


class ExperimentHarness:
    """Executes reproducible experiments and benchmarks on annotated datasets."""

    def __init__(self, dgsa_model: DGSAModel):
        self.dgsa_model = dgsa_model

    def run_rigorous_benchmark(
        self,
        train_docs: List[AnnotatedDocument],
        val_docs: List[AnnotatedDocument],
        test_docs: List[AnnotatedDocument]
    ) -> Dict[str, Any]:
        """
        Executes zero-leakage scientific benchmark strictly adhering to Document-Level partitioning:
        - Baselines fit on train_docs.
        - DGSA decision threshold is calibrated on val_docs (maximizing F1).
        - All 6 models evaluated on held-out test_docs.
        """
        logger.info(
            f"Starting zero-leakage benchmark: {len(train_docs)} Train Docs, "
            f"{len(val_docs)} Val Docs, {len(test_docs)} Test Docs"
        )

        # 1. Flatten Train Data
        train_texts = [s.text for d in train_docs for s in d.sentences]
        train_labels = [s.is_anomaly for d in train_docs for s in d.sentences]
        train_emb = self.dgsa_model.embedding_model.embed_sentences(train_texts)

        # 2. Flatten Test Data (Document-by-document to maintain document-level context for DGSA)
        test_texts = [s.text for d in test_docs for s in d.sentences]
        test_labels = [s.is_anomaly for d in test_docs for s in d.sentences]
        test_emb = self.dgsa_model.embedding_model.embed_sentences(test_texts)

        # 3. Calibrate DGSA threshold on Validation Docs ONLY
        val_scores: List[float] = []
        val_labels: List[int] = []
        for v_doc in val_docs:
            v_text = " ".join(s.text for s in v_doc.sentences)
            v_res = self.dgsa_model.analyze_document(v_text)
            val_scores.extend(v_res.anomaly_scores)
            val_labels.extend([s.is_anomaly for s in v_doc.sentences])

        calibrated_th, calib_df = self.calibrate_threshold(val_scores, val_labels)
        logger.info(f"Threshold calibrated on Validation split: T* = {calibrated_th:.2f}")

        results = []

        # Model 1: TF-IDF + Isolation Forest (Fit on Train, Predict on Test)
        res_tfidf = BaselineRunner.run_tfidf_isolation_forest(train_texts, test_texts)
        m_tfidf = compute_sentence_metrics(test_labels, res_tfidf["preds"], res_tfidf["scores"])
        results.append(self._format_row("TF-IDF + Isolation Forest", m_tfidf, res_tfidf["time_ms"], len(test_texts)))

        # Model 2: S-BERT + Isolation Forest (Fit on Train, Predict on Test)
        res_sbert_if = BaselineRunner.run_sbert_isolation_forest(train_emb, test_emb)
        m_sbert_if = compute_sentence_metrics(test_labels, res_sbert_if["preds"], res_sbert_if["scores"])
        results.append(self._format_row("S-BERT + Isolation Forest", m_sbert_if, res_sbert_if["time_ms"], len(test_texts)))

        # Model 3: S-BERT + One-Class SVM (Fit on Train, Predict on Test)
        res_sbert_svm = BaselineRunner.run_sbert_oneclass_svm(train_emb, test_emb)
        m_sbert_svm = compute_sentence_metrics(test_labels, res_sbert_svm["preds"], res_sbert_svm["scores"])
        results.append(self._format_row("S-BERT + One-Class SVM", m_sbert_svm, res_sbert_svm["time_ms"], len(test_texts)))

        # Model 4: Supervised Logistic Classifier (Fit on Train, Predict on Test)
        res_sup = BaselineRunner.run_supervised_classifier(train_emb, np.array(train_labels), test_emb)
        m_sup = compute_sentence_metrics(test_labels, res_sup["preds"], res_sup["scores"])
        results.append(self._format_row("Supervised Logistic Regression", m_sup, res_sup["time_ms"], len(test_texts)))

        # Model 5: Proposed DGSA without NLI (Lexical fallback contradiction)
        dgsa_no_nli_preds: List[int] = []
        dgsa_no_nli_scores: List[float] = []
        start_no_nli = time.perf_counter()

        # Temporarily force lexical mode
        orig_backend = getattr(self.dgsa_model.contradiction_detector, "backend", "nli_transformer")
        self.dgsa_model.contradiction_detector.backend = "lexical_fallback"

        for t_doc in test_docs:
            d_text = " ".join(s.text for s in t_doc.sentences)
            d_res = self.dgsa_model.analyze_document(d_text, custom_threshold=calibrated_th)
            dgsa_no_nli_scores.extend(d_res.anomaly_scores)
            dgsa_no_nli_preds.extend([1 if s >= calibrated_th else 0 for s in d_res.anomaly_scores])

        elapsed_no_nli = (time.perf_counter() - start_no_nli) * 1000
        m_dgsa_no_nli = compute_sentence_metrics(test_labels, dgsa_no_nli_preds, dgsa_no_nli_scores)
        results.append(self._format_row("DGSA without NLI (Lexical Fallback)", m_dgsa_no_nli, elapsed_no_nli, len(test_texts)))

        # Restore original backend
        self.dgsa_model.contradiction_detector.backend = orig_backend

        # Model 6: Proposed DGSA Full (with NLI Transformer)
        dgsa_preds: List[int] = []
        dgsa_scores: List[float] = []
        test_doc_results: List[DGSAResult] = []
        span_records: List[Dict[str, Any]] = []

        start_dgsa = time.perf_counter()
        for t_doc in test_docs:
            d_text = " ".join(s.text for s in t_doc.sentences)
            d_res = self.dgsa_model.analyze_document(d_text, custom_threshold=calibrated_th)
            test_doc_results.append(d_res)
            dgsa_scores.extend(d_res.anomaly_scores)
            dgsa_preds.extend([1 if a else 0 for a in d_res.is_anomaly])

            # Collect spans for span evaluation
            gold_spans = []
            char_offset = 0
            for s in t_doc.sentences:
                s_idx = d_text.find(s.text, char_offset)
                if s_idx == -1:
                    s_idx = char_offset
                char_offset = s_idx + len(s.text)
                for sp in s.token_spans:
                    gold_spans.append({
                        "text": sp.get("text", ""),
                        "start_char": s_idx + sp.get("start_char", 0),
                        "end_char": s_idx + sp.get("end_char", len(sp.get("text", ""))),
                        "category": s.category
                    })

            pred_spans = []
            for exp in d_res.explanations:
                if exp.is_anomalous:
                    for sp in exp.localized_spans:
                        pred_spans.append({
                            "text": sp.get("text", ""),
                            "start_char": sp.get("start_char", 0),
                            "end_char": sp.get("end_char", 0),
                            "category": exp.category
                        })

            span_records.append({
                "doc_id": t_doc.doc_id,
                "doc_text": d_text,
                "gold_spans": gold_spans,
                "pred_spans": pred_spans
            })

        elapsed_dgsa = (time.perf_counter() - start_dgsa) * 1000
        m_dgsa = compute_sentence_metrics(test_labels, dgsa_preds, dgsa_scores)
        results.append(self._format_row("DGSA Full Framework (Proposed)", m_dgsa, elapsed_dgsa, len(test_texts)))

        benchmark_df = pd.DataFrame(results)

        # 4. Run Span Evaluation
        span_metrics = SpanEvaluator.evaluate_benchmark(span_records)

        # 5. Run Ablation Study on Test Docs across 7 configurations
        ablation_df = self._run_test_ablation(test_docs, test_labels, calibrated_th)

        return {
            "benchmark_results": benchmark_df,
            "ablation_results": ablation_df,
            "span_metrics": span_metrics,
            "calibrated_threshold": calibrated_th,
            "calibration_table": calib_df,
            "test_doc_results": test_doc_results
        }

    def _run_test_ablation(
        self,
        test_docs: List[AnnotatedDocument],
        test_labels: List[int],
        threshold: float
    ) -> pd.DataFrame:
        """
        Runs rigorous ablation across all 7 signal configurations on held-out test documents.
        """
        configs = [
            ("Global Only (alpha=1.0, beta=0, gamma=0)", True, False, False, {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}),
            ("Local Only (alpha=0, beta=1.0, gamma=0)", False, True, False, {"alpha": 0.0, "beta": 1.0, "gamma": 0.0}),
            ("Contradiction Only (alpha=0, beta=0, gamma=1.0)", False, False, True, {"alpha": 0.0, "beta": 0.0, "gamma": 1.0}),
            ("Global + Local (alpha=0.5, beta=0.5, gamma=0)", True, True, False, {"alpha": 0.5, "beta": 0.5, "gamma": 0.0}),
            ("Global + Contradiction (alpha=0.5, beta=0, gamma=0.5)", True, False, True, {"alpha": 0.5, "beta": 0.0, "gamma": 0.5}),
            ("Local + Contradiction (alpha=0, beta=0.5, gamma=0.5)", False, True, True, {"alpha": 0.0, "beta": 0.5, "gamma": 0.5}),
            ("Full DGSA (alpha=0.35, beta=0.40, gamma=0.25)", True, True, True, {"alpha": 0.35, "beta": 0.40, "gamma": 0.25}),
        ]

        rows = []
        for name, en_g, en_l, en_c, weights in configs:
            scores: List[float] = []
            preds: List[int] = []
            start = time.perf_counter()

            for doc in test_docs:
                text = " ".join(s.text for s in doc.sentences)
                res = self.dgsa_model.analyze_document(
                    text=text,
                    custom_threshold=threshold,
                    custom_weights=weights,
                    enable_global=en_g,
                    enable_local=en_l,
                    enable_contradiction=en_c
                )
                scores.extend(res.anomaly_scores)
                preds.extend([1 if a else 0 for a in res.is_anomaly])

            elapsed = (time.perf_counter() - start) * 1000
            m = compute_sentence_metrics(test_labels, preds, scores)
            rows.append(self._format_row(name, m, elapsed, len(test_labels)))

        return pd.DataFrame(rows)

    def run_comparative_benchmark(
        self,
        sentences_texts: List[str],
        y_true: List[int],
        embeddings: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Interactive benchmark for single-document testing in UI.
        """
        if embeddings is None:
            embeddings = self.dgsa_model.embedding_model.embed_sentences(sentences_texts)

        results = []

        # 1. TF-IDF + Isolation Forest
        res_tfidf = BaselineRunner.run_tfidf_isolation_forest(sentences_texts, sentences_texts)
        m_tfidf = compute_sentence_metrics(y_true, res_tfidf["preds"], res_tfidf["scores"])
        results.append(self._format_row("TF-IDF + Isolation Forest", m_tfidf, res_tfidf["time_ms"], len(sentences_texts)))

        # 2. Sentence-BERT + Isolation Forest
        res_sbert_if = BaselineRunner.run_sbert_isolation_forest(embeddings, embeddings)
        m_sbert_if = compute_sentence_metrics(y_true, res_sbert_if["preds"], res_sbert_if["scores"])
        results.append(self._format_row("S-BERT + Isolation Forest", m_sbert_if, res_sbert_if["time_ms"], len(sentences_texts)))

        # 3. Sentence-BERT + One-Class SVM
        res_sbert_svm = BaselineRunner.run_sbert_oneclass_svm(embeddings, embeddings)
        m_sbert_svm = compute_sentence_metrics(y_true, res_sbert_svm["preds"], res_sbert_svm["scores"])
        results.append(self._format_row("S-BERT + One-Class SVM", m_sbert_svm, res_sbert_svm["time_ms"], len(sentences_texts)))

        # 4. Supervised Linear Baseline
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
        Interactive ablation experiments for single document text.
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
        Sweeps threshold from 0.10 to 0.90 to find the optimal decision boundary on validation data.
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
