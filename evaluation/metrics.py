"""
Evaluation metrics module for Semantic Sentinel.
Calculates sentence-level and token-level classification metrics,
confusion matrices, and span overlap evaluations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)


@dataclass
class ClassificationMetrics:
    """Comprehensive sentence-level evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    tp: int
    fp: int
    tn: int
    fn: int
    confusion_matrix: List[List[int]]


@dataclass
class TokenSpanMetrics:
    """Token-level and span-level localization evaluation metrics."""
    token_precision: float
    token_recall: float
    token_f1: float
    mean_span_iou: float
    evaluated_spans_count: int


@dataclass
class EvaluationReport:
    """Comprehensive research evaluation report."""
    sentence_metrics: ClassificationMetrics
    token_metrics: Optional[TokenSpanMetrics]
    inference_time_ms: float
    latency_per_sentence_ms: float


def compute_sentence_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    y_scores: Optional[List[float] | np.ndarray] = None
) -> ClassificationMetrics:
    """
    Computes standard binary classification metrics for sentence-level detection.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # ROC-AUC & PR-AUC if continuous scores provided
    roc_auc_val = None
    pr_auc_val = None
    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            scores = np.asarray(y_scores, dtype=float)
            roc_auc_val = float(roc_auc_score(y_true, scores))
            p_curve, r_curve, _ = precision_recall_curve(y_true, scores)
            pr_auc_val = float(auc(r_curve, p_curve))
        except Exception:
            roc_auc_val = None
            pr_auc_val = None

    return ClassificationMetrics(
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        roc_auc=round(roc_auc_val, 4) if roc_auc_val is not None else None,
        pr_auc=round(pr_auc_val, 4) if pr_auc_val is not None else None,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        confusion_matrix=cm.tolist()
    )


def compute_token_span_metrics(
    true_spans_per_doc: List[List[Tuple[int, int]]],
    pred_spans_per_doc: List[List[Tuple[int, int]]],
    doc_lengths: List[int]
) -> TokenSpanMetrics:
    """
    Evaluates token/character-level span localization accuracy using Intersection-over-Union (IoU)
    and binary character coverage masks.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    ious = []

    for true_spans, pred_spans, length in zip(true_spans_per_doc, pred_spans_per_doc, doc_lengths):
        true_mask = np.zeros(length, dtype=bool)
        pred_mask = np.zeros(length, dtype=bool)

        for s, e in true_spans:
            true_mask[min(s, length):min(e, length)] = True
        for s, e in pred_spans:
            pred_mask[min(s, length):min(e, length)] = True

        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()

        tp = intersection
        fp = np.logical_and(~true_mask, pred_mask).sum()
        fn = np.logical_and(true_mask, ~pred_mask).sum()

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if union > 0:
            ious.append(intersection / union)

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    return TokenSpanMetrics(
        token_precision=round(prec, 4),
        token_recall=round(rec, 4),
        token_f1=round(f1, 4),
        mean_span_iou=round(mean_iou, 4),
        evaluated_spans_count=len(ious)
    )
