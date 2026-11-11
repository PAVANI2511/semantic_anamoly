"""
Dedicated Span Localization Evaluation Module.
Computes token-level precision, recall, F1, exact span match, partial span match,
and span Intersection-over-Union (IoU) across documents and anomaly categories.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class ComprehensiveSpanMetrics:
    """Comprehensive token and span localization metrics."""
    token_precision: float
    token_recall: float
    token_f1: float
    exact_match_count: int
    exact_match_rate: float
    partial_match_count: int
    partial_match_rate: float
    mean_iou: float
    median_iou: float
    total_ground_truth_spans: int
    total_predicted_spans: int
    category_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_precision": self.token_precision,
            "token_recall": self.token_recall,
            "token_f1": self.token_f1,
            "exact_match_count": self.exact_match_count,
            "exact_match_rate": self.exact_match_rate,
            "partial_match_count": self.partial_match_count,
            "partial_match_rate": self.partial_match_rate,
            "mean_iou": self.mean_iou,
            "median_iou": self.median_iou,
            "total_ground_truth_spans": self.total_ground_truth_spans,
            "total_predicted_spans": self.total_predicted_spans,
            "category_metrics": self.category_metrics
        }


class SpanEvaluator:
    """
    Evaluates fine-grained span localization accuracy against gold-standard annotated spans.
    Supports token-level character-mask evaluation, exact boundary matching,
    partial overlap (IoU > 0.0), and IoU distribution statistics.
    """

    @staticmethod
    def compute_iou(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> float:
        """Computes 1D intersection over union between two character intervals."""
        start_a, end_a = span_a
        start_b, end_b = span_b
        inter_start = max(start_a, start_b)
        inter_end = min(end_a, end_b)
        intersection = max(0, inter_end - inter_start)
        union = max(end_a, end_b) - min(start_a, start_b)
        return float(intersection / union) if union > 0 else 0.0

    @classmethod
    def evaluate_spans(
        cls,
        gold_spans: List[Dict[str, Any]],
        pred_spans: List[Dict[str, Any]],
        doc_length: int,
        iou_match_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Evaluates predicted spans for a single document against gold standard spans.
        Each span dict has: 'start_char', 'end_char', and optionally 'category', 'text'.
        """
        # Character-level binary masks for token/character precision-recall
        gold_mask = np.zeros(doc_length, dtype=bool)
        for s in gold_spans:
            sc, ec = max(0, int(s["start_char"])), min(doc_length, int(s["end_char"]))
            if ec > sc:
                gold_mask[sc:ec] = True

        pred_mask = np.zeros(doc_length, dtype=bool)
        for s in pred_spans:
            sc, ec = max(0, int(s["start_char"])), min(doc_length, int(s["end_char"]))
            if ec > sc:
                pred_mask[sc:ec] = True

        tp_chars = int(np.logical_and(gold_mask, pred_mask).sum())
        fp_chars = int(np.logical_and(~gold_mask, pred_mask).sum())
        fn_chars = int(np.logical_and(gold_mask, ~pred_mask).sum())

        char_prec = float(tp_chars / (tp_chars + fp_chars)) if (tp_chars + fp_chars) > 0 else 0.0
        char_rec = float(tp_chars / (tp_chars + fn_chars)) if (tp_chars + fn_chars) > 0 else 0.0
        char_f1 = float(2 * char_prec * char_rec / (char_prec + char_rec)) if (char_prec + char_rec) > 0 else 0.0

        # Boundary and Overlap Evaluation
        exact_matches = 0
        partial_matches = 0
        ious = []

        gold_intervals = [(int(s["start_char"]), int(s["end_char"])) for s in gold_spans if int(s["end_char"]) > int(s["start_char"])]
        pred_intervals = [(int(s["start_char"]), int(s["end_char"])) for s in pred_spans if int(s["end_char"]) > int(s["start_char"])]

        for g_start, g_end in gold_intervals:
            best_iou = 0.0
            is_exact = False
            for p_start, p_end in pred_intervals:
                if g_start == p_start and g_end == p_end:
                    is_exact = True
                curr_iou = cls.compute_iou((g_start, g_end), (p_start, p_end))
                if curr_iou > best_iou:
                    best_iou = curr_iou

            if is_exact:
                exact_matches += 1
            if best_iou > 0.0:
                partial_matches += 1
            if best_iou > 0.0 or len(pred_intervals) > 0:
                ious.append(best_iou)

        return {
            "tp_chars": tp_chars,
            "fp_chars": fp_chars,
            "fn_chars": fn_chars,
            "char_precision": char_prec,
            "char_recall": char_rec,
            "char_f1": char_f1,
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "gold_count": len(gold_intervals),
            "pred_count": len(pred_intervals),
            "ious": ious
        }

    @classmethod
    def evaluate_benchmark(
        cls,
        benchmark_records: List[Dict[str, Any]]
    ) -> ComprehensiveSpanMetrics:
        """
        Runs comprehensive evaluation across an entire benchmark dataset.
        
        benchmark_records format:
        [
            {
                "doc_id": str,
                "doc_text": str,
                "gold_spans": [{"start_char": int, "end_char": int, "category": str, ...}],
                "pred_spans": [{"start_char": int, "end_char": int, "category": str, ...}]
            }
        ]
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_exact = 0
        total_partial = 0
        total_gold = 0
        total_pred = 0
        all_ious: List[float] = []

        category_stats: Dict[str, Dict[str, Any]] = {}

        for rec in benchmark_records:
            doc_text = rec.get("doc_text", "")
            doc_len = len(doc_text)
            gold_spans = rec.get("gold_spans", [])
            pred_spans = rec.get("pred_spans", [])

            doc_eval = cls.evaluate_spans(gold_spans, pred_spans, doc_len)

            total_tp += doc_eval["tp_chars"]
            total_fp += doc_eval["fp_chars"]
            total_fn += doc_eval["fn_chars"]
            total_exact += doc_eval["exact_matches"]
            total_partial += doc_eval["partial_matches"]
            total_gold += doc_eval["gold_count"]
            total_pred += doc_eval["pred_count"]
            all_ious.extend(doc_eval["ious"])

            # Per-category tracking
            for g in gold_spans:
                cat = g.get("category", "unspecified")
                if cat not in category_stats:
                    category_stats[cat] = {"gold": 0, "exact": 0, "partial": 0, "ious": []}
                category_stats[cat]["gold"] += 1

                g_interval = (int(g["start_char"]), int(g["end_char"]))
                best_cat_iou = 0.0
                is_cat_exact = False
                for p in pred_spans:
                    p_interval = (int(p["start_char"]), int(p["end_char"]))
                    if g_interval == p_interval:
                        is_cat_exact = True
                    ciou = cls.compute_iou(g_interval, p_interval)
                    if ciou > best_cat_iou:
                        best_cat_iou = ciou

                if is_cat_exact:
                    category_stats[cat]["exact"] += 1
                if best_cat_iou > 0.0:
                    category_stats[cat]["partial"] += 1
                    category_stats[cat]["ious"].append(best_cat_iou)

        prec = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        rec = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        exact_rate = float(total_exact / total_gold) if total_gold > 0 else 0.0
        partial_rate = float(total_partial / total_gold) if total_gold > 0 else 0.0
        mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
        median_iou = float(np.median(all_ious)) if all_ious else 0.0

        cat_metrics: Dict[str, Dict[str, float]] = {}
        for cat, stats in category_stats.items():
            g_cnt = stats["gold"]
            cat_metrics[cat] = {
                "gold_count": g_cnt,
                "exact_match_rate": round(float(stats["exact"] / g_cnt), 4) if g_cnt > 0 else 0.0,
                "partial_match_rate": round(float(stats["partial"] / g_cnt), 4) if g_cnt > 0 else 0.0,
                "mean_iou": round(float(np.mean(stats["ious"])), 4) if stats["ious"] else 0.0
            }

        return ComprehensiveSpanMetrics(
            token_precision=round(prec, 4),
            token_recall=round(rec, 4),
            token_f1=round(f1, 4),
            exact_match_count=total_exact,
            exact_match_rate=round(exact_rate, 4),
            partial_match_count=total_partial,
            partial_match_rate=round(partial_rate, 4),
            mean_iou=round(mean_iou, 4),
            median_iou=round(median_iou, 4),
            total_ground_truth_spans=total_gold,
            total_predicted_spans=total_pred,
            category_metrics=cat_metrics
        )
