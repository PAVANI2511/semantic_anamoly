"""Evaluation package for Semantic Sentinel."""
from .metrics import (
    ClassificationMetrics,
    TokenSpanMetrics,
    EvaluationReport,
    compute_sentence_metrics,
    compute_token_span_metrics
)
from .baselines import BaselineRunner
from .experiments import ExperimentHarness
from .visualization import (
    create_sentence_score_chart,
    create_signal_decomposition_chart,
    create_category_pie_chart,
    create_confusion_matrix_heatmap,
    create_baseline_comparison_chart
)

__all__ = [
    "ClassificationMetrics",
    "TokenSpanMetrics",
    "EvaluationReport",
    "compute_sentence_metrics",
    "compute_token_span_metrics",
    "BaselineRunner",
    "ExperimentHarness",
    "create_sentence_score_chart",
    "create_signal_decomposition_chart",
    "create_category_pie_chart",
    "create_confusion_matrix_heatmap",
    "create_baseline_comparison_chart"
]
