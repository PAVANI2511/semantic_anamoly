"""
Interactive Plotly visualizers for Semantic Sentinel research dashboard.
Produces publication-quality charts for sentence scores, category distributions,
confusion matrices, and baseline comparisons.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def create_sentence_score_chart(
    anomaly_scores: List[float],
    threshold: float,
    categories: List[str],
    is_anomaly: List[bool]
) -> go.Figure:
    """Creates interactive sentence-level anomaly score timeline with category badges."""
    x = [f"S{i+1}" for i in range(len(anomaly_scores))]
    
    # Colors according to category/anomaly
    colors = []
    for flag, cat in zip(is_anomaly, categories):
        if not flag:
            colors.append("#2ecc71")  # Green for normal
        elif cat == "topic_deviation":
            colors.append("#e74c3c")  # Red
        elif cat == "contradiction":
            colors.append("#9b59b6")  # Purple
        elif cat in {"entity_inconsistency", "attribute_inconsistency"}:
            colors.append("#e67e22")  # Orange
        else:
            colors.append("#f1c40f")  # Yellow

    fig = go.Figure()

    # Base line connecting all scores
    fig.add_trace(go.Scatter(
        x=x,
        y=anomaly_scores,
        mode="lines",
        name="Anomaly Trajectory",
        line=dict(color="#95a5a6", width=1.5, dash="dot"),
        hoverinfo="skip"
    ))

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x,
        y=anomaly_scores,
        mode="markers",
        name="Sentences",
        marker=dict(
            size=11,
            color=colors,
            line=dict(width=1.5, color="#2c3e50")
        ),
        text=[f"Score: {s:.3f}<br>Category: {c}" for s, c in zip(anomaly_scores, categories)],
        hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>"
    ))

    # Threshold horizontal reference line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#c0392b",
        line_width=2,
        annotation_text=f"Decision Threshold ({threshold:.2f})",
        annotation_position="top right"
    )

    fig.update_layout(
        title="Sentence-by-Sentence Anomaly Score Trajectory",
        xaxis_title="Sentence Index",
        yaxis_title="DGSA Anomaly Score A(i)",
        yaxis=dict(range=[-0.05, 1.05]),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_signal_decomposition_chart(
    global_scores: List[float],
    local_scores: List[float],
    contradiction_scores: List[float]
) -> go.Figure:
    """Stacked or grouped bar chart displaying individual component contributions."""
    x = [f"S{i+1}" for i in range(len(global_scores))]

    fig = go.Figure(data=[
        go.Bar(name="Global Deviation G(i)", x=x, y=global_scores, marker_color="#3498db"),
        go.Bar(name="Local Inconsistency L(i)", x=x, y=local_scores, marker_color="#f39c12"),
        go.Bar(name="Contradiction C(i)", x=x, y=contradiction_scores, marker_color="#9b59b6")
    ])

    fig.update_layout(
        barmode="group",
        title="Multi-Granular Signal Breakdown across Sentences",
        xaxis_title="Sentence Index",
        yaxis_title="Normalized Signal Magnitude",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_category_pie_chart(categories: List[str], is_anomaly: List[bool]) -> go.Figure:
    """Pie chart showing distribution of detected anomaly types."""
    anom_cats = [c.replace("_", " ").title() for c, a in zip(categories, is_anomaly) if a]
    if not anom_cats:
        anom_cats = ["Fully Coherent (No Anomalies)"]

    counts = pd.Series(anom_cats).value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    fig = px.pie(
        counts,
        names="Category",
        values="Count",
        title="Distribution of Anomaly Categories",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=320
    )
    return fig


def create_confusion_matrix_heatmap(cm: List[List[int]]) -> go.Figure:
    """Confusion matrix heatmap with TP, FP, TN, FN annotated."""
    z = cm
    x = ["Predicted Normal", "Predicted Anomaly"]
    y = ["Actual Normal", "Actual Anomaly"]
    
    annotations = []
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=x[j], y=y[i],
                text=f"<b>{labels[i][j]}</b><br>{z[i][j]}",
                showarrow=False,
                font=dict(size=14, color="white" if z[i][j] > np.max(z)/2 else "black")
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        colorscale="Blues",
        showscale=False
    ))
    fig.update_layout(
        title="Sentence-Level Confusion Matrix",
        annotations=annotations,
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        height=320
    )
    return fig


def create_baseline_comparison_chart(benchmark_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing F1, Precision, and Recall across algorithms."""
    fig = go.Figure()
    models = benchmark_df["Model"].tolist()

    fig.add_trace(go.Bar(name="F1-Score", x=models, y=benchmark_df["F1-Score"], marker_color="#2980b9"))
    fig.add_trace(go.Bar(name="Precision", x=models, y=benchmark_df["Precision"], marker_color="#27ae60"))
    fig.add_trace(go.Bar(name="Recall", x=models, y=benchmark_df["Recall"], marker_color="#e67e22"))

    fig.update_layout(
        barmode="group",
        title="Baseline Models Comparative Performance",
        yaxis_title="Score [0.0 - 1.0]",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

