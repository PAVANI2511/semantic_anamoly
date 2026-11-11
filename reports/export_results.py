"""
Report Generation and Export Utilities.
Exports analysis outcomes to JSON, CSV, and formatted HTML/PDF-ready reports
while preserving sentence indexes, character offsets, and span localizations.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import csv
import pandas as pd

from utils.logging_config import logger
from models.dgsa_model import DGSAResult


class ExportManager:
    """Manages serializing DGSA results into JSON, CSV, and HTML reports."""

    @staticmethod
    def export_to_json(result: DGSAResult, output_path: str | Path) -> str:
        """Serializes complete structured results into JSON format."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "total_sentences": result.total_sentences,
                "anomalous_sentences": result.num_anomalous,
                "normal_sentences": result.num_normal,
                "anomaly_percentage": result.anomaly_percentage,
                "threshold": result.threshold_used,
                "weights": result.weights_used,
                "inference_time_ms": result.inference_time_ms,
                "document_assessment": result.document_assessment,
                "embedding_model": result.embedding_model_name
            },
            "sentences": []
        }

        for i, s in enumerate(result.sentences):
            exp = result.explanations[i]
            data["sentences"].append({
                "sentence_id": s.sentence_id,
                "text": s.text,
                "paragraph_id": s.paragraph_id,
                "character_offsets": {"start": s.start_char, "end": s.end_char},
                "is_anomalous": result.is_anomaly[i],
                "anomaly_score": result.anomaly_scores[i],
                "category": result.categories[i],
                "component_scores": {
                    "global_deviation": result.global_scores[i],
                    "local_inconsistency": result.local_scores[i],
                    "contradiction": result.contradiction_scores[i]
                },
                "explanation": exp.contextual_explanation,
                "primary_contributing_signal": exp.primary_factor,
                "localized_spans": exp.localized_spans
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported JSON report to {path}")
        return str(path)

    @staticmethod
    def export_to_csv(result: DGSAResult, output_path: str | Path) -> str:
        """Exports tabular sentence analysis report to CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, s in enumerate(result.sentences):
            exp = result.explanations[i]
            spans_str = "; ".join([f"{sp['text']}[{sp['start_char']}:{sp['end_char']}]" for sp in exp.localized_spans])
            rows.append({
                "Sentence_ID": s.sentence_id,
                "Paragraph_ID": s.paragraph_id,
                "Text": s.text,
                "Start_Char": s.start_char,
                "End_Char": s.end_char,
                "Is_Anomalous": result.is_anomaly[i],
                "DGSA_Score": result.anomaly_scores[i],
                "Category": result.categories[i],
                "Global_Score": result.global_scores[i],
                "Local_Score": result.local_scores[i],
                "Contradiction_Score": result.contradiction_scores[i],
                "Explanation": exp.contextual_explanation,
                "Localized_Spans": spans_str
            })

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"Exported CSV report to {path}")
        return str(path)

    @staticmethod
    def export_to_html_report(result: DGSAResult, output_path: str | Path) -> str:
        """Generates an executive HTML report printable or savable as PDF."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows_html = []
        for i, s in enumerate(result.sentences):
            is_anom = result.is_anomaly[i]
            badge_class = "badge-danger" if is_anom else "badge-success"
            status_text = "ANOMALY" if is_anom else "NORMAL"
            cat_text = result.categories[i].replace("_", " ").title() if is_anom else "Consistent"
            exp = result.explanations[i]

            spans_html = ""
            if exp.localized_spans:
                spans_html = "<br><small><b>Spans:</b> " + ", ".join([f"<code>{sp['text']}</code>" for sp in exp.localized_spans]) + "</small>"

            rows_html.append(f"""
            <tr class="{'table-warning' if is_anom else ''}">
                <td><b>#{s.sentence_id + 1}</b></td>
                <td>{s.text}{spans_html}</td>
                <td><span class="{badge_class}">{status_text}</span></td>
                <td><b>{result.anomaly_scores[i]:.3f}</b></td>
                <td>{cat_text}</td>
                <td><small>{exp.contextual_explanation}</small></td>
            </tr>
            """)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Semantic Sentinel - Document Anomaly Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 30px; color: #2c3e50; line-height: 1.5; }}
        h1 {{ color: #1a365d; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
        .summary-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 25px; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; text-align: center; }}
        .metric-card {{ background: white; padding: 12px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .metric-val {{ font-size: 24px; font-weight: bold; color: #2b6cb0; }}
        .metric-lbl {{ font-size: 12px; color: #718096; text-transform: uppercase; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; border: 1px solid #e2e8f0; text-align: left; font-size: 14px; }}
        th {{ background: #edf2f7; font-weight: 600; }}
        .badge-danger {{ background: #fed7d7; color: #9b2c2c; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
        .badge-success {{ background: #c6f6d5; color: #22543d; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
        .table-warning {{ background-color: #fffaf0; }}
        code {{ background: #edf2f7; padding: 2px 4px; border-radius: 3px; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Semantic Sentinel Analysis Report</h1>
    <div class="summary-box">
        <h3>Executive Assessment: {result.document_assessment}</h3>
        <p><b>Embedding Model:</b> {result.embedding_model_name} | <b>Threshold:</b> {result.threshold_used} | <b>Latency:</b> {result.inference_time_ms} ms</p>
        <div class="grid">
            <div class="metric-card"><div class="metric-val">{result.total_sentences}</div><div class="metric-lbl">Total Sentences</div></div>
            <div class="metric-card"><div class="metric-val">{result.num_anomalous}</div><div class="metric-lbl">Anomalies Detected</div></div>
            <div class="metric-card"><div class="metric-val">{result.anomaly_percentage}%</div><div class="metric-lbl">Anomaly Rate</div></div>
            <div class="metric-card"><div class="metric-val">{result.num_normal}</div><div class="metric-lbl">Normal Sentences</div></div>
        </div>
    </div>

    <h3>Detailed Sentence-Level Audit</h3>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th style="width: 40%;">Sentence Text</th>
                <th>Status</th>
                <th>Score A(i)</th>
                <th>Category</th>
                <th>Explanation & Evidence</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows_html)}
        </tbody>
    </table>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Exported HTML/PDF report to {path}")
        return str(path)
