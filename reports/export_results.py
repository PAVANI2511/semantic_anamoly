"""
Report Generation and Export Utilities.
Exports analysis outcomes to JSON, CSV, secure HTML, and genuine PDF reports (via ReportLab)
while preserving sentence indexes, character offsets, and span localizations.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import csv
import html
import pandas as pd

from utils.logging_config import logger
from models.dgsa_model import DGSAResult


class ExportManager:
    """Manages serializing DGSA results into JSON, CSV, secure HTML, and authentic PDF reports."""

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
                "signals_breakdown": exp.signals_breakdown,
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
                "Primary_Signal": exp.primary_factor,
                "Explanation": exp.contextual_explanation,
                "Localized_Spans": spans_str
            })

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"Exported CSV report to {path}")
        return str(path)

    @staticmethod
    def export_to_html_report(result: DGSAResult, output_path: str | Path) -> str:
        """Generates a secure, styled HTML report with full XSS escaping."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows_html = []
        for i, s in enumerate(result.sentences):
            is_anom = result.is_anomaly[i]
            badge_class = "badge-danger" if is_anom else "badge-success"
            status_text = "ANOMALY" if is_anom else "NORMAL"
            cat_text = html.escape(result.categories[i].replace("_", " ").title()) if is_anom else "Consistent"
            exp = result.explanations[i]

            escaped_text = html.escape(s.text)
            escaped_explanation = html.escape(exp.contextual_explanation)

            spans_html = ""
            if exp.localized_spans:
                spans_parts = [f"<code>{html.escape(sp['text'])}</code>" for sp in exp.localized_spans]
                spans_html = "<br><small style='color: #c53030;'><b>Attributed Spans:</b> " + ", ".join(spans_parts) + "</small>"

            rows_html.append(f"""
            <tr class="{'table-warning' if is_anom else ''}">
                <td><b>#{s.sentence_id + 1}</b></td>
                <td>{escaped_text}{spans_html}</td>
                <td><span class="{badge_class}">{status_text}</span></td>
                <td><b>{result.anomaly_scores[i]:.3f}</b></td>
                <td>{cat_text}</td>
                <td><small>{escaped_explanation}</small></td>
            </tr>
            """)

        safe_doc_assessment = html.escape(result.document_assessment)
        safe_model_name = html.escape(result.embedding_model_name)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Semantic Sentinel - Document Anomaly Audit Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 30px; color: #2c3e50; line-height: 1.5; }}
        h1 {{ color: #1a365d; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
        .summary-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 25px; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; text-align: center; margin-top: 15px; }}
        .metric-card {{ background: white; padding: 12px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .metric-val {{ font-size: 24px; font-weight: bold; color: #2b6cb0; }}
        .metric-lbl {{ font-size: 12px; color: #718096; text-transform: uppercase; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; border: 1px solid #e2e8f0; text-align: left; font-size: 14px; vertical-align: top; }}
        th {{ background: #edf2f7; font-weight: 600; }}
        .badge-danger {{ background: #fed7d7; color: #9b2c2c; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
        .badge-success {{ background: #c6f6d5; color: #22543d; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
        .table-warning {{ background-color: #fffaf0; }}
        code {{ background: #fed7d7; color: #9b2c2c; padding: 2px 4px; border-radius: 3px; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Semantic Sentinel Analysis Report</h1>
    <div class="summary-box">
        <h3>Executive Assessment: {safe_doc_assessment}</h3>
        <p><b>Embedding Model:</b> {safe_model_name} | <b>Threshold:</b> {result.threshold_used} | <b>Latency:</b> {result.inference_time_ms} ms</p>
        <p><b>DGSA Formula:</b> A(i) = {result.weights_used.get('alpha', 0.35):.2f}&middot;G(i) + {result.weights_used.get('beta', 0.40):.2f}&middot;L(i) + {result.weights_used.get('gamma', 0.25):.2f}&middot;C(i)</p>
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
                <th style="width: 5%;">ID</th>
                <th style="width: 42%;">Sentence Text & Localized Spans</th>
                <th style="width: 8%;">Status</th>
                <th style="width: 8%;">Score A(i)</th>
                <th style="width: 15%;">Category</th>
                <th style="width: 22%;">Explanation & Contributing Signals</th>
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

        logger.info(f"Exported HTML report to {path}")
        return str(path)

    @staticmethod
    def export_to_pdf(result: DGSAResult, output_path: str | Path) -> str:
        """
        Generates an authentic, professionally styled PDF audit report using ReportLab.
        Includes executive summary, score breakdown, and formatted audit tables.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
        except ImportError as e:
            logger.error(f"ReportLab is not installed: {e}")
            raise RuntimeError("ReportLab is required for PDF generation. Install with: pip install reportlab")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1A365D"),
            alignment=0,
            spaceAfter=8
        )

        h2_style = ParagraphStyle(
            "Heading2Custom",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#2B6CB0"),
            spaceBefore=10,
            spaceAfter=6
        )

        body_style = ParagraphStyle(
            "BodyCustom",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#2D3748")
        )

        cell_style = ParagraphStyle(
            "CellNormal",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#2D3748")
        )

        cell_bold = ParagraphStyle(
            "CellBold",
            parent=cell_style,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1A202C")
        )

        cell_anom = ParagraphStyle(
            "CellAnomaly",
            parent=cell_style,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#C53030")
        )

        story = []

        # Title
        story.append(Paragraph("Semantic Sentinel – Document Anomaly Audit Report", title_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#CBD5E0"), spaceAfter=10))

        # Metadata & Assessment
        assessment_text = f"<b>Document Assessment:</b> {html.escape(result.document_assessment)}"
        story.append(Paragraph(assessment_text, body_style))
        story.append(Spacer(1, 4))

        meta_line = (
            f"<b>Model:</b> {html.escape(result.embedding_model_name)} | "
            f"<b>Decision Threshold:</b> {result.threshold_used} | "
            f"<b>Inference Latency:</b> {result.inference_time_ms:.1f} ms"
        )
        story.append(Paragraph(meta_line, body_style))
        story.append(Spacer(1, 8))

        # Metrics Summary Table
        metrics_data = [
            ["Total Sentences", "Anomalies Flagged", "Anomaly Percentage", "Normal Sentences"],
            [
                str(result.total_sentences),
                str(result.num_anomalous),
                f"{result.anomaly_percentage}%",
                str(result.num_normal)
            ]
        ]
        metrics_table = Table(metrics_data, colWidths=[130, 130, 130, 130])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#4A5568")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 1), (-1, 1), 14),
            ("TEXTCOLOR", (1, 1), (1, 1), colors.HexColor("#C53030") if result.num_anomalous > 0 else colors.HexColor("#22543D")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 14))

        # Detailed Audit Table
        story.append(Paragraph("Sentence-Level Anomaly Audit", h2_style))

        audit_headers = ["ID", "Sentence & Spans", "Status", "Score", "Category", "Explanation"]
        audit_rows = [[
            Paragraph(f"<b>{h}</b>", cell_bold) for h in audit_headers
        ]]

        for i, s in enumerate(result.sentences):
            is_anom = result.is_anomaly[i]
            exp = result.explanations[i]

            s_id_p = Paragraph(f"#{s.sentence_id + 1}", cell_bold)

            # Sentence text + spans
            text_p_content = html.escape(s.text)
            if exp.localized_spans:
                spans_str = ", ".join([html.escape(sp["text"]) for sp in exp.localized_spans])
                text_p_content += f"<br/><font color='#C53030'><b>Spans:</b> {spans_str}</font>"
            sent_p = Paragraph(text_p_content, cell_style)

            status_p = Paragraph("ANOMALY", cell_anom) if is_anom else Paragraph("NORMAL", cell_style)
            score_p = Paragraph(f"{result.anomaly_scores[i]:.3f}", cell_bold if is_anom else cell_style)
            cat_p = Paragraph(html.escape(result.categories[i].replace("_", " ").title()) if is_anom else "Consistent", cell_style)
            exp_p = Paragraph(html.escape(exp.contextual_explanation), cell_style)

            audit_rows.append([s_id_p, sent_p, status_p, score_p, cat_p, exp_p])

        audit_table = Table(
            audit_rows,
            colWidths=[28, 200, 52, 42, 85, 133],
            repeatRows=1
        )

        t_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
            ("ALIGN", (0, 0), (-1, 0), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ]

        # Highlight anomaly rows
        for row_idx, is_anom in enumerate(result.is_anomaly, start=1):
            if is_anom:
                t_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#FFF5F5")))

        audit_table.setStyle(TableStyle(t_style))
        story.append(audit_table)

        doc.build(story)
        logger.info(f"Generated PDF audit report at: {path}")
        return str(path)
