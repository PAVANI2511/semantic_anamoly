"""
SEMANTIC SENTINEL - Context-Aware Semantic Anomaly Detection & Span Localization
Research & Production Web Application powered by the DGSA Framework.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path
import json

# Setup page configuration
st.set_page_config(
    page_title="Semantic Sentinel | DGSA Document Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern research aesthetic
st.markdown("""
<style>
    .main-header {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1e293b;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-title {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .badge-topic { background-color: #fee2e2; color: #b91c1c; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .badge-contra { background-color: #f3e8ff; color: #7e22ce; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .badge-entity { background-color: #ffedd5; color: #c2410c; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .badge-context { background-color: #fef9c3; color: #a16207; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .badge-normal { background-color: #dcfce7; color: #15803d; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .span-highlight { background-color: #fde047; padding: 1px 4px; border-radius: 3px; font-weight: bold; border-bottom: 2px solid #ca8a04; }
</style>
""", unsafe_allow_html=True)

from config.settings import config, DGSAParameters, AnomalyCategories
from preprocessing.document_extractor import DocumentExtractor, ExtractedDocument
from models.embedding_model import TransformerEmbeddingModel, TFIDFEmbeddingModel, BaseEmbeddingModel
from models.dgsa_model import DGSAModel, DGSAResult
from evaluation.visualization import (
    create_sentence_score_chart,
    create_signal_decomposition_chart,
    create_category_pie_chart,
    create_confusion_matrix_heatmap,
    create_baseline_comparison_chart
)
from evaluation.experiments import ExperimentHarness
from evaluation.metrics import compute_sentence_metrics
from data.dataset_loader import DatasetLoader
from reports.export_results import ExportManager


@st.cache_resource(show_spinner=False)
def load_dgsa_system(backend_type: str = "transformer", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Loads and caches model instances across sessions."""
    try:
        if backend_type == "transformer":
            emb_model = TransformerEmbeddingModel(model_name=model_name)
        else:
            emb_model = TFIDFEmbeddingModel()
        return DGSAModel(embedding_model=emb_model)
    except Exception as e:
        # Fallback to TF-IDF if transformer loading fails
        st.warning(f"Could not load {model_name} ({e}). Falling back to TF-IDF Vectorizer backend.")
        emb_model = TFIDFEmbeddingModel()
        return DGSAModel(embedding_model=emb_model)


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.markdown("### 🛡️ **SEMANTIC SENTINEL**")
    st.caption("DGSA – Dual-Granularity Semantic Anomaly Detection")
    st.divider()

    st.markdown("#### **Navigation**")
    nav_option = st.radio(
        "Select View",
        [
            "📄 Document Analysis & Audit",
            "📊 Visual Analytics & Diagnostics",
            "📑 Detailed Sentence Audit",
            "🔬 Research Benchmarks & Baselines",
            "ℹ️ Architecture & Methodology"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("#### **Model Settings**")
    model_choice = st.selectbox(
        "Embedding Backend",
        ["Sentence-BERT (MiniLM-L6-v2)", "Sentence-BERT (MPNet-Base-v2)", "TF-IDF Vectorizer (Baseline)"],
        index=0
    )
    backend_key = "tfidf" if "TF-IDF" in model_choice else "transformer"
    model_id = "sentence-transformers/all-mpnet-base-v2" if "MPNet" in model_choice else "sentence-transformers/all-MiniLM-L6-v2"

    st.markdown("#### **DGSA Hyperparameters**")
    sensitivity = st.select_slider(
        "Detection Sensitivity",
        options=["Low (Conservative)", "Medium (Balanced)", "High (Aggressive)"],
        value="Medium (Balanced)"
    )
    preset_threshold = {"Low (Conservative)": 0.65, "Medium (Balanced)": 0.50, "High (Aggressive)": 0.35}[sensitivity]

    custom_th = st.slider("Anomaly Threshold $T$", min_value=0.10, max_value=0.90, value=preset_threshold, step=0.05)

    with st.expander("Signal Weights Adjustment", expanded=False):
        w_global = st.slider("Global Weight α (Topic)", 0.0, 1.0, 0.40, 0.05)
        w_local = st.slider("Local Weight β (Context)", 0.0, 1.0, 0.35, 0.05)
        w_contra = st.slider("Contradiction Weight γ", 0.0, 1.0, 0.25, 0.05)
        st.caption(f"Normalized sum: {w_global + w_local + w_contra:.2f}")


# Initialize model
with st.spinner("Initializing Semantic Intelligence backend..."):
    dgsa_engine = load_dgsa_system(backend_type=backend_key, model_name=model_id)

# Global Session State
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "document_name" not in st.session_state:
    st.session_state["document_name"] = "Untitled"


# =========================================================================
# PAGE 1: DOCUMENT ANALYSIS & AUDIT
# =========================================================================
if nav_option == "📄 Document Analysis & Audit":
    st.markdown('<div class="main-header">Context-Aware Document Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Identify topic deviations, contradictions, and contextual fractures using dual-granularity semantics.</div>', unsafe_allow_html=True)

    col_input, col_preset = st.columns([3, 1])

    with col_preset:
        st.markdown("#### 📂 **Sample Documents**")
        sample_choice = st.selectbox(
            "Load Research Demonstration",
            [
                "None (Custom)",
                "01 - Coherent Climate Science (Normal)",
                "02 - Quantum Computing + Baking (Topic Deviation)",
                "03 - Financial Earnings (Contradiction)",
                "04 - Medical Emergency (Entity Inconsistency)",
                "05 - Mars Rover Mission (Attribute Inconsistency)"
            ]
        )

    # Sample file mapping
    sample_files = {
        "01 - Coherent Climate Science (Normal)": "data/sample_documents/01_normal_climate_science.txt",
        "02 - Quantum Computing + Baking (Topic Deviation)": "data/sample_documents/02_topic_deviation_quantum_cooking.txt",
        "03 - Financial Earnings (Contradiction)": "data/sample_documents/03_contradiction_financial_report.txt",
        "04 - Medical Emergency (Entity Inconsistency)": "data/sample_documents/04_entity_inconsistency_medical_case.txt",
        "05 - Mars Rover Mission (Attribute Inconsistency)": "data/sample_documents/05_attribute_inconsistency_astronomy.txt"
    }

    preset_text = ""
    if sample_choice in sample_files:
        try:
            with open(sample_files[sample_choice], "r", encoding="utf-8") as f:
                preset_text = f.read()
        except Exception:
            preset_text = ""

    with col_input:
        tab_paste, tab_upload = st.tabs(["✍️ Paste Text", "📁 Upload File (PDF, DOCX, TXT)"])

        input_text = ""
        doc_source = "Pasted Text"

        with tab_paste:
            input_text = st.text_area(
                "Document Text",
                value=preset_text,
                height=220,
                placeholder="Paste document text here to evaluate contextual coherence..."
            )

        with tab_upload:
            uploaded_file = st.file_uploader("Upload Document", type=["txt", "pdf", "docx"])
            if uploaded_file is not None:
                doc_source = uploaded_file.name
                bytes_data = uploaded_file.read()
                extracted = DocumentExtractor.extract_from_bytes(bytes_data, uploaded_file.name)
                if extracted.is_valid:
                    input_text = extracted.raw_text
                    st.success(f"Successfully extracted {len(extracted.raw_text)} characters from '{uploaded_file.name}'")
                else:
                    st.error(f"Extraction failed: {extracted.error_message}")

    run_btn = st.button("🚀 Analyze Document Semantics", type="primary", use_container_width=True)

    if run_btn:
        if not input_text.strip():
            st.error("Please provide document text before running analysis.")
        else:
            with st.spinner("Executing DGSA Semantic Pipeline (Global + Local + Contradiction)..."):
                weights = {"alpha": w_global, "beta": w_local, "gamma": w_contra}
                result = dgsa_engine.analyze_document(
                    text=input_text,
                    custom_threshold=custom_th,
                    custom_weights=weights
                )
                st.session_state["analysis_result"] = result
                st.session_state["document_name"] = doc_source

    # Render results if available
    res: DGSAResult = st.session_state.get("analysis_result")
    if res is not None:
        st.divider()
        st.markdown(f"### 📋 **Executive Assessment: {res.document_assessment}**")

        # Top metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Total Sentences</div><div class="metric-value">{res.total_sentences}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Anomalies Flagged</div><div class="metric-value" style="color: #e11d48;">{res.num_anomalous}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Anomaly Rate</div><div class="metric-value">{res.anomaly_percentage}%</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Inference Time</div><div class="metric-value" style="color: #0284c7;">{res.inference_time_ms} ms</div></div>', unsafe_allow_html=True)

        st.markdown("#### **Interactive Document Preview with Semantic Highlights**")
        st.caption("Normal sentences appear in clean dark text. Flagged anomalies have color-coded category badges and span-level evidence.")

        # Document highlighter view
        doc_html_parts = []
        for i, s in enumerate(res.sentences):
            is_anom = res.is_anomaly[i]
            score = res.anomaly_scores[i]
            cat = res.categories[i]
            exp = res.explanations[i]

            badge_html = ""
            bg_style = ""
            sentence_content = s.text

            if is_anom:
                if cat == "topic_deviation":
                    badge_html = f'<span class="badge-topic">Topic Deviation (Score: {score:.2f})</span>'
                    bg_style = "background-color: #fff1f2; border-left: 4px solid #e11d48; padding: 6px 10px; margin: 4px 0; border-radius: 4px;"
                elif cat == "contradiction":
                    badge_html = f'<span class="badge-contra">Contradiction (Score: {score:.2f})</span>'
                    bg_style = "background-color: #faf5ff; border-left: 4px solid #9333ea; padding: 6px 10px; margin: 4px 0; border-radius: 4px;"
                elif cat in {"entity_inconsistency", "attribute_inconsistency"}:
                    badge_html = f'<span class="badge-entity">{cat.replace("_", " ").title()} (Score: {score:.2f})</span>'
                    bg_style = "background-color: #fff7ed; border-left: 4px solid #ea580c; padding: 6px 10px; margin: 4px 0; border-radius: 4px;"
                else:
                    badge_html = f'<span class="badge-context">Context Fracture (Score: {score:.2f})</span>'
                    bg_style = "background-color: #fefce8; border-left: 4px solid #ca8a04; padding: 6px 10px; margin: 4px 0; border-radius: 4px;"

                # Highlight spans if localized
                if exp.localized_spans:
                    for sp in exp.localized_spans:
                        span_txt = sp["text"]
                        sentence_content = sentence_content.replace(span_txt, f'<span class="span-highlight">{span_txt}</span>')

                doc_html_parts.append(f'<div style="{bg_style}"><b>#{i+1}</b> {badge_html}<br>{sentence_content}</div>')
            else:
                doc_html_parts.append(f'<div style="padding: 4px 0; color: #334155;"><b>#{i+1}</b> {sentence_content}</div>')

        st.markdown(
            f'<div style="background: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 18px; max-height: 420px; overflow-y: auto; line-height: 1.6;">{"".join(doc_html_parts)}</div>',
            unsafe_allow_html=True
        )

        # Quick Export Options
        st.markdown("#### 📥 **Export Results**")
        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
        with exp_col1:
            json_str = json.dumps({
                "document": st.session_state["document_name"],
                "assessment": res.document_assessment,
                "anomalies": [
                    {"sentence_id": s.sentence_id, "score": res.anomaly_scores[i], "category": res.categories[i], "text": s.text}
                    for i, s in enumerate(res.sentences) if res.is_anomaly[i]
                ]
            }, indent=2)
            st.download_button("Download JSON Audit", data=json_str, file_name="semantic_sentinel_audit.json", mime="application/json")

        with exp_col2:
            csv_rows = [
                {"Sentence": s.sentence_id + 1, "Text": s.text, "Anomaly": res.is_anomaly[i], "Score": res.anomaly_scores[i], "Category": res.categories[i]}
                for i, s in enumerate(res.sentences)
            ]
            csv_df = pd.DataFrame(csv_rows)
            st.download_button("Download CSV Audit", data=csv_df.to_csv(index=False), file_name="semantic_sentinel_audit.csv", mime="text/csv")

        with exp_col3:
            html_rep = ExportManager.export_to_html_report(res, "outputs/reports/current_report.html")
            with open(html_rep, "r", encoding="utf-8") as f:
                st.download_button("Download Executive HTML", data=f.read(), file_name="semantic_sentinel_report.html", mime="text/html")

        with exp_col4:
            pdf_rep = ExportManager.export_to_pdf(res, "outputs/reports/current_report.pdf")
            with open(pdf_rep, "rb") as f:
                st.download_button("Download PDF Audit (ReportLab)", data=f.read(), file_name="semantic_sentinel_report.pdf", mime="application/pdf")


# =========================================================================
# PAGE 2: VISUAL ANALYTICS & DIAGNOSTICS
# =========================================================================
elif nav_option == "📊 Visual Analytics & Diagnostics":
    st.markdown('<div class="main-header">Visual Analytics & Diagnostic Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect sentence trajectories, component signal breakdowns, and category distributions.</div>', unsafe_allow_html=True)

    res: DGSAResult = st.session_state.get("analysis_result")
    if res is None:
        st.info("No active analysis found. Please run a document analysis on the 'Document Analysis & Audit' page first.")
    else:
        # Chart 1: Sentence Score Trajectory
        fig_traj = create_sentence_score_chart(
            res.anomaly_scores,
            res.threshold_used,
            res.categories,
            res.is_anomaly
        )
        st.plotly_chart(fig_traj, use_container_width=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            # Chart 2: Signal Decomposition
            fig_decomp = create_signal_decomposition_chart(
                res.global_scores,
                res.local_scores,
                res.contradiction_scores
            )
            st.plotly_chart(fig_decomp, use_container_width=True)

        with col_c2:
            # Chart 3: Category Distribution
            fig_cat = create_category_pie_chart(res.categories, res.is_anomaly)
            st.plotly_chart(fig_cat, use_container_width=True)


# =========================================================================
# PAGE 3: DETAILED SENTENCE AUDIT
# =========================================================================
elif nav_option == "📑 Detailed Sentence Audit":
    st.markdown('<div class="main-header">Explainable AI: Sentence-Level Audit Trail</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detailed model evidence, mathematical signal contributions, and span localizations.</div>', unsafe_allow_html=True)

    res: DGSAResult = st.session_state.get("analysis_result")
    if res is None:
        st.info("No active analysis found. Please run a document analysis first.")
    else:
        anomalies_only = st.checkbox("Show only flagged anomalies", value=True)

        for i, s in enumerate(res.sentences):
            is_anom = res.is_anomaly[i]
            if anomalies_only and not is_anom:
                continue

            exp = res.explanations[i]
            score = res.anomaly_scores[i]
            cat_label = res.categories[i].replace("_", " ").title()

            status_icon = "🚨" if is_anom else "✅"
            header_text = f"{status_icon} Sentence #{s.sentence_id + 1} | Score: {score:.3f} | {cat_label}"

            with st.expander(header_text, expanded=is_anom):
                st.markdown(f"**Original Sentence:** *\"{s.text}\"*")
                st.markdown(f"**Contextual Explanation:** {exp.contextual_explanation}")
                st.markdown(f"**Primary Contributing Signal:** `{exp.primary_factor}` (Evidence Certainty: `{exp.evidence_certainty}`)")

                c_sig1, c_sig2, c_sig3 = st.columns(3)
                c_sig1.metric("Global Deviation G(i)", f"{res.global_scores[i]:.3f}", f"Eff: {exp.signals_breakdown['effective_global_contrib']:.3f}")
                c_sig2.metric("Local Inconsistency L(i)", f"{res.local_scores[i]:.3f}", f"Eff: {exp.signals_breakdown['effective_local_contrib']:.3f}")
                c_sig3.metric("Contradiction Score C(i)", f"{res.contradiction_scores[i]:.3f}", f"Eff: {exp.signals_breakdown['effective_contradiction_contrib']:.3f}")

                if exp.localized_spans:
                    st.markdown("**Localized Anomalous Spans (Token-Level Evidence):**")
                    for sp in exp.localized_spans:
                        st.markdown(f"- 🔲 **`{sp['text']}`** [Char: {sp['start_char']}-{sp['end_char']}] — *Salience: {sp['salience']:.2f}* ({sp['reason']})")
                else:
                    st.caption("ℹ️ *Localization is holistic sentence-level; no isolated word token exceeded span salience threshold.*")

                if exp.surrounding_context:
                    st.markdown("**Surrounding Sentences:**")
                    for ctx in exp.surrounding_context:
                        st.caption(f"Sentence #{ctx['sentence_id'] + 1} ({ctx['relative_pos']}): {ctx['text']}")


# =========================================================================
# PAGE 4: RESEARCH BENCHMARKS & BASELINES
# =========================================================================
elif nav_option == "🔬 Research Benchmarks & Baselines":
    st.markdown('<div class="main-header">Scientific Evaluation & Baseline Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous zero-leakage evaluation against Isolation Forest, One-Class SVM, and Logistic Regression.</div>', unsafe_allow_html=True)

    benchmark_dir = Path("outputs/benchmarks")
    bench_csv = benchmark_dir / "benchmark_results.csv"
    abl_csv = benchmark_dir / "ablation_results.csv"
    span_csv = benchmark_dir / "span_evaluation_results.csv"

    st.markdown("### 🏆 **Empirical Research Benchmark (Zero-Leakage Document Split)**")
    st.caption("Documents partitioned strictly across 50% Train, 25% Validation (threshold tuning), and 25% Test (held-out evaluation). Zero sentence leakage across partitions.")

    # Re-run button
    run_rigorous_btn = st.button("⚡ Re-Run Zero-Leakage Benchmark Across All 6 Models", type="primary")

    if run_rigorous_btn:
        with st.spinner("Executing rigorous Train/Val/Test benchmark with DeBERTa NLI and Baselines..."):
            data_file = Path("data/datasets/research_benchmark.json")
            if not data_file.exists():
                data_file = Path("data/datasets/synthetic_benchmark.json")
            all_docs = DatasetLoader.load_from_json(data_file)
            tr, val, ts = DatasetLoader.split_train_val_test(all_docs, 0.5, 0.25, 0.25, random_seed=42)
            harness = ExperimentHarness(dgsa_engine)
            res_dict = harness.run_rigorous_benchmark(tr, val, ts)
            st.session_state["benchmark_df"] = res_dict["benchmark_results"]
            st.session_state["ablation_df"] = res_dict["ablation_results"]
            st.session_state["span_metrics"] = res_dict["span_metrics"]

    # Load from disk if not in session state
    if "benchmark_df" not in st.session_state and bench_csv.exists():
        st.session_state["benchmark_df"] = pd.read_csv(bench_csv)
    if "ablation_df" not in st.session_state and abl_csv.exists():
        st.session_state["ablation_df"] = pd.read_csv(abl_csv)
    if "span_metrics_df" not in st.session_state and span_csv.exists():
        st.session_state["span_metrics_df"] = pd.read_csv(span_csv)

    if "benchmark_df" in st.session_state:
        df = st.session_state["benchmark_df"]
        st.markdown("#### **1. Comparative Benchmark Performance (Held-Out Test Set)**")
        st.dataframe(df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Latency (ms)"]], use_container_width=True)

        fig_bench = create_baseline_comparison_chart(df)
        st.plotly_chart(fig_bench, use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### **2. Span Localization Metrics (Fine-Grained Attribution)**")
            if "span_metrics_df" in st.session_state:
                sp_df = st.session_state["span_metrics_df"]
                st.dataframe(sp_df, use_container_width=True)
            elif "span_metrics" in st.session_state:
                sp_obj = st.session_state["span_metrics"]
                st.dataframe(pd.DataFrame([sp_obj.to_dict()]), use_container_width=True)

        with col_right:
            st.markdown("#### **3. DGSA Confusion Matrix (Test Split)**")
            dgsa_matches = df[df["Model"].str.contains("DGSA Full|DGSA Framework")]
            if not dgsa_matches.empty:
                dgsa_row = dgsa_matches.iloc[0]
                cm = [[int(dgsa_row["TN"]), int(dgsa_row["FP"])], [int(dgsa_row["FN"]), int(dgsa_row["TP"])]]
                fig_cm = create_confusion_matrix_heatmap(cm)
                st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("#### **4. Full Signal Ablation Study (7 Component Configurations)**")
        st.caption("Evaluating empirical impact of individual contextual signals G(i), L(i), C(i) on test documents.")
        if "ablation_df" in st.session_state:
            st.dataframe(st.session_state["ablation_df"][["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]], use_container_width=True)


# =========================================================================
# PAGE 5: ARCHITECTURE & METHODOLOGY
# =========================================================================
elif nav_option == "ℹ️ Architecture & Methodology":
    st.markdown('<div class="main-header">DGSA Research Framework</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Dual-Granularity Semantic Anomaly Detection and Span Localization in Documents</div>', unsafe_allow_html=True)

    st.markdown("""
    ### **Mathematical Formulation**
    The DGSA framework models document anomaly detection as a multi-granular contextual problem:
    
    $$A(i) = \\alpha G(i) + \\beta L(i) + \\gamma C(i)$$
    
    Where:
    - **$G(i)$ — Global Semantic Deviation:** Measures the cosine divergence of sentence $s_i$ from the document-level semantic centroid $\\mathbf{c}_{doc} = \\frac{1}{N} \\sum_{j=1}^N \\mathbf{e}_j$.
    - **$L(i)$ — Local Contextual Inconsistency:** Measures pairwise coherence within a sliding window $k \\in [-2, +2]$ and penalizes bridge anomalies where surrounding sentences share high continuity while $s_i$ interrupts the local semantic manifold.
    - **$C(i)$ — Contradiction Score:** Evaluates negation shifts, opposing truth values, and antonym oppositions between $s_i$ and preceding contextual claims.
    - **$\\alpha, \\beta, \\gamma$:** Configurable weighting parameters normalized such that $\\alpha + \\beta + \\gamma = 1.0$.

    ---
    ### **Architectural Pipeline**
    1. **Document Ingestion:** Multi-format parser for PDF, DOCX, and TXT with paragraph boundary preservation.
    2. **Sentence Segmentation & Indexing:** Character offset tracking $(s_{start}, s_{end})$ and tokenization.
    3. **Semantic Representation:** Configurable Sentence Transformers (`sentence-transformers/all-mpnet-base-v2` or `all-MiniLM-L6-v2`) generating dense normalized vectors $\\mathbf{e}_i \\in \\mathbb{R}^D$.
    4. **Dual Context Analysis:** Concurrent computation of global document centroid divergence and local sliding-window neighbor coherence.
    5. **Contradiction Verification:** Semantic polarity and opposition detection against antecedent sentences.
    6. **Span Localization:** Token-level salience attribution isolating exact phrases contributing to topic shifts or contradictions.
    7. **Explainable AI Engine:** Transparent reasoning reports providing mathematical component contributions and human-interpretable contextual evidence.
    """)

