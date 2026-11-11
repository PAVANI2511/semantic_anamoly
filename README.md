# SEMANTIC SENTINEL
### Context-Aware Semantic Anomaly Detection and Span Localization in Documents

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch-red.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Executive Summary & Research Motivation

**Semantic Sentinel** is a research-oriented NLP system designed to detect and localize semantic anomalies within unstructured text documents. Unlike conventional proofreading tools that operate on surface-level orthography or syntax (spelling and grammar), Semantic Sentinel analyzes **deep contextual coherence** to identify sentences that logically diverge from document semantics.

The system targets five core anomaly categories:
1. **Topic Deviations:** Extraneous subject matter injected into a coherent domain.
2. **Semantic Contradictions:** Factually or logically conflicting statements across sentences.
3. **Contextual Inconsistencies:** Abrupt ruptures in local discursive progression.
4. **Entity Inconsistencies:** Unprecedented named entities or cross-domain agents.
5. **Attribute Inconsistencies:** Physically or logically incompatible attributes assigned to entities.

---

## 2. Proposed Research Architecture: DGSA

The core intelligence of Semantic Sentinel is governed by **DGSA (Dual-Granularity Semantic Anomaly Detection)**, a hybrid framework combining document-level macro context with sliding-window micro context and contradiction verification.

```
                                  [ Document Input (PDF / DOCX / TXT / Paste) ]
                                                        │
                                                        ▼
                                       [ Preprocessing & Segmentation ]
                                        (Offset Tracking & Tokenization)
                                                        │
                                                        ▼
                                       [ Dense Semantic Representation ]
                                   (Sentence-BERT / Pretrained Transformers)
                                                        │
                         ┌──────────────────────────────┼──────────────────────────────┐
                         ▼                              ▼                              ▼
             [ Global Context Analyzer ]    [ Local Context Analyzer ]     [ Contradiction Detector ]
             - Document centroid distance   - Pairwise neighbor window     - Antonym opposition
             - Macro topic clustering       - Bridge anomaly penalty       - Negation polarity shifts
                         │                              │                              │
                         ▼                              ▼                              ▼
                 Global Score G(i)              Local Score L(i)           Contradiction Score C(i)
                         │                              │                              │
                         └──────────────────────────────┼──────────────────────────────┘
                                                        │
                                                        ▼
                                       [ Dual-Level Composite Scorer ]
                                        A(i) = α·G(i) + β·L(i) + γ·C(i)
                                                        │
                                                        ▼
                                       [ Decision Boundary & Filtering ]
                                         (Sensitivity Preset / Threshold)
                                                        │
                         ┌──────────────────────────────┴──────────────────────────────┐
                         ▼                                                             ▼
             [ Anomaly Span Localizer ]                                    [ Explainable AI Engine ]
             - Token-level salience attribution                            - Component contribution breakdown
             - Exact start/end character offsets                           - Contextual narrative & evidence
                         │                                                             │
                         └──────────────────────────────┬──────────────────────────────┘
                                                        │
                                                        ▼
                                 [ Streamlit UI / JSON / CSV / PDF Report Exports ]
```

### Mathematical Formulation

The composite anomaly score $A(i) \in [0, 1]$ for sentence $s_i$ is computed as:

$$A(i) = \alpha \widetilde{G}(i) + \beta \widetilde{L}(i) + \gamma \widetilde{C}(i)$$

Where:
- $\widetilde{G}(i) = \text{Norm}(1 - \cos(\mathbf{e}_i, \mathbf{c}_{doc}))$ evaluates semantic divergence from the global document centroid $\mathbf{c}_{doc} = \frac{1}{N} \sum_{j=1}^N \mathbf{e}_j$.
- $\widetilde{L}(i) = \text{Norm}\left(1 - \frac{1}{|K_i|} \sum_{j \in K_i} \cos(\mathbf{e}_i, \mathbf{e}_j) + \text{BridgePenalty}(i)\right)$ measures micro-incoherence across sliding window $K_i$.
- $\widetilde{C}(i)$ quantifies negation polarity inversions and semantic antonym oppositions against antecedent claims.
- $\alpha, \beta, \gamma$ are configurable weights ($\alpha=0.40, \beta=0.35, \gamma=0.25$ by default) dynamically normalized during ablation studies.

---

## 3. Project Directory Structure

```
semantic_anamoly/
├── app.py                         # Streamlit interactive research dashboard
├── requirements.txt               # Production dependencies
├── README.md                      # Comprehensive documentation
├── .gitignore                     # Git exclusion rules
│
├── config/
│   ├── __init__.py
│   └── settings.py                # Hyperparameters, paths & anomaly categories
│
├── preprocessing/
│   ├── __init__.py
│   ├── document_extractor.py      # PDF, DOCX, TXT parsers with graceful error recovery
│   ├── text_cleaner.py            # Unicode NFKC normalization
│   └── sentence_segmenter.py      # Offset-preserving segmentation & tokenization
│
├── models/
│   ├── __init__.py
│   ├── embedding_model.py         # Sentence-BERT (MiniLM / MPNet) & TF-IDF backends
│   ├── anomaly_detector.py        # Isolation Forest, One-Class SVM baselines
│   ├── contradiction_detector.py  # Polarity & antonym opposition verification
│   ├── span_localizer.py          # Token-level salience attribution
│   └── dgsa_model.py              # End-to-end DGSA orchestration pipeline
│
├── analysis/
│   ├── __init__.py
│   ├── global_context.py          # Document centroid divergence
│   ├── local_context.py           # Neighborhood coherence & bridge anomaly detection
│   ├── anomaly_scoring.py         # Composite score calculation & category inference
│   └── explanation_engine.py      # Evidence-grounded XAI reasoning reports
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, IoU
│   ├── baselines.py               # 5-way comparative model harness
│   ├── experiments.py             # Benchmark suites, ablations, threshold calibration
│   └── visualization.py           # Interactive Plotly figures & diagnostic heatmaps
│
├── reports/
│   ├── __init__.py
│   └── export_results.py          # JSON, CSV, and executive HTML/PDF report generators
│
├── data/
│   ├── sample_documents/          # 5 curated demonstration documents
│   └── datasets/                  # Synthetic benchmark datasets (JSON & CSV)
│
└── tests/
    ├── test_preprocessing.py      # Offsets, token spans, and file parsing tests
    ├── test_anomaly_detection.py  # DGSA scoring, ablation, and pipeline tests
    └── test_evaluation.py         # Metrics, leakage-free splitting, and baselines
```

---

## 4. Installation & Quick Start

### Step 1: Environment Setup
Ensure Python 3.11+ is installed. Create and activate a virtual environment:
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Unit Tests
Verify that all 16 test suites pass:
```powershell
python -m unittest discover tests
```

### Step 4: Launch Web Dashboard
```powershell
streamlit run app.py
```

---

## 5. Algorithmic Baselines & Comparative Evaluation

Semantic Sentinel implements a rigorous 5-way baseline benchmark across controlled evaluation datasets:

| Model Architecture | Anomaly Paradigm | F1-Score | Precision | Recall | Latency / Sent |
|---|---|---|---|---|---|
| **TF-IDF + Isolation Forest** | Unsupervised Bag-of-Words | 0.4000 | 0.5000 | 0.3333 | 0.12 ms |
| **S-BERT + Isolation Forest** | Unsupervised Dense Density | 0.5714 | 0.6667 | 0.5000 | 2.45 ms |
| **S-BERT + One-Class SVM** | Unsupervised Kernel Boundary | 0.6000 | 0.6000 | 0.6000 | 2.10 ms |
| **Supervised Linear Baseline** | Calibrated Logistic Regression | 0.6667 | 0.7500 | 0.6000 | 1.80 ms |
| **Proposed DGSA Framework** | Multi-Granular Context + Contradiction | **0.8889** | **0.8889** | **0.8889** | 3.15 ms |

*Note: Results obtained on curated controlled datasets; individual results vary across unstructured domains.*

---

## 6. Demonstration Documents

Sample test documents are included under `data/sample_documents/`:
1. `01_normal_climate_science.txt`: Fully coherent scientific text (No anomalies).
2. `02_topic_deviation_quantum_cooking.txt`: Quantum computing text with an intrusive cake recipe sentence.
3. `03_contradiction_financial_report.txt`: Earnings statement with direct revenue contradiction.
4. `04_entity_inconsistency_medical_case.txt`: Cardiology case report with an unexpected bank robbery detective sentence.
5. `05_attribute_inconsistency_astronomy.txt`: Mars surface report with tropical rain forests.

---

## 7. Limitations & Research Roadmap

- **Coreference Chains:** Currently uses capitalization patterns; future iterations can integrate end-to-end neural coreference resolution.
- **Subtle Numerical Contradictions:** Explicit antonym pairs are handled reliably; fine-grained numeric reasoning requires a specialized tabular-NLI model.
- **Multilingual Support:** Sentence-BERT models can be upgraded to `paraphrase-multilingual-mpnet-base-v2` for cross-lingual document analysis.

---

## 8. License

This project is licensed under the MIT License.
