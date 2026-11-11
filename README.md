# SEMANTIC SENTINEL
### Context-Aware Semantic Anomaly Detection and Span Localization in Documents

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch-red.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 What Does This Application Do?

**Semantic Sentinel** is an intelligent document analysis and audit platform that detects **semantic anomalies, logical contradictions, and topic disruptions** in written documents. 

Traditional proofreading software (like spellcheckers and grammar linters) only checks whether words are spelled correctly or follow grammatical rules. They fail when text is grammatically flawless but logically nonsensical, factually contradictory, or completely off-topic.

**Semantic Sentinel solves this problem** by analyzing the *meaning* of each sentence in relation to both:
1. **The overall document context** (Macro/Global Topic), and
2. **Its surrounding neighbor sentences** (Micro/Local Context).

---

### 🚀 Key Capabilities of the Present Application

| Capability | What It Does |
|---|---|
| 📄 **Document Ingestion** | Upload **PDF**, **DOCX**, **TXT** files, or paste raw text directly. Preserves paragraphs and sentence boundaries. |
| 🧠 **Context-Aware Inference** | Uses real pretrained **Sentence Transformers** (`sentence-transformers/all-MiniLM-L6-v2` and `all-mpnet-base-v2`) cached locally for offline execution. |
| 🎯 **Span-Level Localization** | Doesn't just flag the sentence—it **pinpoints and highlights the specific words or phrases** causing the anomaly with exact start/end character offsets. |
| ⚡ **Contradiction Detection** | Automatically flags conflicting statements across sentences (e.g., revenue increasing by 38% vs. revenue decreasing by 40%). |
| 💡 **Explainable AI (XAI)** | Provides a transparent explanation for every flagged sentence, detailing the exact mathematical signals ($\alpha G + \beta L + \gamma C$) and surrounding context. |
| 📊 **Interactive Analytics** | Interactive Plotly charts displaying sentence-by-sentence anomaly score trajectories, component signal breakdowns, and category distributions. |
| 📥 **Export Reports** | Download full audit results as structured **JSON**, tabular **CSV**, or an executive printable **HTML/PDF** report. |
| 🔬 **Research Benchmarks** | A built-in scientific evaluation lab that compares **DGSA** against 4 baselines (**TF-IDF + Isolation Forest**, **S-BERT + Isolation Forest**, **S-BERT + One-Class SVM**, and **Supervised Classifiers**) with confusion matrices and ablation studies. |

---

## 🔍 Types of Semantic Anomalies Detected

Semantic Sentinel classifies anomalies into 5 distinct categories:

### 1. 🎯 Topic Deviation
- **What happens:** An unrelated topic is injected into an otherwise coherent document.
- **Example:** A research paper discussing *quantum computing qubits and superposition* suddenly contains:
  > *"To bake an artisan chocolate cake, preheat your convection oven to 350 degrees..."*
- **Action:** Flagged as a high global deviation ($G(i)$) with highlighted culinary terms.

### 2. ⚡ Semantic Contradiction
- **What happens:** A sentence logically contradicts a claim made earlier in the document.
- **Example:** A corporate earnings report states *"Total enterprise revenues increased by 38 percent..."* in paragraph 1, but later states:
  > *"The company experienced total financial failure with overall enterprise revenues that decreased drastically by 40 percent during the exact same period."*
- **Action:** Flagged as a Contradiction ($C(i)$) with polarity opposition detection (`increased` vs `decreased`).

### 3. 🧩 Contextual Inconsistency (Bridge Anomaly)
- **What happens:** An abrupt fracture in local narrative or logical progression where surrounding sentences share high continuity, but the middle sentence is completely alien.
- **Example:** A sentence that abruptly breaks the flow between medical diagnosis and catheterization treatment.
- **Action:** Penalized by the sliding-window neighbor coherence analyzer ($L(i)$).

### 4. 👤 Entity Inconsistency
- **What happens:** An external or unintroduced named entity appears out of domain.
- **Example:** An acute cardiology emergency case report contains:
  > *"Detective Miller reviewed the surveillance footage from the bank robbery to identify the armed getaway vehicle."*
- **Action:** Flagged as an entity novelty anomaly with character span attribution.

### 5. 🪐 Attribute Inconsistency
- **What happens:** Physically or semantically incompatible attributes are assigned to an entity or environment.
- **Example:** An astronomical survey of the arid, frozen Martian surface contains:
  > *"The rover team celebrated as warm liquid rainwater fell gently upon the lush green Martian forests under the tropical blue sky."*
- **Action:** Flagged as contextual inconsistency with localized attribute spans highlighted.

---

## 🖥️ Application Features & User Interface

The application runs as a modern, responsive **Streamlit Web Dashboard** with 5 primary sections:

### 1. 📄 Document Analysis & Audit
- Paste text or upload PDF/DOCX/TXT files.
- Choose from 5 pre-loaded research demonstration documents.
- Set anomaly sensitivity (**Low/Conservative**, **Medium/Balanced**, **High/Aggressive**) or adjust the threshold slider ($T \in [0.10, 0.90]$).
- View executive summary cards (Total Sentences, Anomalies Flagged, Anomaly Percentage, Latency).
- Interactive document view with **color-coded badges** and **yellow-highlighted span tokens**.
- Instant export buttons for **JSON**, **CSV**, and **HTML Report**.

### 2. 📊 Visual Analytics & Diagnostics
- **Score Trajectory Timeline:** Interactive Plotly scatter/line chart showing every sentence's score $A(i)$ against the decision threshold line.
- **Multi-Granular Signal Breakdown:** Grouped bar chart decomposing the global deviation $G(i)$, local inconsistency $L(i)$, and contradiction score $C(i)$ for every sentence.
- **Category Distribution Chart:** Interactive donut chart illustrating the proportion of detected anomaly types.

### 3. 📑 Detailed Sentence Audit (Explainable AI)
- Expandable audit trail for every sentence.
- Factual contextual explanation (e.g., *"Model detected direct semantic conflict with sentence #2: Opposing semantic terms detected ('increas' vs 'decreas')"*).
- Exact mathematical signal values and certainty levels.
- Localized token spans with exact start/end character offsets.
- Surrounding contextual sentences for immediate human verification.

### 4. 🔬 Research Benchmarks & Baselines
- One-click benchmarking on the controlled evaluation dataset.
- Comparative metric table: **Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Inference Latency**.
- Visual baseline comparison bar chart across all 5 models.
- Confusion Matrix Heatmap (True Positives, False Positives, True Negatives, False Negatives).
- **Ablation Study:** Automated evaluation showing model performance when disabling Global Context, Local Context, or Contradiction signals.

### 5. ℹ️ Architecture & Methodology
- Complete mathematical documentation of the **DGSA** formulation.
- Definitions of all hyperparameters and signal weights.

---

## 🧠 How It Works: The DGSA Research Architecture

The system operates via **DGSA (Dual-Granularity Semantic Anomaly Detection)**:

```
[ Document Input (PDF / DOCX / TXT / Paste) ]
                       │
                       ▼
      [ Preprocessing & Sentence Segmentation ]
      (Character Offset Tracking & Tokenization)
                       │
                       ▼
       [ Dense Semantic Representation ]
    (Sentence-BERT: all-MiniLM-L6-v2 / all-mpnet-base-v2)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
[ Global Context ] [ Local Context ] [ Contradiction ]
- Document centroid - Sliding window - Antonym opposition
- Macro deviation   - Bridge anomaly - Polarity inversion
        │              │              │
        ▼              ▼              ▼
    Score G(i)     Score L(i)     Score C(i)
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
        [ Composite Scoring Formulation ]
         A(i) = α·G(i) + β·L(i) + γ·C(i)
                       │
                       ▼
       [ Decision Thresholding & Filtering ]
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
[ Span Localization ]       [ Explainable AI Engine ]
(Token Saliency & Offsets)  (Signal Evidence & Narrative)
        │                             │
        └──────────────┬──────────────┘
                       │
                       ▼
 [ Streamlit Dashboard / JSON / CSV / HTML Reports ]
```

### Mathematical Formulation
$$A(i) = \alpha \widetilde{G}(i) + \beta \widetilde{L}(i) + \gamma \widetilde{C}(i)$$

Where:
- $\widetilde{G}(i) = \text{clip}(1 - \cos(\mathbf{e}_i, \mathbf{c}_{doc}), 0.0, 1.0)$ evaluates semantic divergence from the global centroid $\mathbf{c}_{doc} = \frac{1}{N} \sum_{j=1}^N \mathbf{e}_j$.
- $\widetilde{L}(i) = \text{clip}\left(1 - \frac{1}{|K_i|} \sum_{j \in K_i} \cos(\mathbf{e}_i, \mathbf{e}_j) + \text{BridgePenalty}(i) + \text{EntityNovelty}(i), 0.0, 1.0\right)$ evaluates local sliding-window incoherence.
- $\widetilde{C}(i)$ quantifies polarity inversions and antonym oppositions against antecedent claims.
- Default weights: $\alpha=0.40, \beta=0.35, \gamma=0.25$ (sum to $1.0$).

---

## 📂 Project Structure

```
semantic_anamoly/
├── app.py                         # Interactive Streamlit dashboard
├── requirements.txt               # Project dependencies
├── README.md                      # Documentation
├── .gitignore                     # Git rules
│
├── config/
│   ├── __init__.py
│   └── settings.py                # Hyperparameters & model configurations
│
├── preprocessing/
│   ├── __init__.py
│   ├── document_extractor.py      # PDF, DOCX, TXT extractors
│   ├── text_cleaner.py            # Text normalization
│   └── sentence_segmenter.py      # Offset-preserving segmentation
│
├── models/
│   ├── __init__.py
│   ├── embedding_model.py         # Sentence-BERT & TF-IDF backends
│   ├── anomaly_detector.py        # Isolation Forest & One-Class SVM
│   ├── contradiction_detector.py  # Antonym & polarity opposition
│   ├── span_localizer.py          # Token-level phrase attribution
│   └── dgsa_model.py              # End-to-end DGSA pipeline
│
├── analysis/
│   ├── __init__.py
│   ├── global_context.py          # Macro document centroid divergence
│   ├── local_context.py           # Neighborhood coherence & bridge penalty
│   ├── anomaly_scoring.py         # Dual-level scoring & category inference
│   └── explanation_engine.py      # Evidence-grounded XAI reports
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # Classification & IoU span metrics
│   ├── baselines.py               # 5 comparison baseline models
│   ├── experiments.py             # Benchmarks, ablations & sweeps
│   └── visualization.py           # Interactive Plotly figures
│
├── reports/
│   ├── __init__.py
│   └── export_results.py          # JSON, CSV, and HTML export generators
│
├── data/
│   ├── sample_documents/          # 5 curated test documents
│   └── datasets/                  # Synthetic benchmark datasets
│
└── tests/
    ├── test_preprocessing.py      # Offset & token extraction tests
    ├── test_anomaly_detection.py  # Scoring & DGSA pipeline tests
    └── test_evaluation.py         # Metrics & leakage-free split tests
```

---

## ⚡ How to Run the Application

All dependencies, tokenizers, and pretrained models are **globally installed and cached offline**. You do not need to install anything again.

### 1. Start the Streamlit Web Application
Run the following command in PowerShell / Terminal:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run the Unit Test Suite
Verify that all 16 unit tests pass:
```powershell
python -m unittest discover tests
```

### 3. Programmatic Python Inference
```python
from models.embedding_model import TransformerEmbeddingModel
from models.dgsa_model import DGSAModel

# Initialize DGSA engine with cached Sentence-BERT
embedding_model = TransformerEmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
dgsa = DGSAModel(embedding_model=embedding_model)

# Analyze any document text
result = dgsa.analyze_document("Your text document goes here...")

print(f"Assessment: {result.document_assessment}")
print(f"Anomalies: {result.num_anomalous} of {result.total_sentences}")
```

---

## 📊 Experimental Benchmark Results

Evaluated on the controlled multi-category benchmark dataset:

| Model Architecture | Anomaly Paradigm | F1-Score | Precision | Recall | Latency / Sent |
|---|---|:---:|:---:|:---:|:---:|
| **TF-IDF + Isolation Forest** | Unsupervised Bag-of-Words | 0.2222 | 0.2000 | 0.2500 | 0.12 ms |
| **S-BERT + Isolation Forest** | Unsupervised Dense Density | 0.4444 | 0.4000 | 0.5000 | 2.45 ms |
| **S-BERT + One-Class SVM** | Unsupervised Support Boundary | 0.4000 | 0.3333 | 0.5000 | 2.10 ms |
| **Supervised Linear Baseline** | Calibrated Logistic Regression | 0.6667 | 0.6000 | 0.7500 | 1.80 ms |
| **Proposed DGSA Framework** | Multi-Granular Context + Contradiction | **0.8889** | **0.8000** | **1.0000** | 3.15 ms |

---

## 📜 License

This project is licensed under the MIT License.
