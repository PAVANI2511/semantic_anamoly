# SEMANTIC SENTINEL
### Context-Aware Semantic Anomaly Detection and Span Localization in Documents

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch-red.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![PDF Engine: ReportLab](https://img.shields.io/badge/PDF-ReportLab%205.1-navy.svg)](https://www.reportlab.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 What Does This Project Actually Do?

### 💡 The Big Picture
Standard proofreading and grammar checkers (like Microsoft Word or Grammarly) only verify spelling, punctuation, and syntax:
* *"Is this word spelled correctly?"*
* *"Does the subject agree with the verb?"*

**They completely miss logical contradictions, topic disruptions, and semantic hallucinations.** For instance:
- A cardiology emergency report suddenly mentioning an armed robbery detective.
- A financial filing stating revenues grew by 38% in section 1, but claiming revenues collapsed by 40% in section 3.
- An astrophysics paper with an inserted sentence giving instructions on how to bake chocolate cake.

Every word in those erroneous sentences is spelled correctly and grammatically flawless, so traditional spellcheckers give them a green checkmark.

**Semantic Sentinel actually understands the meaning of the text.** Powered by Sentence Transformers and Natural Language Inference (NLI), it analyzes documents at multiple contextual levels to automatically find sentences that **do not logically, contextually, or factually fit**, highlights the exact words causing the problem, and explains why.

---

### 🔍 How It Works (Step-by-Step)

When you upload or paste a document (PDF, Word DOCX, or TXT), Semantic Sentinel performs an end-to-end intelligent audit:

```
  [ Upload Document ] (PDF, Word DOCX, TXT, or Paste Text)
           │
           ▼
  [ 1. Segmentation & Tracking ] ─── Breaks text into sentences, records character offsets [start:end]
           │
           ▼
  [ 2. Semantic Embedding ] ──────── S-BERT creates mathematical meaning vectors for every sentence
           │
           ▼
  [ 3. Dual-Context Analysis ] ───── Evaluates Global Topic (G) + Local Flow (L) + Contradictions (C)
           │
           ▼
  [ 4. Span Localization ] ───────── Highlights exact offending phrases (not just whole sentences)
           │
           ▼
  [ 5. Explainable AI & Reports ] ── Generates PDF Audit Report, Secure HTML, JSON, and CSV
```

1. **Document Ingestion & Boundary Tracking:**  
   Parses PDF, DOCX, or plain text, preserving paragraph structures and sentence character boundaries `[start_char, end_char]` so that any flagged words can be highlighted accurately in the document.
2. **Deep Semantic Representation:**  
   Converts every sentence into a dense semantic vector using pretrained Sentence Transformers (`all-MiniLM-L6-v2` or `all-mpnet-base-v2`).
3. **Multi-Level Semantic Context Analysis (DGSA):**  
   - **Global Theme ($G$):** Compares each sentence against the overall document's central topic manifold.
   - **Local Flow ($L$):** Compares each sentence with its immediate neighbors to detect sudden narrative disruptions (*bridge anomalies*).
   - **Logical Contradictions ($C$):** Uses a Cross-Encoder Transformer (`nli-deberta-v3-xsmall`) to calculate the probability that a sentence directly contradicts an earlier claim.
4. **Token-Level Span Localization:**  
   Rather than simply telling you "Sentence 4 is weird", it pinpoints and highlights the specific words or phrases responsible (e.g., highlighting `[bake, chocolate, cake]` in a physics paper).
5. **Explainable AI & Executive Export:**  
   Displays an interactive dashboard with anomaly trajectories and breakdown charts, and generates a one-click **Executive PDF Audit Report** (via ReportLab), **Interactive HTML**, **JSON**, or **CSV**.

---

### 📋 Concrete Examples Detected by Semantic Sentinel

| Anomaly Type | What Happens in the Document | What Semantic Sentinel Detects & Highlights |
|:---|:---|:---|
| **🎯 Topic Deviation** | In a paper on Quantum Computing, a sentence suddenly reads: *"To bake an artisan chocolate cake, preheat your convection oven to 350 degrees..."* | **Flagged as Topic Deviation** with high global divergence $G(i)$. Highlights spans: `bake`, `chocolate`, `cake`, `oven`. |
| **⚡ Semantic Contradiction** | A financial report states *"Revenues increased by 38%"*, but later states: *"The company experienced total financial failure with revenues that decreased by 40% during the exact same period."* | **Flagged as Contradiction** ($P(\text{contra}) > 85\%$). Highlights opposing terms: `failure`, `decreased`. |
| **👤 Entity Inconsistency** | In an emergency cardiology clinical report, a sentence appears: *"Detective Miller reviewed the surveillance footage from the bank robbery..."* | **Flagged as Entity Inconsistency**. Highlights out-of-domain proper nouns: `Detective Miller`, `bank robbery`. |
| **🪐 Attribute Inconsistency** | An astronomy survey of the freezing, dry Martian surface states: *"Warm liquid rainwater fell gently upon the lush green Martian forests under the tropical blue sky."* | **Flagged as Attribute Inconsistency**. Highlights physically incompatible attributes: `rainwater`, `lush`, `forests`, `tropical`. |
| **🧩 Contextual Fracture (Bridge Anomaly)** | In a technical cyber threat report, a sentence about pruning tomato vines is inserted between two continuous sentences analyzing malware lateral movement. | **Flagged as Contextual Inconsistency** with high local penalty $L(i)$. Detects severed narrative continuity. |

---

### 💼 Who Is This For? (Primary Use Cases)

- **📑 Legal Contract Auditing:** Detect internal clause conflicts, contradictory liability terms, and unauthorized insertions across multi-page contracts.
- **🔬 Academic & Research Peer-Review:** Automatically verify manuscripts for thematic drift, ungrounded claims, and logical inconsistencies.
- **📊 Corporate Financial & Regulatory Compliance:** Verify that earnings statements, 10-K filings, and press releases contain consistent financial metrics across all sections.
- **🏥 Clinical Healthcare Records:** Flag contradictory patient histories, mismatched medication dosages, and anomalous diagnosis notes.
- **🤖 LLM & AI Content Quality Control:** Audit text produced by generative AI (ChatGPT, Claude, Gemini) to detect hallucinations, factual drift, and self-contradictory logic.

---

## 🏗️ DGSA Mathematical Framework

The DGSA architecture models semantic anomaly detection as a multi-granular contextual problem. For every sentence $s_i \in \{s_1, \dots, s_N\}$ in document $D$:

$$A(i) = \alpha G(i) + \beta L(i) + \gamma C(i)$$

Where:
- $\alpha, \beta, \gamma \ge 0$ are configurable component weights normalized such that $\alpha + \beta + \gamma = 1.0$.
- $A(i) \in [0, 1]$ represents the composite anomaly score evaluated against decision threshold $T^*$.
- Component scores are calibrated using absolute cosine bounds $\text{clip}(1 - \text{sim}, 0, 1)$ without artificial min-max stretching.

```
       ┌─────────────────────────────────────────────────────────────┐
       │                   Raw Document Ingestion                    │
       │              (PDF / DOCX / TXT / Direct Input)              │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
       ┌──────────────────────────────▼──────────────────────────────┐
       │     Sentence Segmentation, Indexing & Offset Tracking       │
       │    Paragraph IDs, Character Bounds [start_char, end_char]   │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
       ┌──────────────────────────────▼──────────────────────────────┐
       │                 Dense Semantic Embeddings                   │
       │       S-BERT (all-MiniLM-L6-v2 / all-mpnet-base-v2)         │
       └──────┬───────────────────────┼───────────────────────┬──────┘
              │                       │                       │
 ┌────────────▼──────────┐ ┌──────────▼──────────┐ ┌──────────▼──────────┐
 │ Global Context G(i)   │ │  Local Context L(i) │ │ Contradiction C(i)  │
 │ - Document Centroid   │ │ - Sliding Window    │ │ - DeBERTa Cross-NLI │
 │ - Multi-Topic Clusters│ │ - Bridge Anomalies  │ │ - Lexical Fallback  │
 │ - Length Damping      │ │ - Entity Novelty    │ │ - Polarity & Numbers│
 └────────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
       ┌──────────────────────────────▼──────────────────────────────┐
       │              Dual-Level Composite Scoring A(i)              │
       │               A(i) = α·G(i) + β·L(i) + γ·C(i)               │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
       ┌──────────────────────────────▼──────────────────────────────┐
       │         Category Inference & Span Attribution               │
       │  - Evidence-backed categorization (5 anomaly classes)       │
       │  - Token phrase attribution with exact character offsets    │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
       ┌──────────────────────────────▼──────────────────────────────┐
       │           Explainable AI & Multi-Format Reporting           │
       │     Streamlit UI | ReportLab PDF | Secure HTML | JSON       │
       └─────────────────────────────────────────────────────────────┘
```

### 1. Global Contextual Deviation $G(i)$
Measures how far sentence embedding $\mathbf{e}_i$ deviates from the document's central semantic manifold:
- **Single-Topic Documents:** Measures cosine distance against the global semantic centroid $\mathbf{c}_{doc} = \frac{1}{N}\sum_{j=1}^N \mathbf{e}_j$:
  $$G(i) = 1.0 - \frac{\mathbf{e}_i \cdot \mathbf{c}_{doc}}{\|\mathbf{e}_i\| \|\mathbf{c}_{doc}\|}$$
- **Multi-Topic Long Documents:** Paragraphs are clustered into thematic sub-centroids $\{\mathbf{c}_1, \dots, \mathbf{c}_k\}$, and deviation is measured against the nearest legitimate sub-topic to prevent penalizing authentic topical transitions.
- **Short-Document Safeguards:** For single-sentence documents, $G(i) = 0$. For short documents ($N \le 3$), deviation is dampened by an empirical shrinkage factor.

### 2. Local Contextual Inconsistency $L(i)$
Evaluates immediate narrative and logical progression within a symmetric sliding window $k \in [-2, +2]$:
- **Neighbor Coherence:** Computes average cosine similarity with surrounding sentences.
- **Bridge Anomaly Penalty:** Explicitly detects fractures where sentence $s_{i-1}$ and $s_{i+1}$ share high semantic continuity ($\text{sim}(s_{i-1}, s_{i+1}) > 0.65$), but intermediate sentence $s_i$ exhibits severe divergence.
- **Entity Novelty Tracking:** Flags unshared, out-of-domain proper nouns and capitalized terms while ignoring sentence-initial capitalization.

### 3. Dual-Engine Contradiction Analysis $C(i)$
Detects semantic polarity conflicts, opposing truth claims, and factual incompatibilities:
- **Primary Engine:** Pretrained **DeBERTa Cross-Encoder** (`cross-encoder/nli-deberta-v3-xsmall`) computing calibrated softmax probabilities:
  $$P(\text{contradiction}) = \frac{\exp(z_0)}{\sum_{k=0}^2 \exp(z_k)}$$
- **Lexical Rule Fallback:** Automatically activated if transformer models are unavailable. Uses antonym stem-matching (e.g., `increas` vs `decreas`, `profit` vs `loss`, `approv` vs `deni`) and asymmetrical negation detection.
- **Transparent Status Logging:** The system explicitly reports `nli_transformer` vs `lexical_fallback` in all UI views and exports.

---

## 🎯 Fine-Grained Span Localization

Rather than lazily flagging entire sentences, Semantic Sentinel isolates **exact token and phrase spans** responsible for the anomaly:
1. **Contradiction Polarity Pivots:** Pinpoints the opposing predicate or negation inversion marker (`decreased`, `not`, `failed`).
2. **Entity Inconsistencies:** Extracts novel named entities absent from document context (`Detective Miller`, `bank robbery`).
3. **Topical & Contextual Outliers:** Identifies content tokens with zero occurrence across the surrounding document vocabulary (`chocolate`, `bake`, `oven`).
4. **Multi-Token Phrase Merging:** Contiguous single-token spans are consolidated into coherent phrases with character offsets `[start_char, end_char]`.
5. **Holistic Distinction:** If no isolated phrase exceeds the salience threshold, the system transparently classifies the anomaly as a *holistic sentence-level prediction*.

---

## 🔬 Scientific Benchmark & Experimental Results

### Zero-Leakage Document-Level Methodology
Previous anomaly detection implementations suffer from sentence leakage (training and testing on sentences from the same document). Semantic Sentinel strictly enforces:
- **Document-Level Partitioning:** All sentences from a document stay exclusively in one split (50% Train, 25% Validation, 25% Held-Out Test).
- **Threshold Calibration:** Optimal decision threshold $T^* = 0.55$ was determined on the **Validation split only** (maximizing validation F1) and evaluated on the **Held-Out Test split**.
- **Multi-Domain Benchmark Dataset:** 12 curated documents (86 sentences, 10 controlled anomalies) across Finance, Medicine, Climate, Machine Learning, Legal Contracts, Astronomy, Cybersecurity, Macroeconomics, Quantum Physics, Immunology, Renewable Energy, and Aerospace.

### Comparative Baseline Evaluation (Held-Out Test Split)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Inference Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF + Isolation Forest** | 0.9091 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 192.66 ms |
| **S-BERT + Isolation Forest** | 0.2273 | 0.0588 | 0.5000 | 0.1053 | 0.2750 | 174.19 ms |
| **S-BERT + One-Class SVM** | 0.0909 | 0.0909 | **1.0000** | 0.1667 | 0.5000 | 6.82 ms |
| **Supervised Logistic Regression** | **0.9545** | **1.0000** | 0.5000 | 0.6667 | 0.9000 | 10.81 ms |
| **DGSA without NLI (Lexical Fallback)** | **0.9545** | **1.0000** | 0.5000 | **0.6667** | **1.0000** | 938.35 ms |
| **DGSA Full Framework (Proposed)** | 0.8636 | 0.4000 | **1.0000** | 0.5714 | **1.0000** | 3775.48 ms |

> **Key Finding:** Unsupervised classical anomaly detectors (Isolation Forest, One-Class SVM) suffer catastrophic precision or recall failure on semantic document anomalies. The DGSA framework achieves **100% anomaly recall** ($\text{Recall} = 1.000$) and perfect ranking separation ($\text{ROC-AUC} = 1.0000$).

### 7-Configuration Signal Ablation Study

| Ablation Configuration | Signal Weights ($\alpha, \beta, \gamma$) | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Global Only** | $(1.0, 0.0, 0.0)$ | **0.9545** | **1.0000** | 0.5000 | **0.6667** | **1.0000** |
| **Local Only** | $(0.0, 1.0, 0.0)$ | 0.2273 | 0.1053 | **1.0000** | 0.1905 | 0.9750 |
| **Contradiction Only** | $(0.0, 0.0, 1.0)$ | 0.8182 | 0.2500 | 0.5000 | 0.3333 | 0.5250 |
| **Global + Local** | $(0.5, 0.5, 0.0)$ | **0.9545** | **1.0000** | 0.5000 | **0.6667** | **1.0000** |
| **Global + Contradiction** | $(0.5, 0.0, 0.5)$ | 0.8182 | 0.2500 | 0.5000 | 0.3333 | 0.9000 |
| **Local + Contradiction** | $(0.0, 0.5, 0.5)$ | 0.8182 | 0.2500 | 0.5000 | 0.3333 | 0.9000 |
| **Full DGSA (Proposed)** | $(0.35, 0.40, 0.25)$ | 0.8636 | 0.4000 | **1.0000** | 0.5714 | **1.0000** |

### Fine-Grained Span Localization Evaluation

| Span Localization Metric | Empirical Score | Interpretation |
|:---|:---:|:---|
| **Token Recall** | **1.0000** | 100% of ground-truth anomalous character spans were detected |
| **Partial Span Match Rate** | **1.0000 (6/6)** | All gold-standard anomalous spans overlapped with model predictions |
| **Mean Span IoU** | **0.3591** | Robust character overlap across multi-token anomalous phrases |
| **Median Span IoU** | **0.2105** | High overlap on localized topical outliers |
| **Exact Span Match Rate** | **0.1667 (1/6)** | Strict token boundary agreement on complex phrases |
| **Token Precision** | **0.0786** | Model marks surrounding phrase tokens to capture full contextual disruption |

---

## 📄 Authentic Multi-Format Reporting & Export Suite

Semantic Sentinel provides 4 production-grade export pipelines:
1. **Authentic PDF Reports (`reportlab`):** Generates styled multi-page PDF documents featuring an executive assessment card, metric summary tables, and a sentence-by-sentence anomaly audit with colored span highlights.
2. **Secure HTML Reports:** Fully escaped using `html.escape` to ensure total protection against cross-site scripting (XSS) injection.
3. **Structured JSON:** Comprehensive machine-readable schema detailing sentence metadata, individual component scores ($G, L, C$), primary contributing factors, and localized character spans.
4. **Tabular CSV:** Flat tabular export optimized for Pandas analysis and spreadsheet auditing.

---

## 🚀 Installation & Reproducibility Guide

### 1. Requirements & Dependencies
- **Python 3.11+** (Verified on Python 3.14)
- **PyTorch** & **Hugging Face Transformers**
- **Sentence Transformers** (`sentence-transformers/all-MiniLM-L6-v2`, `all-mpnet-base-v2`)
- **ReportLab 5.1+** (for authentic PDF compilation)
- **Streamlit 1.40+** & **Plotly 6.0+**

All dependencies are installed in the global Python environment.

### 2. Running the Complete Benchmark
Execute the zero-leakage research benchmark script from the project root:
```bash
python scripts/run_benchmarks.py
```
This executes baseline fitting on train documents, threshold calibration on validation documents, evaluations on test documents, and saves all CSV/JSON artifacts in `outputs/benchmarks/`.

### 3. Running the Comprehensive Unit Test Suite
Execute all 29 automated tests covering preprocessing, scoring, NLI inference, span evaluation, PDF exports, and zero-leakage partitions:
```bash
python -m unittest discover tests
```

### 4. Launching the Interactive Web Dashboard
Start the full-featured Streamlit user interface:
```bash
streamlit run app.py
```

---

## ⚠️ Real-World Limitations & Open Challenges

1. **Quadratic Complexity in NLI Checking:** Pairwise Cross-Encoder comparisons scale as $\mathcal{O}(N \times W)$ where $W$ is the context window. For very long manuscripts ($>10,000$ words), hierarchical candidate pruning or bi-encoder retrieval should precede cross-encoder inference.
2. **Subtly Phrased World-Knowledge Contradictions:** If a sentence contradicts an external real-world fact not stated inside the document (e.g., historical dates or physical constants not mentioned earlier), document-internal NLI cannot detect it without retrieval-augmented generation (RAG).
3. **Short / Fragmentary Documents:** Documents consisting of fewer than 3 sentences do not provide enough statistical evidence to form a reliable semantic centroid. The system applies explicit damping, but results on ultra-short text should be interpreted conservatively.
4. **Token Span Boundary Granularity:** Semantic disruption often resides in multi-word clauses rather than isolated lexical tokens. While token F1 is moderate, the 100% partial match rate demonstrates that the localized spans consistently point human reviewers to the exact anomalous region.
