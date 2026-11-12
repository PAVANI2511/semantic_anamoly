# Changelog: Semantic Sentinel

All notable changes, architectural improvements, experimental benchmarks, and bug fixes for **Semantic Sentinel – Context-Aware Semantic Anomaly Detection and Span Localization in Documents** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] - 2026-11-12

### 🔬 Research Validation & Multi-Seed Benchmark Edition

#### Added
* **Expanded Multi-Domain Benchmark Dataset (v3.0):**
  * Expanded benchmark from 12 to **20 multi-domain documents** (142 sentences, 17 controlled anomalies) across 20 distinct scientific and professional fields: Environmental Science, Quantum Physics, Corporate Finance, Cardiology, Planetary Astronomy, Cybersecurity, Software Licensing Law, Machine Learning Optimization, Macroeconomics, Oncology/Immunology (*Pure Normal Control 1*), Renewable Energy, Aerospace Engineering, Supply Chain Logistics (*Temporal Inconsistency*), Cardiovascular Pharmacology (*Numerical Inconsistency*), Autonomous Vehicle Perception (*Topic Deviation*), Constitutional Law (*Pure Normal Control 2*), Semiconductor Lithography (*Contextual Inconsistency*), Diplomatic History (*Entity Inconsistency*), Commercial Avionics (*Attribute Inconsistency*), and Hydropower Engineering (*Temporal Inconsistency*).
  * Gold character offsets `[start_char, end_char]` verified for every token span.
  * Added `NUMERICAL_INCONSISTENCY` and `TEMPORAL_INCONSISTENCY` to `AnomalyCategories.ALL_CATEGORIES`, providing native coverage for all 7 anomaly categories.

* **Fair Validation Baseline Threshold Calibration:**
  * Implemented `BaselineRunner.calibrate_threshold` on the **Validation split only** ($T^*_{baseline} \in [0.10, 0.90]$ maximizing validation F1).
  * Applied fair calibration across all 4 baseline models:
    1. TF-IDF + Isolation Forest
    2. S-BERT + Isolation Forest
    3. S-BERT + One-Class SVM
    4. Supervised Logistic Regression
  * Eliminates the previous hardcoded 0.50 threshold to guarantee strict scientific fairness when comparing baselines against DGSA.

* **Multi-Seed Repeated Experiments & Statistical Reliability:**
  * Implemented `ExperimentHarness.run_multi_seed_benchmark` across seeds `[42, 123, 999]` with 50% Train / 25% Validation / 25% Test document-level splits.
  * Output tables now report empirical **mean $\pm$ standard deviation** across seeds for all metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Latency (ms), and Latency/Sentence.
  * Generated persistent research artifacts:
    * `outputs/benchmarks/benchmark_results_multi_seed.csv`
    * `outputs/benchmarks/ablation_results_multi_seed.csv`
    * `outputs/benchmarks/category_performance.csv`
    * `outputs/benchmarks/span_evaluation_results.csv`
    * `outputs/benchmarks/experiment_config.json`

* **Targeted Span Localization:**
  * Replaced unconstrained out-of-vocabulary tagging in `SpanLocalizer` with category-targeted attribution:
    * *Contradiction:* Exact polarity pivots and negation markers (`decreased`, `not`, `failed`).
    * *Entity Inconsistency:* Out-of-context capitalized proper nouns (`Detective Miller`).
    * *Attribute, Numerical & Temporal Inconsistency:* Numbers, metric units, temporal prepositions, and contrasting descriptors (`3125.0 grams`, `five weeks before`).
    * *Topic Deviation:* Salient content vocabulary outliers with zero document occurrence (`bake`, `chocolate cake`).
  * Boosted Token Precision from **0.0786 to 0.2170 (+176.1% relative improvement)**.

* **Transparent Explainable AI (XAI):**
  * Structured separation in `AnomalyExplanation` between:
    * `direct_model_evidence`: Continuous neural outputs (CrossEncoder contradiction probabilities, cosine distances from centroid, neighborhood divergence, neural confidence).
    * `heuristic_indicators`: Rule-based diagnostic clues (bridge continuity checks, entity novelty, numerical mismatch).
    * `contradiction_backend`: Explicitly records whether `nli_transformer` or `lexical_fallback` produced the inference.

* **Comprehensive Unit Test Suite:**
  * Expanded automated tests to **37 tests** (100% pass rate) covering:
    * Mocked successful NLI loading (`unittest.mock.patch`).
    * Mocked loading failure and graceful lexical fallback.
    * Dynamic backend switching and singleton model caching.
    * Multi-seed 3-way document-level zero-leakage partitions.
    * Fair baseline threshold calibration.
    * Edge cases: empty documents, whitespace strings, single-sentence documents ($G(i)=0$).
    * Multi-format export validation (PDF, HTML, JSON, CSV).

#### Fixed
* **Pyrefly `Optional` Import Diagnostic:** Resolved missing `Optional` typing import in `evaluation/experiments.py`.
* **NLI Backend Dynamic Switching Test:** Fixed `test_dynamic_backend_switching` in `tests/test_nli_contradiction.py` to assert against actual backend availability rather than falsely expecting offline environments to switch without models.
* **Redundant Baseline Return:** Removed duplicate return statement in `evaluation/baselines.py`.
* **Ablation Computation Efficiency:** Vectorized the 7 ablation signal combinations ($A = \alpha G + \beta L + \gamma C$) in `evaluation/experiments.py` using cached single-pass test results, reducing multi-seed benchmark execution time from >10 minutes to ~45 seconds on CPU.

---

## [2.0.0] - 2026-11-11

### 🚀 Complete DGSA Framework Architecture Upgrade

#### Added
* **Dual-Granularity Semantic Anomaly Detection (DGSA):**
  * Tripartite mathematical formulation: $A(i) = \alpha G(i) + \beta L(i) + \gamma C(i)$.
  * Global semantic deviation ($G(i)$) using document centroid cosine divergence.
  * Local contextual inconsistency ($L(i)$) using sliding neighborhood cosine similarity and bridge discontinuity detection.
  * Deep NLI contradiction detection ($C(i)$) using Hugging Face Cross-Encoder (`cross-encoder/nli-deberta-v3-xsmall`).
* **Document Extraction & Boundary Tracking:**
  * Support for PDF (`pdfplumber` / `pypdf`), DOCX (`python-docx`), and plain text.
  * Exact character offset preservation `[start_char, end_char]` across sentences and paragraphs.
* **Authentic Multi-Format Reporting:**
  * Executive PDF generation using ReportLab 5.1 with executive scorecards and color-coded anomaly highlights.
  * Secure XSS-escaped interactive HTML reports.
  * Machine-readable structured JSON and spreadsheet-ready CSV.
* **Streamlit Web Application (`app.py`):**
  * Full dashboard featuring document input, parameter adjustment sliders ($\alpha, \beta, \gamma, T^*$), anomaly timeline visualization, interactive baseline comparisons, ablation heatmaps, and one-click PDF downloads.

---

## [1.0.0] - 2026-11-10

### 📦 Initial Prototype
* Initial project setup and repository initialization.
* Basic TF-IDF sentence representation and heuristic keyword anomaly rules.
* Prototype Streamlit user interface.
