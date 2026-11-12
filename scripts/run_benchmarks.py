"""
Benchmark execution script.
Runs multi-seed zero-leakage research benchmark across all 6 models, 7 ablation configurations,
fine-grained category-wise evaluation, and token/span localization evaluations.
Saves reproducible output artifacts to outputs/benchmarks/ and outputs/reports/.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set offline mode for fast execution with cached models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np

from utils.logging_config import logger
from config.settings import config
from models.dgsa_model import DGSAModel
from data.dataset_loader import DatasetLoader
from evaluation.experiments import ExperimentHarness
from reports.export_results import ExportManager


def main():
    print("=" * 75)
    print("SEMANTIC SENTINEL: REPRODUCIBLE MULTI-SEED RESEARCH BENCHMARK EXECUTION")
    print("=" * 75)

    # 1. Load Dataset
    data_path = Path("data/datasets/research_benchmark.json")
    if not data_path.exists():
        data_path = Path("data/datasets/synthetic_benchmark.json")

    docs = DatasetLoader.load_from_json(data_path)
    print(f"Loaded {len(docs)} multi-domain documents.")

    # 2. Initialize DGSA Model
    print("\nInitializing DGSA Model with all-MiniLM-L6-v2 & CrossEncoder NLI...")
    start_init = time.perf_counter()
    model = DGSAModel()
    print(f"Model initialized in {(time.perf_counter() - start_init):.2f}s")
    print(f"Contradiction Detector Backend: {model.contradiction_detector.backend}")

    # 3. Run Multi-Seed Scientific Benchmark
    harness = ExperimentHarness(model)
    seeds = [42, 123, 999]
    print(f"\nExecuting multi-seed scientific benchmark across seeds {seeds} (50% Train, 25% Val, 25% Test)...")
    multi_res = harness.run_multi_seed_benchmark(
        documents=docs,
        seeds=seeds,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25
    )

    summary_benchmark = multi_res["summary_benchmark"]
    raw_benchmark = multi_res["raw_benchmark"]
    summary_ablation = multi_res["summary_ablation"]
    raw_ablation = multi_res["raw_ablation"]
    summary_category = multi_res["summary_category"]
    span_summary = multi_res["aggregated_span_metrics"]

    # 4. Save Artifacts to outputs/benchmarks
    out_dir = Path("outputs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_benchmark.to_csv(out_dir / "benchmark_results_multi_seed.csv", index=False)
    raw_benchmark.to_csv(out_dir / "benchmark_results_raw.csv", index=False)
    summary_benchmark.to_csv(out_dir / "benchmark_results.csv", index=False)
    with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_benchmark.to_dict(orient="records"), f, indent=2)

    summary_ablation.to_csv(out_dir / "ablation_results_multi_seed.csv", index=False)
    summary_ablation.to_csv(out_dir / "ablation_results.csv", index=False)
    raw_ablation.to_csv(out_dir / "ablation_results_raw.csv", index=False)

    summary_category.to_csv(out_dir / "category_performance.csv", index=False)
    pd.DataFrame([span_summary]).to_csv(out_dir / "span_evaluation_results.csv", index=False)

    # Save comprehensive experiment metadata
    harness.save_experiment_metadata(
        out_path=out_dir / "experiment_config.json",
        seeds=seeds,
        dataset_meta={
            "dataset_name": "DGSA Multi-Domain Semantic Anomaly Benchmark",
            "version": "3.0",
            "total_documents": len(docs)
        },
        split_stats={
            "total_documents": len(docs),
            "train_docs_per_seed": int(round(len(docs) * 0.50)),
            "val_docs_per_seed": int(round(len(docs) * 0.25)),
            "test_docs_per_seed": len(docs) - int(round(len(docs) * 0.50)) - int(round(len(docs) * 0.25))
        }
    )

    # 5. Generate Sample Verification Audit Reports (PDF, HTML, JSON, CSV)
    rep_dir = Path("outputs/reports")
    rep_dir.mkdir(parents=True, exist_ok=True)

    sample_doc = docs[2]  # Finance earnings document
    doc_text = " ".join(s.text for s in sample_doc.sentences)
    sample_res = model.analyze_document(doc_text)

    pdf_path = ExportManager.export_to_pdf(sample_res, rep_dir / "benchmark_test_audit.pdf")
    html_path = ExportManager.export_to_html_report(sample_res, rep_dir / "benchmark_test_audit.html")
    json_path = ExportManager.export_to_json(sample_res, rep_dir / "benchmark_test_audit.json")
    csv_path = ExportManager.export_to_csv(sample_res, rep_dir / "benchmark_test_audit.csv")

    print(f"\nGenerated verification reports in {rep_dir}:")
    print(f" - PDF:  {pdf_path}")
    print(f" - HTML: {html_path}")
    print(f" - JSON: {json_path}")
    print(f" - CSV:  {csv_path}")

    # 6. Print Summary Tables
    print("\n" + "=" * 75)
    print("COMPARATIVE BENCHMARK RESULTS (HELD-OUT TEST SET - MEAN +/- STD OVER 3 SEEDS)")
    print("=" * 75)
    print(summary_benchmark[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Latency (ms)"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("DGSA ABLATION STUDY RESULTS (7 CONFIGURATIONS - MEAN +/- STD OVER 3 SEEDS)")
    print("=" * 75)
    print(summary_ablation[["Configuration", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("CATEGORY-WISE ANOMALY DETECTION BREAKDOWN")
    print("=" * 75)
    print(summary_category.to_string(index=False))

    print("\n" + "=" * 75)
    print("SPAN LOCALIZATION METRICS (AGGREGATED ACROSS SEEDS)")
    print("=" * 75)
    print(f"Average Token Precision:  {span_summary['avg_token_precision']:.4f}")
    print(f"Average Token Recall:     {span_summary['avg_token_recall']:.4f}")
    print(f"Average Token F1:         {span_summary['avg_token_f1']:.4f}")
    print(f"Average Token IoU:        {span_summary['avg_token_iou']:.4f}")
    print(f"Exact Span Match Ratio:   {span_summary['avg_exact_match_ratio']:.4f}")
    print("=" * 75)
    print("All artifacts successfully saved to outputs/benchmarks/ and outputs/reports/.")


if __name__ == "__main__":
    main()
