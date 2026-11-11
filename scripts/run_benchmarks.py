"""
Benchmark execution script.
Runs zero-leakage research benchmark across all 6 models, 7 ablation configurations,
and token/span localization evaluations. Saves persistent output artifacts.
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
    print("=" * 70)
    print("SEMANTIC SENTINEL: ZERO-LEAKAGE RESEARCH BENCHMARK EXECUTION")
    print("=" * 70)

    # 1. Load Dataset
    data_path = Path("data/datasets/research_benchmark.json")
    if not data_path.exists():
        data_path = Path("data/datasets/synthetic_benchmark.json")

    docs = DatasetLoader.load_from_json(data_path)
    print(f"Loaded {len(docs)} multi-domain documents.")

    # 2. Strict Document-Level Split (50% Train, 25% Val, 25% Test)
    train_docs, val_docs, test_docs = DatasetLoader.split_train_val_test(
        docs, train_ratio=0.50, val_ratio=0.25, test_ratio=0.25, random_seed=42
    )

    print(f"Train Documents: {[d.doc_id for d in train_docs]}")
    print(f"Validation Documents: {[d.doc_id for d in val_docs]}")
    print(f"Test Documents: {[d.doc_id for d in test_docs]}")

    # 3. Initialize DGSA Model
    print("\nInitializing DGSA Model with all-MiniLM-L6-v2 & CrossEncoder NLI...")
    start_init = time.perf_counter()
    model = DGSAModel()
    print(f"Model initialized in {(time.perf_counter() - start_init):.2f}s")
    print(f"Contradiction Detector Backend: {model.contradiction_detector.backend}")

    # 4. Run Rigorous Benchmark
    harness = ExperimentHarness(model)
    print("\nRunning zero-leakage benchmark on held-out test documents...")
    res = harness.run_rigorous_benchmark(train_docs, val_docs, test_docs)

    benchmark_df = res["benchmark_results"]
    ablation_df = res["ablation_results"]
    span_metrics = res["span_metrics"]
    calibrated_th = res["calibrated_threshold"]
    calib_df = res["calibration_table"]

    # 5. Save Artifacts to outputs/benchmarks
    out_dir = Path("outputs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_df.to_csv(out_dir / "benchmark_results.csv", index=False)
    with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_df.to_dict(orient="records"), f, indent=2)

    ablation_df.to_csv(out_dir / "ablation_results.csv", index=False)
    calib_df.to_csv(out_dir / "calibration_sweep.csv", index=False)

    span_dict = span_metrics.to_dict()
    span_rows = [{
        "token_precision": span_metrics.token_precision,
        "token_recall": span_metrics.token_recall,
        "token_f1": span_metrics.token_f1,
        "exact_match_count": span_metrics.exact_match_count,
        "exact_match_rate": span_metrics.exact_match_rate,
        "partial_match_count": span_metrics.partial_match_count,
        "partial_match_rate": span_metrics.partial_match_rate,
        "mean_iou": span_metrics.mean_iou,
        "median_iou": span_metrics.median_iou,
        "total_gold_spans": span_metrics.total_ground_truth_spans,
        "total_pred_spans": span_metrics.total_predicted_spans
    }]
    pd.DataFrame(span_rows).to_csv(out_dir / "span_evaluation_results.csv", index=False)

    exp_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": config.model.primary_embedding_model,
        "nli_model": config.model.primary_nli_model,
        "contradiction_backend": model.contradiction_detector.backend,
        "train_doc_ids": [d.doc_id for d in train_docs],
        "val_doc_ids": [d.doc_id for d in val_docs],
        "test_doc_ids": [d.doc_id for d in test_docs],
        "calibrated_threshold": calibrated_th,
        "dgsa_weights": {"alpha": model.scorer.alpha, "beta": model.scorer.beta, "gamma": model.scorer.gamma}
    }
    with open(out_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    # 6. Generate Verification Reports (PDF & HTML)
    rep_dir = Path("outputs/reports")
    rep_dir.mkdir(parents=True, exist_ok=True)
    if res["test_doc_results"]:
        test_sample = res["test_doc_results"][0]
        pdf_path = ExportManager.export_to_pdf(test_sample, rep_dir / "benchmark_test_audit.pdf")
        html_path = ExportManager.export_to_html_report(test_sample, rep_dir / "benchmark_test_audit.html")
        json_path = ExportManager.export_to_json(test_sample, rep_dir / "benchmark_test_audit.json")
        csv_path = ExportManager.export_to_csv(test_sample, rep_dir / "benchmark_test_audit.csv")
        print(f"\nGenerated verification reports in {rep_dir}:")
        print(f" - PDF: {pdf_path}")
        print(f" - HTML: {html_path}")
        print(f" - JSON: {json_path}")
        print(f" - CSV: {csv_path}")

    # Print Summary Tables
    print("\n" + "=" * 70)
    print("COMPARATIVE BENCHMARK RESULTS (HELD-OUT TEST SET)")
    print("=" * 70)
    print(benchmark_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Latency (ms)"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS (HELD-OUT TEST SET)")
    print("=" * 70)
    print(ablation_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("SPAN LOCALIZATION METRICS (HELD-OUT TEST SET)")
    print("=" * 70)
    print(f"Token Precision:    {span_metrics.token_precision:.4f}")
    print(f"Token Recall:       {span_metrics.token_recall:.4f}")
    print(f"Token F1:           {span_metrics.token_f1:.4f}")
    print(f"Exact Match Rate:   {span_metrics.exact_match_rate:.4f} ({span_metrics.exact_match_count}/{span_metrics.total_ground_truth_spans})")
    print(f"Partial Match Rate: {span_metrics.partial_match_rate:.4f} ({span_metrics.partial_match_count}/{span_metrics.total_ground_truth_spans})")
    print(f"Mean Span IoU:      {span_metrics.mean_iou:.4f}")
    print(f"Median Span IoU:    {span_metrics.median_iou:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
