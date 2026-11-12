"""
Benchmark execution script.
Runs multi-seed zero-leakage research benchmark across all 6 models, 7 ablation configurations,
fine-grained category-wise evaluation, and token/span localization evaluations.
Evaluates synthetic, real-world, and combined benchmark suites separately.
Saves reproducible output artifacts to outputs/benchmarks/ and outputs/reports/.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Run DGSA Research Benchmarks")
parser.add_argument(
    "--offline",
    action="store_true",
    default=False,
    help="Force offline mode using local Hugging Face cache (equivalent to SEMANTIC_SENTINEL_OFFLINE=1)"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="all",
    choices=["all", "synthetic", "real_world"],
    help="Which benchmark suite to evaluate: 'all' (default), 'synthetic', or 'real_world'"
)
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=[42, 123, 999],
    help="Random seeds for multi-seed statistical evaluation"
)
args, unknown = parser.parse_known_args()

# Configure offline mode opt-in
is_offline = args.offline or os.environ.get("SEMANTIC_SENTINEL_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
if is_offline:
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
    print(f"Execution Mode: {'Offline (Local Cache)' if is_offline else 'Online / Local Cache'}")
    print(f"Target Dataset: {args.dataset.upper()} | Seeds: {args.seeds}")
    print("=" * 75)

    # 1. Load Datasets
    synth_docs = DatasetLoader.load_synthetic_benchmark("data/datasets")
    try:
        real_docs = DatasetLoader.load_real_world_benchmark("data/datasets")
    except Exception as e:
        logger.warning(f"Could not load real-world benchmark ({e}). Generating now...")
        from data.create_real_world_benchmark import build_and_save_real_world_dataset
        build_and_save_real_world_dataset()
        real_docs = DatasetLoader.load_real_world_benchmark("data/datasets")

    print(f"Loaded {len(synth_docs)} Synthetic / Research Benchmark documents.")
    print(f"Loaded {len(real_docs)} Real-World Human-Curated Benchmark documents.")

    # 2. Initialize DGSA Model
    print(f"\nInitializing DGSA Model with {config.model.primary_embedding_model} & {config.model.default_nli_model}...")
    start_init = time.perf_counter()
    model = DGSAModel()
    print(f"Model initialized in {(time.perf_counter() - start_init):.2f}s")
    print(f"Contradiction Detector Backend: {model.contradiction_detector.backend}")

    harness = ExperimentHarness(model)
    out_dir = Path("outputs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Primary Synthetic / Research Benchmark Execution
    print(f"\n[1/3] Executing Multi-Seed Benchmark on Synthetic/Research Dataset across seeds {args.seeds}...")
    synth_res = harness.run_multi_seed_benchmark(
        documents=synth_docs,
        seeds=args.seeds,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25
    )

    summary_benchmark = synth_res["summary_benchmark"]
    raw_benchmark = synth_res["raw_benchmark"]
    summary_ablation = synth_res["summary_ablation"]
    raw_ablation = synth_res["raw_ablation"]
    summary_category = synth_res["summary_category"]
    span_summary = synth_res["aggregated_span_metrics"]
    confusion_matrices = synth_res["confusion_matrices"]

    # Save Primary Synthetic Benchmark Artifacts
    summary_benchmark.to_csv(out_dir / "benchmark_results_multi_seed.csv", index=False)
    raw_benchmark.to_csv(out_dir / "benchmark_results_raw.csv", index=False)
    summary_benchmark.to_csv(out_dir / "benchmark_results.csv", index=False)
    with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_benchmark.to_dict(orient="records"), f, indent=2)

    summary_ablation.to_csv(out_dir / "ablation_results_multi_seed.csv", index=False)
    summary_ablation.to_csv(out_dir / "ablation_results.csv", index=False)
    raw_ablation.to_csv(out_dir / "ablation_results_raw.csv", index=False)

    summary_category.to_csv(out_dir / "category_performance.csv", index=False)
    confusion_matrices.to_csv(out_dir / "confusion_matrices.csv", index=False)
    pd.DataFrame([span_summary]).to_csv(out_dir / "span_evaluation_results.csv", index=False)

    # 4. Real-World Benchmark Evaluation
    print(f"\n[2/3] Executing Multi-Seed Benchmark on Real-World Human-Curated Dataset across seeds {args.seeds}...")
    real_res = harness.run_multi_seed_benchmark(
        documents=real_docs,
        seeds=args.seeds,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25
    )
    real_summary_benchmark = real_res["summary_benchmark"]
    real_summary_benchmark.to_csv(out_dir / "real_world_benchmark_results.csv", index=False)
    real_res["summary_ablation"].to_csv(out_dir / "real_world_ablation_results.csv", index=False)

    # 5. Combined Benchmark Suite Evaluation
    combined_docs = synth_docs + real_docs
    print(f"\n[3/3] Executing Benchmark on Combined Suite ({len(combined_docs)} documents)...")
    combined_res = harness.run_multi_seed_benchmark(
        documents=combined_docs,
        seeds=args.seeds,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25
    )
    combined_res["summary_benchmark"].to_csv(out_dir / "combined_benchmark_results.csv", index=False)

    # Save comprehensive experiment metadata
    harness.save_experiment_metadata(
        out_path=out_dir / "experiment_config.json",
        seeds=args.seeds,
        dataset_meta={
            "dataset_name": "DGSA Research & Real-World Anomaly Benchmark Suite",
            "version": "3.1",
            "synthetic_documents": len(synth_docs),
            "real_world_documents": len(real_docs),
            "total_documents": len(combined_docs)
        },
        split_stats={
            "total_documents": len(synth_docs),
            "train_docs_per_seed": int(round(len(synth_docs) * 0.50)),
            "val_docs_per_seed": int(round(len(synth_docs) * 0.25)),
            "test_docs_per_seed": len(synth_docs) - int(round(len(synth_docs) * 0.50)) - int(round(len(synth_docs) * 0.25))
        }
    )

    # 6. Generate Sample Verification Audit Reports (PDF, HTML, JSON, CSV)
    rep_dir = Path("outputs/reports")
    rep_dir.mkdir(parents=True, exist_ok=True)

    sample_doc = real_docs[1]  # Form 10-K document
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

    # 7. Print Research Tables
    print("\n" + "=" * 75)
    print("SYNTHETIC RESEARCH BENCHMARK RESULTS (HELD-OUT TEST SET - MEAN +/- STD OVER 3 SEEDS)")
    print("=" * 75)
    print(summary_benchmark[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Latency (ms)"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("REAL-WORLD BENCHMARK RESULTS (HELD-OUT TEST SET - MEAN +/- STD OVER 3 SEEDS)")
    print("=" * 75)
    print(real_summary_benchmark[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Latency (ms)"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("DGSA ABLATION STUDY RESULTS (7 CONFIGURATIONS - MEAN +/- STD OVER 3 SEEDS)")
    print("=" * 75)
    print(summary_ablation[["Configuration", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("CONFUSION MATRICES SUMMARY (SUM ACROSS SEEDS)")
    print("=" * 75)
    print(confusion_matrices[["Model / Configuration", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1-Score", "Accuracy"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("CATEGORY-WISE ANOMALY DETECTION BREAKDOWN (SYNTHETIC)")
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
    print("All research artifacts successfully saved to outputs/benchmarks/ and outputs/reports/.")


if __name__ == "__main__":
    main()
