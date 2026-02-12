"""
Story 3.3: Spark GCN benchmark runner.

Runs the distributed GCN with varying partition counts,
collects all metrics, and saves to CSV.
"""
import sys
import os
import time
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    NUM_LAYERS, HIDDEN_DIM, LEARNING_RATE, NUM_EPOCHS,
    SEEDS, NUM_REPEATS, PARTITION_COUNTS, RESULTS_DIR,
    SPARK_DRIVER_MEMORY
)
from src.data.loader import load_reddit_dataset
from src.model.gcn_spark import SparkGCN
from src.utils.metrics import save_epoch_metrics, get_peak_memory_mb


def run_spark_benchmark(
    num_workers: int = 4,
    num_epochs: int = NUM_EPOCHS,
    seeds: list = None,
    results_dir: str = None,
    verbose: bool = True,
):
    """
    Run Spark GCN benchmark for a given worker count with repeated runs.

    Returns:
        all_metrics: List of per-run metric lists
    """
    if seeds is None:
        seeds = SEEDS[:NUM_REPEATS]
    if results_dir is None:
        results_dir = RESULTS_DIR

    print("=" * 60)
    print(f"SPARK GCN BENCHMARK (workers={num_workers})")
    print("=" * 60)

    # Load dataset
    print("\n[Benchmark] Loading Reddit dataset...")
    adj, features, labels, train_mask, val_mask, test_mask = load_reddit_dataset()

    input_dim = features.shape[1]
    num_classes = len(np.unique(labels))

    print(f"\n[Benchmark] Config: L={NUM_LAYERS}, d={HIDDEN_DIM}, lr={LEARNING_RATE}, "
          f"epochs={num_epochs}, workers={num_workers}")
    print(f"[Benchmark] Running {len(seeds)} repeated runs with seeds: {seeds}")

    all_metrics = []

    for run_idx, seed in enumerate(seeds):
        print(f"\n--- Run {run_idx + 1}/{len(seeds)} (seed={seed}) ---")

        model = SparkGCN(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            output_dim=num_classes,
            num_layers=NUM_LAYERS,
            learning_rate=LEARNING_RATE,
            seed=seed,
            num_workers=num_workers,
            driver_memory=SPARK_DRIVER_MEMORY,
        )

        run_start = time.perf_counter()
        epoch_metrics = model.train(
            adj, features, labels,
            train_mask, val_mask, test_mask,
            num_epochs=num_epochs,
            verbose=verbose,
        )
        run_time = time.perf_counter() - run_start

        print(f"  Run total time: {run_time:.2f}s")
        print(f"  Final Test F1: {epoch_metrics[-1]['test_f1']:.4f}")

        # Save per-run metrics
        config_name = f"spark_{num_workers}_run{run_idx + 1}"
        save_epoch_metrics(epoch_metrics, config_name, results_dir)

        all_metrics.append(epoch_metrics)

    # Save the first run as the canonical result
    save_epoch_metrics(all_metrics[0], f"spark_{num_workers}", results_dir)

    print(f"\n[Benchmark] Spark (workers={num_workers}) benchmark complete. "
          f"Peak memory: {get_peak_memory_mb():.1f} MB")
    return all_metrics


def run_all_spark_benchmarks(
    partition_counts: list = None,
    num_epochs: int = NUM_EPOCHS,
    seeds: list = None,
    results_dir: str = None,
    verbose: bool = True,
):
    """Run Spark benchmarks for all partition counts."""
    if partition_counts is None:
        partition_counts = PARTITION_COUNTS
    if seeds is None:
        seeds = SEEDS[:NUM_REPEATS]

    all_results = {}
    for n_workers in partition_counts:
        metrics = run_spark_benchmark(
            num_workers=n_workers,
            num_epochs=num_epochs,
            seeds=seeds,
            results_dir=results_dir,
            verbose=verbose,
        )
        all_results[f"spark_{n_workers}"] = metrics

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spark GCN benchmark")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of workers (omit to run all: 1,2,4,8,16)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--repeats", type=int, default=NUM_REPEATS)
    args = parser.parse_args()

    seeds = SEEDS[:args.repeats]

    if args.workers is not None:
        run_spark_benchmark(
            num_workers=args.workers,
            num_epochs=args.epochs,
            seeds=seeds,
        )
    else:
        run_all_spark_benchmarks(
            num_epochs=args.epochs,
            seeds=seeds,
        )
