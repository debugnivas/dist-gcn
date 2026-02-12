"""
Story 3.3: Sequential GCN benchmark runner.

Loads the Reddit dataset, runs the sequential GCN baseline,
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
    SEEDS, NUM_REPEATS, RESULTS_DIR
)
from src.data.loader import load_reddit_dataset
from src.model.gcn_sequential import SequentialGCN
from src.utils.metrics import save_epoch_metrics, get_peak_memory_mb


def run_sequential_benchmark(
    num_epochs: int = NUM_EPOCHS,
    seeds: list = None,
    results_dir: str = None,
    verbose: bool = True,
):
    """
    Run sequential GCN benchmark with repeated runs.

    Returns:
        all_metrics: List of per-run metric lists
    """
    if seeds is None:
        seeds = SEEDS[:NUM_REPEATS]
    if results_dir is None:
        results_dir = RESULTS_DIR

    print("=" * 60)
    print("SEQUENTIAL GCN BENCHMARK")
    print("=" * 60)

    # Load dataset
    print("\n[Benchmark] Loading Reddit dataset...")
    adj, features, labels, train_mask, val_mask, test_mask = load_reddit_dataset()

    input_dim = features.shape[1]
    num_classes = len(np.unique(labels))

    print(f"\n[Benchmark] Config: L={NUM_LAYERS}, d={HIDDEN_DIM}, lr={LEARNING_RATE}, "
          f"epochs={num_epochs}")
    print(f"[Benchmark] Input dim: {input_dim}, Classes: {num_classes}")
    print(f"[Benchmark] Running {len(seeds)} repeated runs with seeds: {seeds}")

    all_metrics = []

    for run_idx, seed in enumerate(seeds):
        print(f"\n--- Run {run_idx + 1}/{len(seeds)} (seed={seed}) ---")

        model = SequentialGCN(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            output_dim=num_classes,
            num_layers=NUM_LAYERS,
            learning_rate=LEARNING_RATE,
            seed=seed,
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
        config_name = f"sequential_run{run_idx + 1}"
        save_epoch_metrics(epoch_metrics, config_name, results_dir)

        all_metrics.append(epoch_metrics)

    # Save combined metrics
    save_epoch_metrics(all_metrics[0], "sequential", results_dir)

    print(f"\n[Benchmark] Sequential benchmark complete. Peak memory: {get_peak_memory_mb():.1f} MB")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sequential GCN benchmark")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--repeats", type=int, default=NUM_REPEATS)
    args = parser.parse_args()

    seeds = SEEDS[:args.repeats]
    run_sequential_benchmark(num_epochs=args.epochs, seeds=seeds)
