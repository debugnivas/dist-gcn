"""
Story 3.3 / 3.4: Results comparison and aggregation.

Loads all per-run CSV files, computes summary statistics,
generates comparison CSV and all visualization charts.
"""
import sys
import os
import glob
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RESULTS_DIR
from src.utils.metrics import (
    compute_speedup, compute_comm_overhead, get_peak_memory_mb,
    save_benchmark_summary,
)
from src.utils.visualization import generate_all_charts


def aggregate_results(results_dir: str = None):
    """
    Load all benchmark CSVs, compute summary statistics, and generate
    comparison outputs.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    print("=" * 60)
    print("RESULTS COMPARISON & AGGREGATION")
    print("=" * 60)

    # Find all run CSV files
    csv_pattern = os.path.join(results_dir, "*_run*_metrics.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        # Try non-run files
        csv_pattern = os.path.join(results_dir, "*_metrics.csv")
        csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        print("[Compare] No benchmark CSV files found in", results_dir)
        return

    print(f"[Compare] Found {len(csv_files)} result files")

    # Group runs by configuration
    config_runs = {}
    for f in csv_files:
        basename = os.path.basename(f).replace("_metrics.csv", "")
        # Parse: config_name is everything before _runN (or the whole name)
        if "_run" in basename:
            parts = basename.rsplit("_run", 1)
            config_name = parts[0]
        else:
            config_name = basename

        if config_name not in config_runs:
            config_runs[config_name] = []

        df = pd.read_csv(f)
        metrics_list = df.to_dict('records')
        config_runs[config_name].append(metrics_list)

    print(f"[Compare] Configurations found: {list(config_runs.keys())}")

    # Generate benchmark summary
    save_benchmark_summary(config_runs, results_dir)

    # Generate all charts
    print("\n[Compare] Generating visualization charts...")
    generate_all_charts(results_dir)

    # Print performance comparison
    _print_comparison(config_runs)


def _print_comparison(config_runs: dict):
    """Print a quick comparison summary."""
    print("\n" + "=" * 80)
    print("QUICK COMPARISON")
    print("=" * 80)

    # Get sequential baseline
    seq_time = None
    if 'sequential' in config_runs:
        seq_runs = config_runs['sequential']
        seq_times = [np.mean([m['epoch_time'] for m in run]) for run in seq_runs]
        seq_time = np.mean(seq_times)
        seq_f1 = np.mean([run[-1].get('test_f1', 0) for run in seq_runs])
        print(f"  Sequential: {seq_time:.2f}s/epoch, F1={seq_f1:.4f}")

    for config_name in sorted(config_runs.keys()):
        if config_name == 'sequential':
            continue
        runs = config_runs[config_name]
        avg_times = [np.mean([m['epoch_time'] for m in run]) for run in runs]
        avg_comm = [np.mean([m.get('comm_time', 0) for m in run]) for run in runs]
        avg_f1 = [run[-1].get('test_f1', 0) for run in runs]

        mean_time = np.mean(avg_times)
        mean_comm = np.mean(avg_comm)
        mean_f1 = np.mean(avg_f1)

        speedup = seq_time / mean_time if seq_time and mean_time > 0 else 0
        comm_pct = (mean_comm / mean_time * 100) if mean_time > 0 else 0

        print(f"  {config_name}: {mean_time:.2f}s/epoch, "
              f"Speedup={speedup:.2f}x, "
              f"Comm={comm_pct:.1f}%, "
              f"F1={mean_f1:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    aggregate_results(args.results_dir)
