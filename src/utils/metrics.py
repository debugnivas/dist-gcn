"""
Story 3.1: Metrics computation.

Speedup, communication cost, F1-score, accuracy calculations.
CSV output for benchmark results.
"""
import os
import resource
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import f1_score, accuracy_score

from src.config import RESULTS_DIR


def compute_f1(preds: np.ndarray, labels: np.ndarray,
               mask: np.ndarray) -> float:
    """Compute macro F1-score on masked nodes."""
    if mask.sum() == 0:
        return 0.0
    return float(f1_score(
        labels[mask], preds[mask],
        average='macro', zero_division=0
    ))


def compute_accuracy(preds: np.ndarray, labels: np.ndarray,
                     mask: np.ndarray) -> float:
    """Compute accuracy on masked nodes."""
    if mask.sum() == 0:
        return 0.0
    return float(accuracy_score(labels[mask], preds[mask]))


def compute_speedup(t_sequential: float, t_parallel: float) -> float:
    """Compute speedup = T_sequential / T_parallel."""
    if t_parallel <= 0:
        return float('inf')
    return t_sequential / t_parallel


def compute_comm_overhead(comm_time: float, total_time: float) -> float:
    """Communication overhead as percentage of total epoch time."""
    if total_time <= 0:
        return 0.0
    return (comm_time / total_time) * 100.0


def get_peak_memory_mb() -> float:
    """
    Get peak memory usage in MB.
    macOS: ru_maxrss is in bytes.
    Linux: ru_maxrss is in KB.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS returns bytes, Linux returns KB
    import sys
    if sys.platform == 'darwin':
        return usage.ru_maxrss / (1024 * 1024)
    else:
        return usage.ru_maxrss / 1024


def save_epoch_metrics(
    metrics_list: List[Dict],
    config_name: str,
    results_dir: str = None,
) -> str:
    """
    Save per-epoch metrics to CSV.

    Unified schema:
    epoch, total_time, forward_time, backward_time, comm_time, update_time,
    loss, train_acc, val_f1, test_f1, peak_memory_mb

    Returns path to saved CSV.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    for m in metrics_list:
        rows.append({
            'epoch': m.get('epoch', 0),
            'epoch_time': m.get('epoch_time', 0.0),
            'forward_time': m.get('forward_time', 0.0),
            'backward_time': m.get('backward_time', 0.0),
            'comm_time': m.get('comm_time', 0.0),
            'update_time': m.get('update_time', 0.0),
            'loss': m.get('loss', 0.0),
            'train_acc': m.get('train_acc', 0.0),
            'val_f1': m.get('val_f1', 0.0),
            'test_f1': m.get('test_f1', 0.0),
        })

    df = pd.DataFrame(rows)
    # Add peak memory as a column (same for all epochs in a run)
    df['peak_memory_mb'] = get_peak_memory_mb()
    df['config'] = config_name

    path = os.path.join(results_dir, f"{config_name}_metrics.csv")
    df.to_csv(path, index=False)
    print(f"[Metrics] Saved to {path}")
    return path


def save_benchmark_summary(
    all_run_metrics: Dict[str, List[List[Dict]]],
    results_dir: str = None,
) -> str:
    """
    Save aggregated benchmark summary across all configurations and repeats.

    Args:
        all_run_metrics: {config_name: [run1_metrics, run2_metrics, ...]}
            where each run is a list of per-epoch dicts.

    Returns path to saved CSV.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    for config_name, runs in all_run_metrics.items():
        # Compute per-run summary: average epoch time, total time, final metrics
        run_summaries = []
        for run_metrics in runs:
            avg_epoch_time = np.mean([m['epoch_time'] for m in run_metrics])
            total_time = sum(m['epoch_time'] for m in run_metrics)
            avg_comm_time = np.mean([m['comm_time'] for m in run_metrics])
            final_test_f1 = run_metrics[-1].get('test_f1', 0.0)
            final_val_f1 = run_metrics[-1].get('val_f1', 0.0)
            final_loss = run_metrics[-1].get('loss', 0.0)
            final_train_acc = run_metrics[-1].get('train_acc', 0.0)

            run_summaries.append({
                'avg_epoch_time': avg_epoch_time,
                'total_time': total_time,
                'avg_comm_time': avg_comm_time,
                'test_f1': final_test_f1,
                'val_f1': final_val_f1,
                'loss': final_loss,
                'train_acc': final_train_acc,
            })

        # Aggregate across runs
        rows.append({
            'config': config_name,
            'avg_epoch_time_mean': np.mean([s['avg_epoch_time'] for s in run_summaries]),
            'avg_epoch_time_std': np.std([s['avg_epoch_time'] for s in run_summaries]),
            'total_time_mean': np.mean([s['total_time'] for s in run_summaries]),
            'total_time_std': np.std([s['total_time'] for s in run_summaries]),
            'avg_comm_time_mean': np.mean([s['avg_comm_time'] for s in run_summaries]),
            'avg_comm_time_std': np.std([s['avg_comm_time'] for s in run_summaries]),
            'test_f1_mean': np.mean([s['test_f1'] for s in run_summaries]),
            'test_f1_std': np.std([s['test_f1'] for s in run_summaries]),
            'val_f1_mean': np.mean([s['val_f1'] for s in run_summaries]),
            'loss_mean': np.mean([s['loss'] for s in run_summaries]),
            'train_acc_mean': np.mean([s['train_acc'] for s in run_summaries]),
            'peak_memory_mb': get_peak_memory_mb(),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(results_dir, "benchmark_summary.csv")
    df.to_csv(path, index=False)
    print(f"[Metrics] Benchmark summary saved to {path}")
    return path
