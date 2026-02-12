"""
Story 3.4: Performance Comparison & Visualization.

Generates comparative analysis charts and summary tables from benchmark results.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR


def load_summary(results_dir: str = None) -> pd.DataFrame:
    """Load benchmark summary CSV."""
    if results_dir is None:
        results_dir = RESULTS_DIR
    path = os.path.join(results_dir, "benchmark_summary.csv")
    return pd.read_csv(path)


def plot_speedup_chart(df: pd.DataFrame, results_dir: str = None):
    """
    Line plot: Speedup vs. Number of Workers.
    Includes ideal linear speedup reference line.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    # Get sequential baseline time
    seq_row = df[df['config'] == 'sequential']
    if seq_row.empty:
        print("[Viz] No sequential baseline found. Skipping speedup chart.")
        return
    t_seq = seq_row['avg_epoch_time_mean'].values[0]

    # Spark configs
    spark_df = df[df['config'].str.startswith('spark_')].copy()
    if spark_df.empty:
        print("[Viz] No Spark results found. Skipping speedup chart.")
        return

    spark_df['workers'] = spark_df['config'].str.extract(r'spark_(\d+)').astype(int)
    spark_df = spark_df.sort_values('workers')
    spark_df['speedup'] = t_seq / spark_df['avg_epoch_time_mean']

    fig, ax = plt.subplots(figsize=(10, 6))
    workers = spark_df['workers'].values
    speedups = spark_df['speedup'].values

    # Actual speedup
    ax.plot(workers, speedups, 'bo-', linewidth=2, markersize=8, label='Actual Speedup')

    # Ideal linear speedup
    max_w = max(workers)
    ideal = np.arange(1, max_w + 1)
    ax.plot(ideal, ideal, 'r--', linewidth=1.5, alpha=0.7, label='Ideal Linear Speedup')

    ax.set_xlabel('Number of Workers (Partitions)', fontsize=12)
    ax.set_ylabel('Speedup (T_seq / T_parallel)', fontsize=12)
    ax.set_title('Speedup vs. Number of Workers', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(workers)

    plt.tight_layout()
    path = os.path.join(results_dir, "speedup_chart.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Viz] Speedup chart saved to {path}")


def plot_epoch_time_chart(df: pd.DataFrame, results_dir: str = None):
    """
    Bar chart: Average per-epoch wall-clock time across all configurations.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    fig, ax = plt.subplots(figsize=(12, 6))

    configs = df['config'].values
    times = df['avg_epoch_time_mean'].values
    stds = df['avg_epoch_time_std'].values

    x = np.arange(len(configs))
    bars = ax.bar(x, times, yerr=stds, capsize=5, color='steelblue', alpha=0.8)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Average Epoch Time (seconds)', fontsize=12)
    ax.set_title('Per-Epoch Training Time Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(results_dir, "epoch_time_chart.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Viz] Epoch time chart saved to {path}")


def plot_comm_overhead_chart(df: pd.DataFrame, results_dir: str = None):
    """
    Stacked bar chart: Computation time vs. Communication time per configuration.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    spark_df = df[df['config'].str.startswith('spark_')].copy()
    if spark_df.empty:
        print("[Viz] No Spark results. Skipping comm overhead chart.")
        return

    spark_df['workers'] = spark_df['config'].str.extract(r'spark_(\d+)').astype(int)
    spark_df = spark_df.sort_values('workers')

    fig, ax = plt.subplots(figsize=(10, 6))

    configs = spark_df['config'].values
    comm_times = spark_df['avg_comm_time_mean'].values
    total_times = spark_df['avg_epoch_time_mean'].values
    comp_times = total_times - comm_times

    x = np.arange(len(configs))
    width = 0.6

    ax.bar(x, comp_times, width, label='Computation Time', color='steelblue', alpha=0.8)
    ax.bar(x, comm_times, width, bottom=comp_times, label='Communication Time',
           color='coral', alpha=0.8)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Computation vs. Communication Time Breakdown', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "comm_overhead_chart.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Viz] Communication overhead chart saved to {path}")


def plot_accuracy_vs_speedup(df: pd.DataFrame, results_dir: str = None):
    """
    Scatter plot: F1-score vs. Speedup for each configuration.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    seq_row = df[df['config'] == 'sequential']
    if seq_row.empty:
        return
    t_seq = seq_row['avg_epoch_time_mean'].values[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in df.iterrows():
        speedup = t_seq / row['avg_epoch_time_mean'] if row['avg_epoch_time_mean'] > 0 else 1.0
        f1 = row['test_f1_mean']
        ax.scatter(speedup, f1, s=100, zorder=5)
        ax.annotate(row['config'], (speedup, f1),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel('Speedup', fontsize=12)
    ax.set_ylabel('Test F1-Score (Macro)', fontsize=12)
    ax.set_title('Accuracy vs. Speedup Trade-off', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "accuracy_vs_speedup.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Viz] Accuracy vs speedup chart saved to {path}")


def plot_scaling_efficiency(df: pd.DataFrame, results_dir: str = None):
    """
    Line plot: Efficiency (Speedup / P) vs. Number of Workers P.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    seq_row = df[df['config'] == 'sequential']
    if seq_row.empty:
        return
    t_seq = seq_row['avg_epoch_time_mean'].values[0]

    spark_df = df[df['config'].str.startswith('spark_')].copy()
    if spark_df.empty:
        return

    spark_df['workers'] = spark_df['config'].str.extract(r'spark_(\d+)').astype(int)
    spark_df = spark_df.sort_values('workers')
    spark_df['speedup'] = t_seq / spark_df['avg_epoch_time_mean']
    spark_df['efficiency'] = spark_df['speedup'] / spark_df['workers']

    fig, ax = plt.subplots(figsize=(10, 6))

    workers = spark_df['workers'].values
    efficiency = spark_df['efficiency'].values

    ax.plot(workers, efficiency, 'go-', linewidth=2, markersize=8, label='Scaling Efficiency')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Ideal Efficiency (1.0)')

    ax.set_xlabel('Number of Workers (P)', fontsize=12)
    ax.set_ylabel('Efficiency (Speedup / P)', fontsize=12)
    ax.set_title('Scaling Efficiency', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(workers)
    ax.set_ylim(0, 1.5)

    plt.tight_layout()
    path = os.path.join(results_dir, "scaling_efficiency_chart.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Viz] Scaling efficiency chart saved to {path}")


def generate_summary_table(df: pd.DataFrame, results_dir: str = None) -> pd.DataFrame:
    """
    Generate and save consolidated summary table.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    seq_row = df[df['config'] == 'sequential']
    t_seq = seq_row['avg_epoch_time_mean'].values[0] if not seq_row.empty else 1.0

    summary = df.copy()
    summary['speedup'] = t_seq / summary['avg_epoch_time_mean']
    summary['comm_overhead_pct'] = (
        summary['avg_comm_time_mean'] / summary['avg_epoch_time_mean'] * 100
    ).fillna(0)

    # Extract workers count
    summary['workers'] = summary['config'].apply(
        lambda x: int(x.split('_')[1]) if 'spark' in x else 1
    )
    summary['efficiency'] = summary['speedup'] / summary['workers']

    # Select columns for display
    display_cols = [
        'config', 'workers', 'avg_epoch_time_mean', 'avg_epoch_time_std',
        'total_time_mean', 'speedup', 'efficiency', 'comm_overhead_pct',
        'test_f1_mean', 'test_f1_std', 'peak_memory_mb'
    ]
    display_df = summary[display_cols].round(4)

    # Print to console
    print("\n" + "=" * 120)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 120)
    print(display_df.to_string(index=False))
    print("=" * 120)

    # Save
    path = os.path.join(results_dir, "summary_table.csv")
    display_df.to_csv(path, index=False)
    print(f"\n[Viz] Summary table saved to {path}")

    return display_df


def generate_all_charts(results_dir: str = None):
    """Generate all visualization charts from benchmark summary."""
    df = load_summary(results_dir)
    plot_speedup_chart(df, results_dir)
    plot_epoch_time_chart(df, results_dir)
    plot_comm_overhead_chart(df, results_dir)
    plot_accuracy_vs_speedup(df, results_dir)
    plot_scaling_efficiency(df, results_dir)
    generate_summary_table(df, results_dir)
    print("\n[Viz] All charts generated successfully.")
