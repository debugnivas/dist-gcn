"""
Master benchmark script that runs all configurations efficiently.
Loads the Reddit dataset once and runs sequential + Spark[4,8,16].
Outputs results to CSV and generates charts + performance report.
"""
import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment
os.environ['OMP_NUM_THREADS'] = '1'  # Single-threaded NumPy for fair benchmarks
os.environ.setdefault(
    'JAVA_HOME',
    '/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home'
)

from src.config import (
    NUM_LAYERS, HIDDEN_DIM, LEARNING_RATE, NUM_EPOCHS,
    SEEDS, NUM_REPEATS, PARTITION_COUNTS, RESULTS_DIR, DATASET_NAME,
    SPARK_DRIVER_MEMORY
)
from src.data.loader import load_reddit_dataset
from src.model.gcn_sequential import SequentialGCN
from src.model.gcn_spark import SparkGCN
from src.utils.metrics import (
    save_epoch_metrics, save_benchmark_summary, get_peak_memory_mb
)
from src.utils.visualization import generate_all_charts


def write_performance_report(all_run_metrics, results_dir, dataset_info, total_time):
    """Write a human-readable performance report to a text file."""
    path = os.path.join(results_dir, "performance_report.txt")

    with open(path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("  DISTRIBUTED GCN PERFORMANCE REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Dataset: {dataset_info['name']}\n")
        f.write(f"  Nodes: {dataset_info['num_nodes']}\n")
        f.write(f"  Edges: {dataset_info['num_edges']}\n")
        f.write(f"  Features: {dataset_info['input_dim']}\n")
        f.write(f"  Classes: {dataset_info['num_classes']}\n")
        f.write(f"  Train/Val/Test: {dataset_info['train']}/{dataset_info['val']}/{dataset_info['test']}\n\n")

        f.write(f"GCN Configuration:\n")
        f.write(f"  Layers: {NUM_LAYERS}\n")
        f.write(f"  Hidden dim: {HIDDEN_DIM}\n")
        f.write(f"  Learning rate: {LEARNING_RATE}\n")
        f.write(f"  Epochs: {NUM_EPOCHS}\n")
        f.write(f"  Repeats per config: {NUM_REPEATS}\n")
        f.write(f"  Seeds: {SEEDS[:NUM_REPEATS]}\n\n")

        f.write(f"Total benchmark time: {total_time:.1f}s ({total_time/60:.1f} min)\n")
        f.write(f"Peak memory: {get_peak_memory_mb():.1f} MB\n\n")

        # Get sequential baseline time
        seq_time = None
        if 'sequential' in all_run_metrics:
            seq_runs = all_run_metrics['sequential']
            seq_avg_times = [np.mean([m['epoch_time'] for m in run]) for run in seq_runs]
            seq_time = np.mean(seq_avg_times)

        f.write("-" * 80 + "\n")
        f.write(f"{'Config':<20} {'Epoch Time':>12} {'Std':>8} {'Speedup':>10} "
                f"{'Efficiency':>12} {'Comm %':>8} {'Test F1':>10} {'F1 Std':>8}\n")
        f.write("-" * 80 + "\n")

        for config_name in sorted(all_run_metrics.keys(),
                                   key=lambda x: (0 if x == 'sequential' else 1,
                                                  int(x.split('_')[1]) if '_' in x else 0)):
            runs = all_run_metrics[config_name]
            avg_times = [np.mean([m['epoch_time'] for m in run]) for run in runs]
            avg_comm = [np.mean([m.get('comm_time', 0) for m in run]) for run in runs]
            avg_f1 = [run[-1].get('test_f1', 0) for run in runs]

            mean_time = np.mean(avg_times)
            std_time = np.std(avg_times)
            mean_comm = np.mean(avg_comm)
            mean_f1 = np.mean(avg_f1)
            std_f1 = np.std(avg_f1)

            speedup = seq_time / mean_time if seq_time and mean_time > 0 else 1.0
            workers = int(config_name.split('_')[1]) if '_' in config_name else 1
            efficiency = speedup / workers
            comm_pct = (mean_comm / mean_time * 100) if mean_time > 0 else 0

            f.write(f"{config_name:<20} {mean_time:>10.2f}s {std_time:>8.2f} "
                    f"{speedup:>9.2f}x {efficiency:>11.4f} "
                    f"{comm_pct:>7.1f}% {mean_f1:>9.4f} {std_f1:>8.4f}\n")

        f.write("-" * 80 + "\n\n")

        # Detailed per-epoch breakdown for each config
        f.write("=" * 80 + "\n")
        f.write("  DETAILED PER-EPOCH METRICS (Run 1 of each config)\n")
        f.write("=" * 80 + "\n\n")

        for config_name in sorted(all_run_metrics.keys(),
                                   key=lambda x: (0 if x == 'sequential' else 1,
                                                  int(x.split('_')[1]) if '_' in x else 0)):
            runs = all_run_metrics[config_name]
            first_run = runs[0]

            f.write(f"\n--- {config_name} ---\n")
            f.write(f"{'Epoch':>6} {'Time':>10} {'Fwd':>10} {'Bwd':>10} "
                    f"{'Comm':>10} {'Loss':>10} {'TrainAcc':>10} {'ValF1':>10} {'TestF1':>10}\n")

            for m in first_run:
                f.write(f"{m['epoch']:>6d} {m['epoch_time']:>9.2f}s "
                        f"{m['forward_time']:>9.2f}s {m['backward_time']:>9.2f}s "
                        f"{m.get('comm_time', 0):>9.2f}s {m['loss']:>10.4f} "
                        f"{m['train_acc']:>10.4f} {m['val_f1']:>10.4f} "
                        f"{m['test_f1']:>10.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"[Report] Performance report saved to {path}")
    return path


def main():
    total_start = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    num_epochs = NUM_EPOCHS
    seeds = SEEDS[:NUM_REPEATS]
    partition_counts = PARTITION_COUNTS

    print("=" * 70)
    print("  DISTRIBUTED GCN BENCHMARK SUITE — Reddit Dataset")
    print(f"  Epochs: {num_epochs}, Repeats: {len(seeds)}, Seeds: {seeds}")
    print(f"  Spark partition counts: {partition_counts}")
    print(f"  (P=1,2 skipped — too large per partition for Spark serialization)")
    print("=" * 70)

    # --- Load dataset once ---
    print("\n[1/4] Loading Reddit dataset...")
    load_start = time.perf_counter()
    adj, features, labels, train_mask, val_mask, test_mask = load_reddit_dataset()
    input_dim = features.shape[1]
    num_classes = len(np.unique(labels))
    num_nodes = features.shape[0]
    print(f"  Dataset loaded in {time.perf_counter() - load_start:.1f}s")

    dataset_info = {
        'name': DATASET_NAME,
        'num_nodes': num_nodes,
        'num_edges': adj.nnz,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'train': int(train_mask.sum()),
        'val': int(val_mask.sum()),
        'test': int(test_mask.sum()),
    }

    all_run_metrics = {}

    # --- Sequential Benchmark ---
    print("\n[2/4] Running SEQUENTIAL baseline...")
    seq_runs = []
    for run_idx, seed in enumerate(seeds):
        print(f"\n  --- Sequential Run {run_idx + 1}/{len(seeds)} (seed={seed}) ---")
        model = SequentialGCN(
            input_dim=input_dim, hidden_dim=HIDDEN_DIM,
            output_dim=num_classes, num_layers=NUM_LAYERS,
            learning_rate=LEARNING_RATE, seed=seed,
        )
        run_start = time.perf_counter()
        epoch_metrics = model.train(
            adj, features, labels,
            train_mask, val_mask, test_mask,
            num_epochs=num_epochs, verbose=True,
        )
        run_time = time.perf_counter() - run_start
        print(f"  Run time: {run_time:.2f}s | Test F1: {epoch_metrics[-1]['test_f1']:.4f}")

        save_epoch_metrics(epoch_metrics, f"sequential_run{run_idx+1}", RESULTS_DIR)
        seq_runs.append(epoch_metrics)

    all_run_metrics['sequential'] = seq_runs
    save_epoch_metrics(seq_runs[0], "sequential", RESULTS_DIR)

    # --- Spark Benchmarks (P=4, 8, 16) ---
    print("\n[3/4] Running SPARK benchmarks...")
    for n_workers in partition_counts:
        print(f"\n{'='*60}")
        print(f"  SPARK local[{n_workers}] ({n_workers} partitions)")
        print(f"{'='*60}")

        spark_runs = []
        for run_idx, seed in enumerate(seeds):
            print(f"\n  --- Spark[{n_workers}] Run {run_idx + 1}/{len(seeds)} (seed={seed}) ---")
            model = SparkGCN(
                input_dim=input_dim, hidden_dim=HIDDEN_DIM,
                output_dim=num_classes, num_layers=NUM_LAYERS,
                learning_rate=LEARNING_RATE, seed=seed,
                num_workers=n_workers, driver_memory=SPARK_DRIVER_MEMORY,
            )
            try:
                run_start = time.perf_counter()
                epoch_metrics = model.train(
                    adj, features, labels,
                    train_mask, val_mask, test_mask,
                    num_epochs=num_epochs, verbose=True,
                )
                run_time = time.perf_counter() - run_start
                print(f"  Run time: {run_time:.2f}s | "
                      f"Test F1: {epoch_metrics[-1]['test_f1']:.4f}")

                save_epoch_metrics(
                    epoch_metrics, f"spark_{n_workers}_run{run_idx+1}", RESULTS_DIR)
                spark_runs.append(epoch_metrics)
            except Exception as e:
                print(f"  ERROR in Spark[{n_workers}] Run {run_idx+1}: {e}")
                import traceback
                traceback.print_exc()

        if spark_runs:
            all_run_metrics[f'spark_{n_workers}'] = spark_runs
            save_epoch_metrics(spark_runs[0], f"spark_{n_workers}", RESULTS_DIR)

    # --- Generate Summary, Charts, & Report ---
    print("\n[4/4] Generating summary, charts, and performance report...")
    save_benchmark_summary(all_run_metrics, RESULTS_DIR)
    generate_all_charts(RESULTS_DIR)

    total_time = time.perf_counter() - total_start
    write_performance_report(all_run_metrics, RESULTS_DIR, dataset_info, total_time)

    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"  Peak memory: {get_peak_memory_mb():.1f} MB")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
