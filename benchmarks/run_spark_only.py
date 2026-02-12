"""
Run only Spark benchmarks (sequential results already exist).
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault(
    'JAVA_HOME',
    '/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home'
)

from src.config import (
    NUM_LAYERS, HIDDEN_DIM, LEARNING_RATE, NUM_EPOCHS,
    SEEDS, NUM_REPEATS, PARTITION_COUNTS, RESULTS_DIR
)
from src.data.loader import load_reddit_dataset
from src.model.gcn_spark import SparkGCN
from src.utils.metrics import save_epoch_metrics, get_peak_memory_mb


def main():
    total_start = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    num_epochs = NUM_EPOCHS
    seeds = SEEDS[:NUM_REPEATS]
    partition_counts = PARTITION_COUNTS

    # Load dataset once
    print("[1/2] Loading Reddit dataset...")
    adj, features, labels, train_mask, val_mask, test_mask = load_reddit_dataset()
    input_dim = features.shape[1]
    num_classes = len(np.unique(labels))

    # Run Spark for each configuration
    print("\n[2/2] Running SPARK benchmarks...")
    for n_workers in partition_counts:
        print(f"\n{'='*50}")
        print(f"  SPARK local[{n_workers}] ({n_workers} workers)")
        print(f"{'='*50}")

        spark_runs = []
        for run_idx, seed in enumerate(seeds):
            print(f"\n  --- Spark[{n_workers}] Run {run_idx + 1}/{len(seeds)} (seed={seed}) ---")

            model = SparkGCN(
                input_dim=input_dim, hidden_dim=HIDDEN_DIM,
                output_dim=num_classes, num_layers=NUM_LAYERS,
                learning_rate=LEARNING_RATE, seed=seed,
                num_workers=n_workers, driver_memory="16g",
            )

            try:
                run_start = time.perf_counter()
                epoch_metrics = model.train(
                    adj, features, labels,
                    train_mask, val_mask, test_mask,
                    num_epochs=num_epochs, verbose=True,
                )
                run_time = time.perf_counter() - run_start
                print(f"  Run time: {run_time:.2f}s | Test F1: {epoch_metrics[-1]['test_f1']:.4f}")

                save_epoch_metrics(epoch_metrics, f"spark_{n_workers}_run{run_idx+1}", RESULTS_DIR)
                spark_runs.append(epoch_metrics)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        if spark_runs:
            save_epoch_metrics(spark_runs[0], f"spark_{n_workers}", RESULTS_DIR)

    total_time = time.perf_counter() - total_start
    print(f"\nSpark benchmarks complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Peak memory: {get_peak_memory_mb():.1f} MB")


if __name__ == "__main__":
    main()
