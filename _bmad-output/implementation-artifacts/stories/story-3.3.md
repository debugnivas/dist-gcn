# Story 3.3: Performance Benchmarking

Status: ready-for-dev

## Story

As a developer,
I want systematic benchmark runs across all configurations on the Reddit dataset,
so that I can quantify speedup, communication overhead, and accuracy trade-offs empirically.

## Acceptance Criteria

1. Runs the full experiment matrix:
   | Configuration | Workers/Partitions | Dataset |
   |---|---|---|
   | Sequential (baseline) | 1 (no Spark) | Reddit |
   | Spark local[1] | 1 partition | Reddit |
   | Spark local[2] | 2 partitions | Reddit |
   | Spark local[4] | 4 partitions | Reddit |
   | Spark local[8] | 8 partitions | Reddit |
   | Spark local[16] | 16 partitions | Reddit |
2. Metrics collected per configuration: total training time (N epochs), average per-epoch time, computation vs. communication time breakdown (Spark only), final test F1-score, peak memory usage.
3. Each configuration is run at least 3 times.
4. Reports mean and standard deviation across repeated runs.
5. All results saved to CSV files in `results/`.

## Tasks / Subtasks

- [ ] Task 1: Create sequential benchmark runner (AC: #1, #2, #5)
  - [ ] Create `benchmarks/run_sequential.py`
  - [ ] Load Reddit dataset, run sequential GCN, collect all metrics from Story 3.1 infrastructure
  - [ ] Save results to `results/sequential_metrics.csv`
- [ ] Task 2: Create Spark benchmark runner (AC: #1, #2, #5)
  - [ ] Create `benchmarks/run_spark.py`
  - [ ] Accept N (worker count) as CLI argument
  - [ ] For each N in {1, 2, 4, 8, 16}: initialize SparkSession with `local[N]`, run distributed GCN, collect metrics
  - [ ] Save results to `results/spark_{N}_metrics.csv`
- [ ] Task 3: Implement repeated runs with statistics (AC: #3, #4)
  - [ ] Run each configuration 3 times with different random seeds (but same hyperparameters)
  - [ ] Compute mean and std for each metric across runs
  - [ ] Save aggregated statistics to `results/benchmark_summary.csv`
- [ ] Task 4: Create comparison aggregator (AC: #4, #5)
  - [ ] Create `benchmarks/compare_results.py`
  - [ ] Load all per-run CSV files from `results/`
  - [ ] Compute: speedup = T_sequential_mean / T_spark_N_mean for each N
  - [ ] Compute: communication overhead percentage per configuration
  - [ ] Output consolidated comparison CSV

## Dev Notes

- **Target files:** `benchmarks/run_sequential.py`, `benchmarks/run_spark.py`, `benchmarks/compare_results.py`
- **Hyperparameters for benchmarks:** Use identical settings across all runs: L=2, d=128, same learning rate, same number of epochs. Only the parallelism level changes.
- **Repeated runs:** Use seeds [42, 123, 456] (or similar) for the 3 runs. Record the seed in the CSV.
- **Runtime estimate:** On 16 cores / 64GB RAM, expect ~30-60 min total for all configurations.
- **Spark overhead:** `local[1]` will be slower than pure sequential due to Spark framework overhead. This is expected and should be noted.
- **Memory tracking:** Reset peak memory tracking between runs if using `resource.getrusage`.

### Project Structure Notes

- `benchmarks/run_sequential.py` — Sequential baseline runner
- `benchmarks/run_spark.py` — Spark GCN runner (parameterized by N)
- `benchmarks/compare_results.py` — Results aggregation and comparison
- `results/` — All CSV output
- Depends on: Stories 2.1, 2.3, 3.1

### References

- [Source: _bmad-output/planning-artifacts/requirements.md#REQ-P3-002]
