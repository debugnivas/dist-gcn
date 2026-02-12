# Story 3.1: Performance Measurement Infrastructure

Status: ready-for-dev

## Story

As a developer,
I want comprehensive timing and metric collection instrumented in both sequential and distributed implementations,
so that I can accurately compare performance across configurations.

## Acceptance Criteria

1. High-resolution wall-clock timing using `time.perf_counter` for:
   - Total training time
   - Per-epoch time
   - Forward pass time (message passing / map-reduce)
   - Backward pass time
   - Communication time (Spark only): broadcast duration + gradient aggregation duration
   - Weight update time
2. Accuracy metrics collected per-epoch: training loss, training accuracy, validation F1-score (macro), test F1-score.
3. System metrics: peak memory usage per run (via `resource` module or `psutil`).
4. All metrics saved to CSV files in `results/` for post-processing.
5. Metrics collection works identically for both sequential and Spark versions (same CSV schema).

## Tasks / Subtasks

- [ ] Task 1: Implement high-resolution timer utility (AC: #1)
  - [ ] Create `src/utils/timer.py` with context-manager or decorator-based timer using `time.perf_counter`
  - [ ] Support nested timing sections (forward, backward, communication, update within an epoch)
  - [ ] Return timing breakdown dict per epoch
- [ ] Task 2: Implement metrics collector (AC: #1, #2, #3)
  - [ ] Create `src/utils/metrics.py`
  - [ ] Speedup calculation: `T_sequential / T_parallel`
  - [ ] Communication cost: `comm_time / total_epoch_time * 100` (percentage)
  - [ ] F1-score wrapper using `sklearn.metrics.f1_score(average='macro')`
  - [ ] Peak memory tracking using `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`
- [ ] Task 3: Implement CSV output (AC: #4, #5)
  - [ ] Define unified CSV schema: `epoch, total_time, forward_time, backward_time, comm_time, update_time, loss, train_acc, val_f1, test_f1, peak_memory_mb`
  - [ ] Write results to `results/{config_name}_metrics.csv` (e.g., `sequential_metrics.csv`, `spark_4_metrics.csv`)
  - [ ] Include header row and configuration metadata as comments
- [ ] Task 4: Integrate timing into sequential GCN (AC: #1)
  - [ ] Wrap forward pass, backward pass, weight update sections with timer
  - [ ] Record `comm_time = 0` for sequential (consistent schema)
- [ ] Task 5: Integrate timing into Spark GCN (AC: #1)
  - [ ] Wrap broadcast, forward pass (map/reduce), backward pass, gradient aggregation, weight update with timer
  - [ ] `comm_time = broadcast_time + gradient_aggregation_time`

## Dev Notes

- **Target files:** `src/utils/timer.py`, `src/utils/metrics.py`
- **`time.perf_counter`** provides nanosecond-precision monotonic clock — required per the spec.
- **Spark timing caveat:** Spark actions are lazy. Timing must be placed around the action that triggers computation (e.g., `.collect()`, `.reduce()`, `.count()`), not around the transformation definitions.
- **Memory measurement on macOS:** `resource.getrusage` returns bytes on macOS (not KB like Linux). Convert to MB: `ru_maxrss / (1024 * 1024)`.
- **CSV schema must be identical** for sequential and Spark runs to enable the comparison script (Story 3.4) to load all results uniformly.

### Project Structure Notes

- `src/utils/timer.py` — Timer utility
- `src/utils/metrics.py` — Metric computation (speedup, comm cost, F1)
- `results/` — Output directory for CSV files
- Modifies: `src/model/gcn_sequential.py` (Story 2.1), `src/model/gcn_spark.py` (Story 2.3)

### References

- [Source: _bmad-output/planning-artifacts/requirements.md#REQ-P2-005]
