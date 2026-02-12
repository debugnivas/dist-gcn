# Story 2.2: Parameter Server Simulation

Status: ready-for-dev

## Story

As a developer,
I want a parameter server module using Spark's driver as the central coordinator,
so that global weights are broadcast to workers and gradients are synchronously aggregated each epoch (BSP model).

## Acceptance Criteria

1. The Spark driver holds the global weight matrices W(1)...W(L).
2. At the start of each epoch, weights are broadcast to all workers via `SparkContext.broadcast()`.
3. After local computation, each partition's gradients are collected and aggregated (summed/averaged) on the driver.
4. Implements synchronous update: all workers use the same weight version per epoch (Bulk Synchronous Parallel model).
5. Previous broadcast variables are properly unpersisted/destroyed before creating new ones each epoch to prevent memory leaks.

## Tasks / Subtasks

- [ ] Task 1: Implement weight broadcast mechanism (AC: #1, #2)
  - [ ] Function `broadcast_weights(sc, weights_dict)` -> returns broadcast variable
  - [ ] Weights dict structure: `{layer_idx: numpy_array}` for each layer
  - [ ] Use `sc.broadcast(weights_dict)` to send to all executors
- [ ] Task 2: Implement gradient aggregation (AC: #3)
  - [ ] Function `aggregate_gradients(gradient_rdd)` -> returns global gradient dict
  - [ ] Use `rdd.treeReduce()` or `rdd.reduce()` to sum local gradients across partitions
  - [ ] Average gradients by number of partitions (or total nodes) for proper scaling
- [ ] Task 3: Implement weight update on driver (AC: #4)
  - [ ] Function `update_weights(weights, gradients, lr)` -> returns updated weights
  - [ ] SGD update: `W(l) = W(l) - eta * global_gradient(l)`
- [ ] Task 4: Implement broadcast variable lifecycle management (AC: #5)
  - [ ] Unpersist/destroy previous epoch's broadcast variable before creating new one
  - [ ] Prevent Spark broadcast memory accumulation over many epochs

## Dev Notes

- **Target file:** `src/model/parameter_server.py`
- **Broadcast variables** in Spark are read-only on workers. The driver creates a new broadcast each epoch with updated weights.
- **Gradient aggregation:** `treeReduce` is preferred over `reduce` for large partition counts — it uses a tree topology reducing communication rounds from O(P) to O(log P).
- **BSP semantics:** Every worker must finish its local computation before gradients are aggregated and new weights are broadcast. Spark's synchronous execution model naturally enforces this within an epoch.
- **Memory management:** Calling `.unpersist()` on the old broadcast variable is critical. Without it, Spark accumulates broadcast data in memory across epochs.
- **This module is consumed by `gcn_spark.py`** (Story 2.3) which orchestrates the training loop.

### Project Structure Notes

- File location: `src/model/parameter_server.py`
- Depends on: PySpark (`SparkContext`, broadcast)
- Consumed by: `src/model/gcn_spark.py`

### References

- [Source: _bmad-output/planning-artifacts/requirements.md#REQ-P2-006]
- [Source: _bmad-output/planning-artifacts/requirements.md#P1 - Design, Distributed Algorithm Phase 2 Steps 2.1, 2.4]
