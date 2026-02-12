# Story 2.3: Spark-Based Distributed GCN (MapReduce)

Status: ready-for-dev

## Story

As a developer,
I want a distributed GCN training implementation using PySpark MapReduce,
so that GCN training scales across multiple partitions/workers following the assignment's distributed algorithm.

## Acceptance Criteria

1. The graph is represented as Spark RDDs, with each partition corresponding to a sub-graph from the partitioner (Story 1.2).
2. **Map Phase:** For each node v, the map function emits `(node_v, neighbor_feature)` key-value pairs. Each node is processed independently (data parallelism).
3. **Reduce Phase:** For each node v, the reduce function aggregates all neighbor features via SUM to compute `msg_agg`, then applies weight matrix multiplication and activation (ReLU).
4. **Weight Broadcast:** Global weight matrices are broadcast to all executors using `SparkContext.broadcast()` at the start of each epoch (via parameter_server.py).
5. **Gradient Aggregation:** After local backward passes, gradients from all partitions are aggregated via `rdd.treeReduce()` or `rdd.reduce()` to compute global gradient updates.
6. The driver updates global weights using aggregated gradients and broadcasts updated weights for the next epoch.
7. The number of Spark partitions/workers is configurable via `local[N]` where N in {1, 2, 4, 8, 16}.
8. Full training loop with configurable epochs, using identical hyperparameters to the sequential version (L=2, d=128, ReLU, cross-entropy, SGD).
9. Computes and logs per-epoch: training time, loss, training accuracy, validation F1-score (same as sequential for comparison).

## Tasks / Subtasks

- [ ] Task 1: Initialize Spark session and load graph as RDDs (AC: #1, #7)
  - [ ] Create SparkSession with `local[N]` master URL (N configurable)
  - [ ] Use `partitioner.py` output to create RDD with P partitions
  - [ ] Each RDD element: `(partition_id, {node_ids, features, local_adj, boundary_edges, labels})`
  - [ ] Repartition RDD to P partitions
- [ ] Task 2: Implement distributed forward pass (Map + Reduce) (AC: #2, #3)
  - [ ] **Map:** For each partition, for each node v, emit `(v, H_u(l-1))` for all neighbors u of v
  - [ ] **Reduce:** For each node v, SUM all incoming neighbor features to get `msg_agg`
  - [ ] Apply: `H_v(l) = ReLU(msg_agg @ W(l))` using broadcast weights
  - [ ] Handle intra-partition edges locally; cross-partition boundary nodes need feature exchange
  - [ ] Implement for L=2 layers sequentially within each epoch
- [ ] Task 3: Implement distributed backward pass (AC: #5)
  - [ ] Compute cross-entropy loss on local training nodes
  - [ ] Compute local gradients dW(l) for each layer via backpropagation
  - [ ] Each partition produces its own local gradient dict
- [ ] Task 4: Integrate parameter server (AC: #4, #5, #6)
  - [ ] Use `parameter_server.py` to broadcast weights at epoch start
  - [ ] Use `parameter_server.py` to aggregate gradients via treeReduce
  - [ ] Update weights on driver and broadcast for next epoch
- [ ] Task 5: Implement full distributed training loop (AC: #8, #9)
  - [ ] Epoch loop: broadcast weights -> forward pass (map/reduce) -> backward pass -> aggregate gradients -> update weights
  - [ ] Log per-epoch metrics: time, loss, accuracy, F1-score
  - [ ] Ensure hyperparameters match sequential version exactly

## Dev Notes

- **Target file:** `src/model/gcn_spark.py`
- **RDD design choice:** Two approaches for map-reduce on graphs:
  1. **Partition-level:** Each RDD partition processes its sub-graph as a unit (more efficient, less shuffle).
  2. **Node-level:** Each RDD element is a single node (more granular, more Spark overhead).
  - **Recommended: Partition-level** using `mapPartitions()` for efficiency. Each partition performs local sparse matmul on its sub-graph, then boundary features are exchanged.
- **Cross-partition edges:** For boundary nodes, features from other partitions are needed. Options:
  - Replicate boundary node features during partitioning (halo nodes) — simpler, uses more memory.
  - Exchange boundary features via a separate RDD join/cogroup step each layer — more complex, more "MapReduce-like".
  - Document the chosen approach in code comments.
- **Spark session configuration:** Set `spark.driver.memory`, `spark.executor.memory` appropriately for the 64GB machine.
- **Same random seed** as sequential for weight initialization to ensure comparable starting point.

### Project Structure Notes

- File location: `src/model/gcn_spark.py`
- Depends on: `src/data/partitioner.py` (Story 1.2), `src/model/parameter_server.py` (Story 2.2), `src/model/layers.py` (Story 2.1)
- Consumed by: `benchmarks/run_spark.py` (Story 3.3)

### References

- [Source: _bmad-output/planning-artifacts/requirements.md#REQ-P2-004]
- [Source: _bmad-output/planning-artifacts/requirements.md#P1 - Design, Distributed Algorithm Phase 2]
