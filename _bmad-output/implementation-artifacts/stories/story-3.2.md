# Story 3.2: Correctness Testing

Status: ready-for-dev

## Story

As a developer,
I want unit and integration tests verifying both GCN implementations produce correct results,
so that I can confirm accuracy and equivalence before benchmarking.

## Acceptance Criteria

1. **Unit Tests — GCN Layer Forward Pass:** Given known inputs (features, adjacency, weights), verify output matches hand-computed expected values.
2. **Unit Tests — Backward Pass:** Verify gradients via numerical gradient checking (finite differences).
3. **Unit Tests — Message Aggregation:** Verify neighbor aggregation produces correct sum for known graph structures.
4. **Unit Tests — Graph Partitioning:** Verify all nodes are assigned exactly once (no duplicates, no missing), edges are correctly mapped to partitions.
5. **Integration Test — Sequential GCN:** Run sequential GCN on a small known graph (Karate Club, 34 nodes) and verify convergence and reasonable accuracy.
6. **Integration Test — Spark GCN:** Run Spark GCN on the same small graph and verify it produces equivalent results (within floating-point tolerance) to the sequential version.
7. **Equivalence Test:** For the same hyperparameters and random seed, the sequential and Spark GCN (with 1 partition) produce identical outputs within numerical tolerance of 1e-5.

## Tasks / Subtasks

- [ ] Task 1: Unit tests for GCN layer forward pass (AC: #1)
  - [ ] Create `tests/test_gcn_sequential.py`
  - [ ] Test with a 3-4 node graph with known adjacency, features, weights
  - [ ] Hand-compute expected output: `ReLU(A_hat @ X @ W)`
  - [ ] Assert output matches within 1e-6 tolerance
- [ ] Task 2: Unit tests for backward pass (AC: #2)
  - [ ] Numerical gradient checking: perturb each weight by epsilon, compute (loss+ - loss-) / (2*epsilon)
  - [ ] Compare analytical gradients from backprop with numerical gradients
  - [ ] Assert relative error < 1e-4
- [ ] Task 3: Unit tests for message aggregation (AC: #3)
  - [ ] Test with a known graph: node with 3 neighbors, verify aggregated message = sum of neighbor features
  - [ ] Test edge cases: isolated node (no neighbors), fully connected node
- [ ] Task 4: Unit tests for graph partitioning (AC: #4)
  - [ ] Create `tests/test_partitioner.py`
  - [ ] Verify: union of all partition node sets = full node set
  - [ ] Verify: no node appears in more than one partition
  - [ ] Verify: all edges are accounted for (intra-partition + cross-partition = total edges)
  - [ ] Verify: partition sizes are approximately balanced for hash-based partitioning
- [ ] Task 5: Integration test — sequential GCN on Karate Club (AC: #5)
  - [ ] Load Karate Club graph via `networkx.karate_club_graph()`
  - [ ] Train sequential GCN for 100+ epochs
  - [ ] Verify loss decreases over epochs (convergence)
  - [ ] Verify reasonable classification accuracy (> 60% on this small graph)
- [ ] Task 6: Integration test — Spark GCN on Karate Club (AC: #6)
  - [ ] Create `tests/test_gcn_spark.py`
  - [ ] Run Spark GCN on Karate Club with same hyperparameters as sequential test
  - [ ] Verify loss decreases and accuracy is reasonable
- [ ] Task 7: Equivalence test (AC: #7)
  - [ ] Run sequential GCN and Spark GCN (1 partition) with identical seed, hyperparameters
  - [ ] Assert all layer outputs match within 1e-5 after each epoch
  - [ ] Assert final predictions are identical

## Dev Notes

- **Target files:** `tests/test_gcn_sequential.py`, `tests/test_gcn_spark.py`, `tests/test_partitioner.py`, `tests/test_metrics.py`
- **Test framework:** Use `pytest` (standard Python testing).
- **Karate Club graph:** `networkx.karate_club_graph()` has 34 nodes, 78 edges, 2 communities. Create synthetic features (random or one-hot node IDs) and binary labels (community membership).
- **Numerical gradient checking:** Use `epsilon = 1e-5` for finite differences. This is the standard method to verify backprop correctness.
- **Spark tests:** Initialize a local SparkContext in test fixtures (`local[1]`). Tear down after each test to avoid port conflicts.
- **Equivalence test is critical:** If sequential and Spark (1 partition) don't match, there's a bug in the distributed implementation. This must pass before running benchmarks.

### Project Structure Notes

- `tests/test_gcn_sequential.py` — Sequential GCN unit + integration tests
- `tests/test_gcn_spark.py` — Spark GCN integration + equivalence tests
- `tests/test_partitioner.py` — Partitioning correctness tests
- `tests/test_metrics.py` — Metric calculation tests

### References

- [Source: _bmad-output/planning-artifacts/requirements.md#REQ-P3-001]
