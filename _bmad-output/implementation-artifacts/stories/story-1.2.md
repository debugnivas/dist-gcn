# Story 1.2: Graph Partitioning

Status: ready-for-dev

## Story

As a developer,
I want a graph partitioner that splits the full graph into P balanced sub-graphs,
so that the distributed GCN can process independent partitions in parallel via Spark RDDs.

## Acceptance Criteria

1. Supports hash-based partitioning as the primary method: `partition_id = node_id % P`.
2. Each partition contains approximately |V|/P nodes and their associated edges (intra-partition edges).
3. Cross-partition (boundary) edges are identified and tracked separately for communication cost measurement.
4. Partitioning output is compatible with Spark RDD creation (e.g., list of per-partition data dicts or tuples).
5. Reports partition quality metrics: balance ratio (max_partition_size / avg_partition_size) and edge-cut percentage (cross-partition edges / total edges).

## Tasks / Subtasks

- [ ] Task 1: Implement hash-based node assignment (AC: #1, #2)
  - [ ] Assign each node to `node_id % P`
  - [ ] Group nodes into P partitions
  - [ ] Associate each partition with its subset of node features and labels
- [ ] Task 2: Partition edges (AC: #2, #3)
  - [ ] For each edge (u, v): if same partition, add as intra-partition edge; else mark as cross-partition boundary edge
  - [ ] Build per-partition local adjacency structures (sparse matrix or adjacency list)
  - [ ] Build boundary edge data structure mapping (source_partition, target_partition) -> edge list
- [ ] Task 3: Compute partition quality metrics (AC: #5)
  - [ ] Balance ratio: `max(partition_sizes) / mean(partition_sizes)`
  - [ ] Edge-cut percentage: `num_cross_partition_edges / total_edges * 100`
  - [ ] Log/return these metrics
- [ ] Task 4: Spark-compatible output format (AC: #4)
  - [ ] Return partitions as a list of dicts: `[{node_ids, features, labels, local_adj, boundary_edges}, ...]`
  - [ ] Each partition dict contains everything needed for Spark `sc.parallelize()` distribution

## Dev Notes

- **Target file:** `src/data/partitioner.py`
- **Input:** Sparse adjacency matrix (from `loader.py`), node features, labels, partition count P.
- **Hash partitioning** is simple and deterministic — ensures reproducibility with fixed seeds.
- **Boundary edges** are critical for measuring communication cost later (REQ-P2-005). Store them as a separate data structure.
- **Memory:** When P is large, individual partitions become small. For P=16 on 233K nodes, each partition has ~14.5K nodes. Keep per-partition data as NumPy arrays, not Python lists.
- **Do NOT add Spark dependency here.** This module returns plain Python/NumPy structures that the Spark GCN layer will parallelize.

### Project Structure Notes

- File location: `src/data/partitioner.py`
- Depends on: `src/data/loader.py` (for graph data)
- Consumed by: `src/model/gcn_spark.py` (for RDD creation)

### References

- [Source: _bmad-output/planning-artifacts/requirements.md#REQ-P2-003]
