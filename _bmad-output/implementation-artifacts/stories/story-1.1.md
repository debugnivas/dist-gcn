# Story 1.1: Reddit Dataset Loading & Preprocessing

Status: ready-for-dev

## Story

As a developer,
I want a reusable dataset loader that loads and preprocesses the Reddit dataset,
so that both sequential and Spark GCN implementations can consume a clean, normalized graph representation.

## Acceptance Criteria

1. Downloads or loads the Reddit dataset from GraphSAGE (~233K nodes, ~114M edges, 602 features, 41 classes).
2. Constructs a graph representation as a sparse adjacency matrix (scipy.sparse CSR format) and/or adjacency list.
3. Normalizes node features to zero-mean, unit-variance (per-feature standardization).
4. Provides train/validation/test masks (boolean arrays) matching the original dataset splits.
5. The loader is reusable by both the sequential GCN (`gcn_sequential.py`) and the Spark GCN (`gcn_spark.py`).
6. Handles the case where data is not yet downloaded — auto-downloads from `https://data.dgl.ai/dataset/reddit.zip` and extracts to `data/`.

## Tasks / Subtasks

- [ ] Task 1: Implement Reddit dataset downloader (AC: #6)
  - [ ] Download `reddit.zip` from DGL if not present in `data/`
  - [ ] Extract and verify file integrity (node features, edges, labels, splits)
- [ ] Task 2: Implement graph construction (AC: #2)
  - [ ] Parse edge list into scipy.sparse CSR adjacency matrix
  - [ ] Optionally build adjacency list dict for sequential neighbor lookups
- [ ] Task 3: Implement feature preprocessing (AC: #3)
  - [ ] Load 602-dimensional node feature matrix (numpy array)
  - [ ] Normalize features: zero-mean, unit-variance per feature column
- [ ] Task 4: Implement label and mask loading (AC: #1, #4)
  - [ ] Load 41-class node labels as integer array
  - [ ] Load or reconstruct train/val/test boolean masks
- [ ] Task 5: Create unified loader API (AC: #5)
  - [ ] Single function/class returning: `adj_sparse`, `features`, `labels`, `train_mask`, `val_mask`, `test_mask`
  - [ ] Ensure returned types are compatible with both NumPy (sequential) and Spark (distributed) consumers

## Dev Notes

- **Target file:** `src/data/loader.py`
- **Dataset:** Reddit from GraphSAGE — files typically include `reddit_data.npz` (features + labels) and `reddit_graph.npz` (adjacency).
- **Sparse format:** Use `scipy.sparse.csr_matrix` for the adjacency matrix. This is critical for memory efficiency on the 233K-node graph.
- **Normalization:** Use sklearn `StandardScaler` or manual `(X - mean) / std` per feature column. Handle zero-variance columns gracefully.
- **No Spark dependency in this module.** The loader should return pure NumPy/SciPy objects. The Spark GCN will convert to RDDs downstream.

### Project Structure Notes

- File location: `src/data/loader.py`
- Data directory: `data/` (with a `data/README.md` for download instructions)
- Must also create `src/data/__init__.py` and `src/__init__.py` if not present

### References

- [Source: _bmad-output/planning-artifacts/requirements.md#REQ-P2-002]
- Dataset URL: https://data.dgl.ai/dataset/reddit.zip

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
