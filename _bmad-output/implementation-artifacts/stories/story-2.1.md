# Story 2.1: Sequential GCN Baseline

Status: ready-for-dev

## Story

As a developer,
I want a full sequential GCN implementation in pure Python/NumPy following the exact assignment pseudocode,
so that it serves as the correctness and performance baseline for the distributed version.

## Acceptance Criteria

1. Implements the exact algorithm from the assignment pseudocode: forward pass (message passing with neighbor aggregation via SUM), loss computation (cross-entropy), backpropagation, weight update (SGD).
2. Runs on the full Reddit dataset (or a sampled subset that fits in memory).
3. Computes and logs per-epoch: training time, loss, training accuracy, validation F1-score.
4. Supports configurable hyperparameters: number of layers L (default 2), hidden dimension d (default 128), learning rate eta, number of epochs.
5. Stores adjacency as a scipy.sparse matrix for memory efficiency.
6. Uses ReLU activation function (sigma) between layers.
7. Supports reproducibility via fixed random seed for weight initialization.

## Tasks / Subtasks

- [ ] Task 1: Implement GCN layer logic (AC: #1, #6)
  - [ ] Create `src/model/layers.py` with GCN layer: forward pass (`H_out = sigma(A_hat @ H_in @ W)`)
  - [ ] Implement ReLU activation
  - [ ] Implement backward pass for the layer (gradient of loss w.r.t. weights and input)
  - [ ] Message aggregation = sparse matrix multiply (A_hat @ H) where A_hat is the adjacency matrix
- [ ] Task 2: Implement full sequential training loop (AC: #1, #2, #4, #7)
  - [ ] Create `src/model/gcn_sequential.py`
  - [ ] Initialize weight matrices W(1)...W(L) with Xavier/random init, seeded for reproducibility
  - [ ] Forward pass: for each layer, compute `H(l) = ReLU(A_sparse @ H(l-1) @ W(l))` (last layer: no ReLU, softmax for classification)
  - [ ] Loss: cross-entropy loss on training nodes
  - [ ] Backward pass: compute gradients dL/dW(l) for each layer via chain rule
  - [ ] Weight update: `W(l) = W(l) - eta * dW(l)`
- [ ] Task 3: Implement loss and accuracy computation (AC: #3)
  - [ ] Cross-entropy loss for multi-class (41 classes)
  - [ ] Training accuracy on train mask
  - [ ] Validation F1-score (macro) using sklearn.metrics.f1_score
- [ ] Task 4: Implement per-epoch logging (AC: #3)
  - [ ] Log: epoch number, training time, loss, train accuracy, val F1-score
  - [ ] Print to console and optionally write to CSV

## Dev Notes

- **Target files:** `src/model/gcn_sequential.py`, `src/model/layers.py`
- **Architecture:** 2-layer GCN (L=2), hidden dim d=128, 602 input features, 41 output classes. Per assignment constraints.
- **Sparse matmul:** Use `scipy.sparse.csr_matrix @ numpy.ndarray` for the adjacency-feature multiplication. This is the key performance operation.
- **Backpropagation:** Must be manual (no autograd). Compute gradients of cross-entropy → softmax → linear → ReLU → sparse matmul chain.
- **Adjacency normalization:** Use symmetric normalization `D^{-1/2} A D^{-1/2}` or row normalization `D^{-1} A` as per standard GCN. Add self-loops (A + I) before normalization.
- **No Spark, no PyTorch, no external ML frameworks** for this implementation — pure NumPy/SciPy only.
- **Config:** Use `src/config.py` for hyperparameter defaults.

### Project Structure Notes

- `src/model/layers.py` — GCN layer forward/backward logic (shared by sequential and potentially Spark)
- `src/model/gcn_sequential.py` — Full sequential training loop
- `src/config.py` — Hyperparameter configuration
- Depends on: `src/data/loader.py` (Story 1.1)

### References

- [Source: _bmad-output/planning-artifacts/requirements.md#REQ-P2-001]
- [Source: _bmad-output/planning-artifacts/requirements.md#P1 - Design, Algorithm (Sequential GCN)]
