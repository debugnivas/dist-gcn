# Requirements Document: ML SysOps Assignment 1

## Distributed/Parallel Graph Convolutional Network (GCN) Implementation

---

## Table of Contents

1. [P0 - Problem Formulation](#p0---problem-formulation)
2. [P1 - Design](#p1---design)
3. [P1 - Implementation Details](#p1---implementation-details-environment--platform)
4. [P2 - Implementation](#p2---implementation)
5. [P3 - Testing and Demonstration](#p3---testing-and-demonstration)
6. [Deliverables](#deliverables)
7. [Environment Setup](#environment-setup)

---

## P0 - Problem Formulation

### Problem Statement

Graph Convolutional Networks (GCNs) are deep learning models that extend traditional convolution to graph-structured data, learning from both node features and edges to make predictions via neighborhood message-passing. The core computational bottleneck of GCN lies in the **message-passing phase**, where each node aggregates feature vectors from all its neighbors. This leads to two critical scalability problems:

1. **Time Scalability (Neighbor Explosion):** Every node must communicate with all its neighbors across every layer. For graphs with billions of edges, processing these sequentially results in prohibitively long training times. For example, processing 10^9 edges repeatedly is unacceptable.

2. **Space Scalability (Memory Explosion):** The model must store activations for every node at every layer for backpropagation. For a graph with 100M nodes and 3 layers, this can exceed 150 GB of RAM for node features alone, plus the adjacency structure.

### Parallelization/Distribution Goal

Parallelize and distribute the GCN training algorithm using **Apache Spark (MapReduce paradigm)** to achieve:

- **Speedup:** Near-linear reduction in training time as the number of workers/partitions increases. Measure `Speedup = T_sequential / T_parallel` for varying numbers of partitions (1, 2, 4, 8, 16).
- **Communication Cost:** Quantify the overhead introduced by parameter synchronization and cross-partition edge data transfer. Communication cost measured as wall-clock time spent in synchronization vs. computation.
- **Response Time:** Measure per-epoch wall-clock time for both sequential and distributed versions.
- **Accuracy Preservation:** The distributed implementation must achieve comparable prediction accuracy (within 1-2% F1-score) to the sequential baseline.

### Expected Outcomes

| Metric | Sequential Baseline | Distributed Target |
|--------|--------------------|--------------------|
| Training Time (per epoch) | Baseline T_seq | T_seq / P (ideal), measured actual |
| Speedup | 1.0x | Near-linear with diminishing returns |
| Communication Overhead | 0 | Measured as % of total epoch time |
| Prediction Accuracy (F1) | Baseline F1 | Within 1-2% of baseline |
| Memory per Worker | Full graph in memory | ~1/P of graph per worker |

---

## P1 - Design

### Algorithm (Sequential GCN)

The sequential GCN algorithm as defined in the assignment:

```
Input: Full Graph G(V,E), Features X, Labels Y, Learning Rate eta, Weights W(1)...W(L)

for epoch = 1 to MAX_EPOCHS:
    H(0) <- X
    // FORWARD PASS
    for layer l = 1 to L:
        for each node v in V:
            N(v) <- get_neighbours(v)
            msg_agg <- zero vector
            for each neighbor u in N(v):
                msg_agg <- msg_agg + H_u(l-1)
            H_v(l) <- sigma(msg_agg . W(l))
    // LOSS & BACKWARD PASS
    Loss <- L(H(L), Y)
    Compute Gradients nabla_W(l) via Backpropagation
    // WEIGHT UPDATE
    for layer l = 1 to L:
        W(l) <- W(l) - eta * nabla_W(l)
Return Trained Weights W
```

### Algorithm Analysis

| Step | Time Complexity | Space Complexity |
|------|----------------|-----------------|
| Forward Pass | O(L * \|V\| * (D_avg * d + d^2)) | O(L * \|V\| * d) (Activations) |
| Loss | O(\|V\| * d) | O(1) |
| Backpropagation | O(L * \|V\| * (D_avg * d + d^2)) | O(L * \|V\| * d) (Gradients) |
| Update | O(L * d^2) | O(1) |
| **Total** | **O(L * \|V\| * (D_avg * d + d^2))** | **O(L * \|V\| * d + \|E\|)** |

### Design Choices for Parallelization

The following techniques are used to parallelize the GCN:

1. **Data and Graph Partitioning:** The full graph is partitioned into P sub-graphs using a graph partitioning strategy (hash-based partitioning or METIS-style edge-cut minimization). Each worker receives its own partition G_i and corresponding features X_i.

2. **Map/Data Parallelism (Spark RDD / MapReduce):** The message-passing (forward pass) is expressed as Map and Reduce operations:
   - **Map Phase:** For each node v, the mapper emits (node_v, feature_u) pairs for every neighbor u. Each node can be mapped independently across partitions.
   - **Reduce Phase:** For each node v, the reducer aggregates all incoming messages via SUM to produce msg_agg, then applies the weight matrix and activation.

3. **Parameter Server Pattern (via Spark Broadcast):** Global weight matrices W(l) are broadcast to all workers at the start of each epoch. After local gradient computation, gradients are aggregated centrally (allreduce / tree-reduce) to update global weights.

### Distributed Algorithm

```
Phase 1: Setup & Partitioning
  G <- Load_Full_Graph()
  G1, G2, ..., GP <- PARTITION_GRAPH(G, P)
  Distribute Gi to Worker i (along with local Features Xi)
  Initialize Global Weights W on Parameter Server (Driver)

Phase 2: Distributed Training Loop (on Worker i)
  for epoch = 1 to MAX_EPOCHS:
    // Step 2.1: Pull Weights (Broadcast from Driver)
    W_local <- Broadcast(W)
    H(0) <- Xi

    // Step 2.2: Local Forward Pass (Map + Reduce)
    for l = 1 to L:
      for each node v in Vi:
        // Map: Generate messages for neighbors
        msg_list <- { H_u(l-1) | u in N(v) }
        // Reduce: Aggregate messages
        msg_agg <- reduce(msg_list, SUM)
        H_v(l) <- sigma(msg_agg . W_local(l))

    // Step 2.3: Local Backward Pass
    Compute Loss using local labels Y_local
    Compute Gradients nabla_W_local via Backpropagation

    // Step 2.4: Push Gradients (Aggregate via Reduce)
    nabla_W_global <- AllReduce(nabla_W_local)
    W <- W - eta * nabla_W_global
```

### Performance Metrics (Theoretical)

**Speedup:**
```
T_seq ~ L * (|E| * d + |V| * d^2)
T_parallel = O(L * ((|E|/P) * d + (|V|/P) * d^2)) + O(L * d^2 * (P/B))
Speedup = T_seq / T_parallel
```

**Communication Cost:**
1. Parameter Server Synchronization: Each worker pulls/pushes O(L * d^2) weight data per epoch.
2. Cross-Partition Edge Data: For boundary nodes, feature vectors (size d) must be exchanged between partitions. Cost depends on edge-cut quality.

---

## P1 - Implementation Details (Environment & Platform)

### Development Machine Specifications

| Component | Specification |
|-----------|--------------|
| CPU | Apple Silicon (arm64), 16 cores |
| RAM | 64 GB |
| OS | macOS (Darwin 24.5.0) |
| Architecture | ARM64 (Apple M-series) |

### Minimum Software Requirements

| Software | Minimum Version | Recommended Version | Purpose |
|----------|----------------|--------------------|---------| 
| **Java JDK** | 11 | 11 (Amazon Corretto) | Spark runtime engine (JVM-based) |
| **Python** | 3.9 | 3.10 - 3.12 | Primary development language |
| **Apache Spark** | 3.4.0 | 3.5.x | Distributed computing framework (MapReduce) |
| **PySpark** | 3.4.0 | 3.5.x | Python API for Apache Spark |
| **pip** | 21.0+ | latest | Python package manager |

> **Note on Python 3.14:** The machine currently has Python 3.14.1 installed. PySpark 3.5.x does **not** officially support Python 3.14. A Python 3.10-3.12 virtual environment must be created (e.g., via `pyenv` or `conda`) for Spark compatibility.

### Python Libraries Required

| Library | Version | Purpose |
|---------|---------|---------|
| `pyspark` | >= 3.4.0 | Spark MapReduce engine, RDD operations, broadcast variables |
| `numpy` | >= 1.24.0 | Matrix operations, weight initialization, activations |
| `scipy` | >= 1.10.0 | Sparse matrix support for adjacency matrices |
| `scikit-learn` | >= 1.2.0 | Metrics (F1-score, accuracy), train/test split |
| `networkx` | >= 3.0 | Graph loading, partitioning utilities, sequential baseline |
| `matplotlib` | >= 3.7.0 | Plotting performance metrics, speedup charts |
| `pandas` | >= 2.0.0 | Data loading, CSV/TSV parsing for Reddit dataset |
| `torch` (PyTorch) | >= 2.0.0 | (Optional) Sequential GCN baseline for validation |
| `torch-geometric` | >= 2.3.0 | (Optional) Reference GCN implementation for accuracy validation |

### Execution Platform

- **Local Spark Standalone Mode:** Spark will run in local mode (`local[N]`) on the development machine to simulate parallelism with N = {1, 2, 4, 8, 16} worker threads (cores).
- **Spark Master URL:** `local[1]` for sequential simulation, `local[2]`, `local[4]`, `local[8]`, `local[16]` for parallel simulations.
- Each "worker" is a Spark executor thread sharing the same JVM but processing independent RDD partitions, simulating the distributed MapReduce model.

### Project Structure

```
mlsysops-assignment/
|-- src/
|   |-- __init__.py
|   |-- config.py                   # Hyperparameters, Spark config, paths
|   |-- data/
|   |   |-- __init__.py
|   |   |-- loader.py               # Reddit dataset loading & preprocessing
|   |   |-- partitioner.py          # Graph partitioning (hash-based / balanced)
|   |-- model/
|   |   |-- __init__.py
|   |   |-- gcn_sequential.py       # Sequential (non-parallel) GCN baseline
|   |   |-- gcn_spark.py            # Spark-based distributed GCN (MapReduce)
|   |   |-- layers.py               # GCN layer logic (forward, backward)
|   |   |-- parameter_server.py     # Weight broadcast & gradient aggregation
|   |-- utils/
|   |   |-- __init__.py
|   |   |-- metrics.py              # Speedup, comm cost, F1, accuracy calculations
|   |   |-- visualization.py        # Plotting utilities
|   |   |-- timer.py                # High-resolution timing utilities
|-- tests/
|   |-- test_gcn_sequential.py      # Correctness tests for sequential GCN
|   |-- test_gcn_spark.py           # Correctness tests for Spark GCN
|   |-- test_partitioner.py         # Partition quality tests
|   |-- test_metrics.py             # Metric calculation tests
|-- benchmarks/
|   |-- run_sequential.py           # Run sequential baseline & record metrics
|   |-- run_spark.py                # Run Spark GCN with varying partitions
|   |-- compare_results.py          # Generate comparison tables and charts
|-- data/
|   |-- README.md                   # Instructions for downloading Reddit dataset
|-- results/
|   |-- (generated plots and CSVs)
|-- requirements.txt
|-- README.md
|-- run_all.sh                      # Master script to run all experiments
```

---

## P2 - Implementation

### REQ-P2-001: Sequential GCN Baseline

**Description:** Implement the full GCN algorithm (as specified in the assignment pseudocode) in pure Python/NumPy without any parallelism.

**Acceptance Criteria:**
- Implements the exact algorithm from the assignment: forward pass (message passing with neighbor aggregation), loss computation, backpropagation, weight update.
- Runs on the full Reddit dataset (or a sampled subset that fits in memory).
- Computes and logs per-epoch: training time, loss, training accuracy, validation F1-score.
- Supports configurable: number of layers (L), hidden dimension (d), learning rate (eta), epochs.
- Stores adjacency as a sparse matrix (scipy.sparse) for memory efficiency.

### REQ-P2-002: Reddit Dataset Loading and Preprocessing

**Description:** Load and preprocess the Reddit dataset for GCN training.

**Dataset:** Reddit dataset from GraphSAGE (approx. 233K nodes, 114M edges, 602 features, 41 classes). This is a large-scale graph suitable for demonstrating parallelism benefits.

**Acceptance Criteria:**
- Downloads or loads the Reddit dataset (node features, edges, labels, train/val/test splits).
- Constructs a graph representation (adjacency list or sparse adjacency matrix).
- Normalizes node features (zero-mean, unit-variance).
- Provides train/validation/test masks.
- Dataset loader is reusable by both sequential and Spark implementations.

### REQ-P2-003: Graph Partitioning

**Description:** Implement graph partitioning to split the graph into P balanced sub-graphs.

**Acceptance Criteria:**
- Supports hash-based partitioning (node_id mod P) as the primary method.
- Each partition contains approximately |V|/P nodes and their associated edges.
- Cross-partition (boundary) edges are identified and tracked for communication cost measurement.
- Partitioning output is compatible with Spark RDD creation.
- Reports partition quality metrics: balance ratio, edge-cut percentage.

### REQ-P2-004: Spark-Based Distributed GCN (MapReduce)

**Description:** Implement the distributed GCN training algorithm using PySpark, following the MapReduce paradigm described in the assignment's distributed algorithm.

**Acceptance Criteria:**
- **Graph as RDDs:** The graph is represented as Spark RDDs, with each partition corresponding to a sub-graph.
- **Map Phase:** For each node v, the map function emits (node_v, neighbor_feature) key-value pairs. Mapper processes each node independently (data parallelism).
- **Reduce Phase:** For each node v, the reduce function aggregates all neighbor features via SUM to compute msg_agg, then applies weight matrix multiplication and activation (sigma/ReLU).
- **Weight Broadcast:** Global weight matrices are broadcast to all executors using `SparkContext.broadcast()` at the start of each epoch.
- **Gradient Aggregation:** After local backward passes, gradients from all partitions are aggregated (via `rdd.treeReduce()` or `rdd.reduce()`) to compute global gradient updates.
- **Weight Update:** The driver updates global weights using aggregated gradients and broadcasts updated weights for the next epoch.
- **Configurable Parallelism:** The number of Spark partitions/workers is configurable via `local[N]` where N in {1, 2, 4, 8, 16}.
- **Epoch Loop:** Full training loop with configurable epochs, identical hyperparameters to sequential version.

### REQ-P2-005: Performance Measurement Infrastructure

**Description:** Instrument both sequential and distributed implementations with comprehensive timing and metric collection.

**Acceptance Criteria:**
- **Timing:** High-resolution wall-clock timing (time.perf_counter) for:
  - Total training time
  - Per-epoch time
  - Forward pass time (message passing / map-reduce)
  - Backward pass time
  - Communication time (broadcast + gradient aggregation) — Spark version only
  - Weight update time
- **Accuracy Metrics:** Per-epoch training loss, training accuracy, validation F1-score (macro), test F1-score.
- **System Metrics:** Peak memory usage per run.
- **Output:** All metrics saved to CSV files for post-processing.

### REQ-P2-006: Parameter Server Simulation

**Description:** Implement the parameter server pattern using Spark's driver program as the central coordinator.

**Acceptance Criteria:**
- The Spark driver holds the global weight matrices.
- At the start of each epoch, weights are broadcast to all workers via `SparkContext.broadcast()`.
- After local computation, each partition's gradients are collected and aggregated on the driver.
- Synchronous update: all workers use the same weight version per epoch (BSP - Bulk Synchronous Parallel model).

---

## P3 - Testing and Demonstration

### REQ-P3-001: Correctness Testing

**Description:** Verify that both sequential and distributed GCN implementations produce correct results.

**Acceptance Criteria:**
- **Unit Tests:**
  - Test GCN layer forward pass: Given known inputs (features, adjacency, weights), verify output matches hand-computed expected values.
  - Test backward pass: Verify gradients via numerical gradient checking (finite differences).
  - Test message aggregation: Verify neighbor aggregation produces correct sum.
  - Test graph partitioning: Verify all nodes are assigned, no duplicates, edges are correctly mapped.
- **Integration Tests:**
  - Run sequential GCN on a small known graph (e.g., Karate Club, 34 nodes) and verify convergence and reasonable accuracy.
  - Run Spark GCN on the same small graph and verify it produces equivalent results (within floating-point tolerance) to the sequential version.
- **Equivalence Test:**
  - For the same hyperparameters and random seed, the sequential and Spark GCN (with 1 partition) should produce identical outputs (within numerical tolerance of 1e-5).

### REQ-P3-002: Performance Benchmarking

**Description:** Systematically benchmark and compare sequential vs. distributed GCN performance.

**Acceptance Criteria:**
- **Experiment Matrix:**

  | Configuration | Workers/Partitions | Dataset |
  |--------------|-------------------|---------|
  | Sequential (baseline) | 1 (no Spark) | Reddit |
  | Spark local[1] | 1 partition | Reddit |
  | Spark local[2] | 2 partitions | Reddit |
  | Spark local[4] | 4 partitions | Reddit |
  | Spark local[8] | 8 partitions | Reddit |
  | Spark local[16] | 16 partitions | Reddit |

- **Metrics Collected Per Configuration:**
  - Total training time (N epochs)
  - Average per-epoch time
  - Breakdown: computation time vs. communication time (Spark only)
  - Final test F1-score
  - Peak memory usage

- **Repeated Runs:** Each configuration run at least 3 times; report mean and standard deviation.

### REQ-P3-003: Performance Comparison and Visualization

**Description:** Generate comparative analysis charts and tables.

**Acceptance Criteria:**
- **Speedup Chart:** Line plot of Speedup (y-axis) vs. Number of Workers (x-axis). Include ideal linear speedup line for reference.
- **Epoch Time Chart:** Bar chart comparing per-epoch wall-clock time across configurations.
- **Communication Overhead Chart:** Stacked bar chart showing computation time vs. communication time per configuration.
- **Accuracy vs. Speedup Trade-off:** Scatter plot or table showing F1-score vs. speedup for each configuration.
- **Scaling Efficiency Chart:** Plot of efficiency (Speedup / P) vs. number of workers P.
- **Summary Table:** Consolidated table with all metrics for all configurations.

### REQ-P3-004: Demonstration Script

**Description:** Provide a single master script that runs the complete demonstration end-to-end.

**Acceptance Criteria:**
- `run_all.sh` script that:
  1. Checks environment prerequisites (Java, Python, Spark).
  2. Downloads/prepares the Reddit dataset (if not already present).
  3. Runs sequential GCN baseline.
  4. Runs Spark GCN for each partition count (1, 2, 4, 8, 16).
  5. Generates comparison results (CSV + charts).
  6. Prints summary to console.
- Script is idempotent (safe to re-run).
- Estimated total runtime: ~30-60 minutes on the target machine.

---

## Deliverables

### DEL-001: GitHub Repository

- All source code committed and pushed to GitHub.
- Clean commit history with meaningful messages.
- Repository link included in the assignment facing sheet.

### DEL-002: Code as PDF

- All source code files exported to a single PDF file.
- Code must be readable (proper formatting, syntax highlighting if possible).

### DEL-003: Report

The report must contain the following sections (in order):

| Section | Content |
|---------|---------|
| **Abstract** | Brief overview of the project: GCN parallelization using Spark MapReduce. |
| **Introduction** | Background on GNNs/GCNs, motivation for parallelization, scope of the project. |
| **Problem Statement** | Formal statement of the scalability problem (time + space) in GCN training. |
| **Algorithm** | The sequential GCN pseudocode as given in the assignment. |
| **Algorithm Analysis** | Time and space complexity table for the sequential algorithm. |
| **Design** | Design choices: graph partitioning, MapReduce pattern, parameter server. |
| **Distributed Algorithm** | The distributed GCN pseudocode with Spark MapReduce. |
| **Performance Metrics** | Theoretical analysis of speedup and communication cost formulas. |
| **Results** | Empirical results: tables, speedup charts, accuracy comparison, communication overhead. |
| **Discussion** | Analysis of results: where speedup is achieved, bottlenecks, deviation from ideal, practical insights. |
| **Conclusion** | Summary of findings, lessons learned, future work. |

> **Note:** The report sections through "Performance Metrics" are already drafted in the existing assignment document. The "Results", "Discussion", and "Conclusion" sections need to be completed after implementation and benchmarking.

---

## Environment Setup

### Step-by-Step Setup Instructions

#### 1. Install Python 3.10-3.12 (Required for PySpark Compatibility)

```bash
# Using pyenv (recommended)
brew install pyenv
pyenv install 3.12.8
pyenv local 3.12.8

# OR using conda
conda create -n gcn-spark python=3.12
conda activate gcn-spark
```

#### 2. Verify Java 11

```bash
java -version
# Expected: openjdk version "11.x.x" (Amazon Corretto 11 already installed)
```

Java 11 is already installed on the machine (Amazon Corretto 11.0.28). No additional setup needed.

#### 3. Set JAVA_HOME

```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home
```

#### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pyspark>=3.4.0,<3.6.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
networkx>=3.0
matplotlib>=3.7.0
pandas>=2.0.0
```

#### 5. Verify Spark Installation

```bash
python -c "from pyspark import SparkContext; sc = SparkContext('local[2]', 'test'); print(f'Spark running with {sc.defaultParallelism} cores'); sc.stop()"
```

#### 6. Download Reddit Dataset

```bash
# The dataset loader script will handle download automatically
# Alternatively, manually download from:
# https://data.dgl.ai/dataset/reddit.zip
# Extract to data/ directory
```

### Environment Variables

```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home
export SPARK_HOME=$(python -c "import pyspark; print(pyspark.__path__[0])")
export PYSPARK_PYTHON=$(which python3)
```

---

## Constraints and Assumptions

1. **Spark Only for Parallelism:** The distributed/parallel implementation must use Apache Spark (PySpark) with MapReduce operations (map, reduceByKey, treeReduce). No other parallel frameworks (Dask, Ray, MPI) are permitted.
2. **Local Simulation:** All parallelism is simulated on a single machine using Spark's `local[N]` mode with N threads.
3. **Dataset:** Reddit dataset (~233K nodes, ~114M edges) is the primary benchmark. A smaller graph (Karate Club or Cora) is used for correctness testing.
4. **GCN Architecture:** 2-layer GCN (L=2), hidden dimension d=128, ReLU activation, cross-entropy loss, as per the assignment algorithm.
5. **Comparison Required:** Every benchmark must include the sequential (no-Spark) baseline for direct comparison.
6. **Reproducibility:** All experiments must be reproducible with fixed random seeds.
7. **No GPU Required:** All computation is CPU-based to fairly measure parallelization benefits.
