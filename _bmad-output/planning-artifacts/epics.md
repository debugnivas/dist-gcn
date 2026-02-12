---
stepsCompleted: []
inputDocuments:
  - requirements.md
---

# ML SysOps Assignment 1 - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for the Distributed/Parallel GCN Implementation project, decomposing the requirements into implementable, developer-ready stories.

## Requirements Inventory

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-P2-001 | Sequential GCN Baseline | P2 |
| REQ-P2-002 | Reddit Dataset Loading and Preprocessing | P2 |
| REQ-P2-003 | Graph Partitioning | P2 |
| REQ-P2-004 | Spark-Based Distributed GCN (MapReduce) | P2 |
| REQ-P2-005 | Performance Measurement Infrastructure | P2 |
| REQ-P2-006 | Parameter Server Simulation | P2 |

### Testing & Demonstration Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-P3-001 | Correctness Testing | P3 |
| REQ-P3-002 | Performance Benchmarking | P3 |
| REQ-P3-003 | Performance Comparison and Visualization | P3 |
| REQ-P3-004 | Demonstration Script | P3 |

### FR Coverage Map

| Requirement | Epic | Story |
|-------------|------|-------|
| REQ-P2-002 | Epic 1 | Story 1.1 |
| REQ-P2-003 | Epic 1 | Story 1.2 |
| REQ-P2-001 | Epic 2 | Story 2.1 |
| REQ-P2-006 | Epic 2 | Story 2.2 |
| REQ-P2-004 | Epic 2 | Story 2.3 |
| REQ-P2-005 | Epic 3 | Story 3.1 |
| REQ-P3-001 | Epic 3 | Story 3.2 |
| REQ-P3-002 | Epic 3 | Story 3.3 |
| REQ-P3-003 | Epic 3 | Story 3.4 |
| REQ-P3-004 | Epic 3 | Story 3.5 |

## Epic List

| Epic | Title | Stories | Dependencies |
|------|-------|---------|--------------|
| Epic 1 | Data Pipeline & Graph Infrastructure | 1.1, 1.2 | None |
| Epic 2 | GCN Core Implementation | 2.1, 2.2, 2.3 | Epic 1 |
| Epic 3 | Instrumentation, Testing & Benchmarking | 3.1, 3.2, 3.3, 3.4, 3.5 | Epic 1, Epic 2 |

---

## Epic 1: Data Pipeline & Graph Infrastructure

**Goal:** Build the data loading, preprocessing, and graph partitioning foundation that both the sequential and distributed GCN implementations depend on.

### Story 1.1: Reddit Dataset Loading & Preprocessing

As a developer,
I want a reusable dataset loader that loads and preprocesses the Reddit dataset,
so that both sequential and Spark GCN implementations can consume a clean, normalized graph representation.

**Acceptance Criteria:**

**Given** the Reddit dataset files exist (or a download URL is provided)
**When** the loader is invoked
**Then** it downloads/loads the Reddit dataset (233K nodes, 114M edges, 602 features, 41 classes), constructs a graph (adjacency list or sparse adjacency matrix via scipy.sparse), normalizes node features (zero-mean, unit-variance), and provides train/validation/test masks.
**And** the loader is reusable by both sequential and Spark implementations.

### Story 1.2: Graph Partitioning

As a developer,
I want a graph partitioner that splits the full graph into P balanced sub-graphs,
so that the distributed GCN can process independent partitions in parallel.

**Acceptance Criteria:**

**Given** a loaded graph and a partition count P
**When** the partitioner runs
**Then** it produces P sub-graphs using hash-based partitioning (node_id mod P), each containing ~|V|/P nodes with associated edges, identifies and tracks cross-partition boundary edges, outputs partitions compatible with Spark RDD creation, and reports partition quality metrics (balance ratio, edge-cut percentage).

---

## Epic 2: GCN Core Implementation

**Goal:** Implement the sequential GCN baseline, the parameter server pattern, and the Spark-based distributed GCN following the MapReduce paradigm specified in the assignment.

### Story 2.1: Sequential GCN Baseline

As a developer,
I want a full sequential GCN implementation in pure Python/NumPy,
so that it serves as the correctness and performance baseline for the distributed version.

**Acceptance Criteria:**

**Given** the loaded Reddit dataset (or sampled subset) and configurable hyperparameters (L, d, eta, epochs)
**When** training is run
**Then** the GCN implements the exact assignment algorithm (forward pass with neighbor aggregation, loss computation, backpropagation, weight update), stores adjacency as scipy.sparse, and logs per-epoch training time, loss, training accuracy, and validation F1-score.

### Story 2.2: Parameter Server Simulation

As a developer,
I want a parameter server module using Spark's driver as the central coordinator,
so that global weights are broadcast and gradients are synchronously aggregated each epoch.

**Acceptance Criteria:**

**Given** the Spark driver holds global weight matrices
**When** an epoch begins
**Then** weights are broadcast to all workers via `SparkContext.broadcast()`, after local computation each partition's gradients are collected and aggregated on the driver, and all workers use the same weight version per epoch (BSP model).

### Story 2.3: Spark-Based Distributed GCN (MapReduce)

As a developer,
I want a distributed GCN training implementation using PySpark MapReduce,
so that the GCN training scales across multiple partitions/workers.

**Acceptance Criteria:**

**Given** a partitioned graph loaded as Spark RDDs, broadcast weights, and configurable parallelism (local[N], N in {1,2,4,8,16})
**When** the distributed training loop runs
**Then** the Map phase emits (node_v, neighbor_feature) pairs independently per node, the Reduce phase aggregates via SUM and applies weight multiplication + activation, weights are broadcast via `SparkContext.broadcast()` each epoch, gradients are aggregated via `rdd.treeReduce()` or `rdd.reduce()`, the driver updates global weights, and the full epoch loop runs with identical hyperparameters to the sequential version.

---

## Epic 3: Instrumentation, Testing & Benchmarking

**Goal:** Instrument both implementations with comprehensive metrics, verify correctness, run systematic benchmarks, generate visualizations, and provide a single demonstration script.

### Story 3.1: Performance Measurement Infrastructure

As a developer,
I want comprehensive timing and metric collection instrumented in both implementations,
so that I can accurately compare sequential vs. distributed performance.

**Acceptance Criteria:**

**Given** both sequential and Spark GCN implementations
**When** a training run executes
**Then** high-resolution wall-clock timing (time.perf_counter) captures: total training time, per-epoch time, forward pass time, backward pass time, communication time (Spark only: broadcast + gradient aggregation), weight update time. Accuracy metrics include per-epoch loss, training accuracy, validation F1-score (macro), and test F1-score. System metrics include peak memory usage. All metrics are saved to CSV files.

### Story 3.2: Correctness Testing

As a developer,
I want unit and integration tests verifying both GCN implementations produce correct results,
so that I can confirm accuracy before benchmarking.

**Acceptance Criteria:**

**Given** both sequential and Spark GCN implementations
**When** the test suite runs
**Then** unit tests verify: GCN layer forward pass against hand-computed values, backward pass via numerical gradient checking (finite differences), neighbor message aggregation correctness, graph partitioning (all nodes assigned, no duplicates, correct edge mapping). Integration tests verify: sequential GCN converges on a small known graph (e.g., Karate Club, 34 nodes), Spark GCN produces equivalent results on the same small graph (within floating-point tolerance). Equivalence test: with same hyperparameters and random seed, sequential and Spark GCN with 1 partition produce identical outputs (within 1e-5 tolerance).

### Story 3.3: Performance Benchmarking

As a developer,
I want systematic benchmark runs across all configurations,
so that I can quantify speedup, communication overhead, and accuracy trade-offs.

**Acceptance Criteria:**

**Given** the experiment matrix: Sequential baseline (no Spark), Spark local[1], local[2], local[4], local[8], local[16] on the Reddit dataset
**When** benchmarks are run (at least 3 repeated runs per configuration)
**Then** each run records: total training time (N epochs), average per-epoch time, computation vs. communication time breakdown (Spark only), final test F1-score, peak memory usage. Results report mean and standard deviation across runs.

### Story 3.4: Performance Comparison & Visualization

As a developer,
I want comparative analysis charts and summary tables,
so that the results clearly demonstrate speedup and trade-offs.

**Acceptance Criteria:**

**Given** benchmark CSV results from all configurations
**When** the visualization script runs
**Then** it generates: Speedup chart (line plot, Speedup vs. Workers, with ideal linear reference), Epoch time chart (bar chart comparing per-epoch wall-clock time), Communication overhead chart (stacked bar: computation vs. communication time), Accuracy vs. Speedup trade-off (scatter/table: F1-score vs. speedup), Scaling efficiency chart (Speedup/P vs. P), and a consolidated summary table with all metrics.

### Story 3.5: Demonstration Script

As a developer,
I want a single master script (`run_all.sh`) that runs the complete demonstration end-to-end,
so that the entire experiment is reproducible in one command.

**Acceptance Criteria:**

**Given** a configured environment
**When** `run_all.sh` is executed
**Then** it: checks environment prerequisites (Java, Python, Spark), downloads/prepares the Reddit dataset if not present, runs sequential GCN baseline, runs Spark GCN for each partition count (1, 2, 4, 8, 16), generates comparison results (CSV + charts), prints summary to console. The script is idempotent (safe to re-run). Estimated runtime: ~30-60 minutes.
