# Story 3.5: Demonstration Script

Status: ready-for-dev

## Story

As a developer,
I want a single master script (`run_all.sh`) that runs the complete demonstration end-to-end,
so that the entire experiment is reproducible in one command.

## Acceptance Criteria

1. Checks environment prerequisites: Java 11+, Python 3.10-3.12, PySpark installed, required libraries present.
2. Downloads/prepares the Reddit dataset if not already present in `data/`.
3. Runs the sequential GCN baseline and records metrics.
4. Runs Spark GCN for each partition count: 1, 2, 4, 8, 16 — recording metrics for each.
5. Generates comparison results: aggregated CSV and all visualization charts.
6. Prints a summary table to console at the end.
7. The script is idempotent (safe to re-run without side effects; skips completed steps or overwrites cleanly).
8. Estimated total runtime: ~30-60 minutes on the target machine (16 cores, 64GB RAM).

## Tasks / Subtasks

- [ ] Task 1: Implement environment check (AC: #1)
  - [ ] Check `java -version` returns 11+
  - [ ] Check `python --version` returns 3.10-3.12
  - [ ] Check `python -c "import pyspark"` succeeds
  - [ ] Check all required libraries importable: numpy, scipy, sklearn, networkx, matplotlib, pandas
  - [ ] Print clear error messages and exit on failure
- [ ] Task 2: Implement dataset preparation step (AC: #2)
  - [ ] Check if `data/reddit_data.npz` (or equivalent) exists
  - [ ] If not, invoke the dataset loader to download and extract
- [ ] Task 3: Run sequential baseline (AC: #3)
  - [ ] Call `python benchmarks/run_sequential.py`
  - [ ] Verify output CSV created in `results/`
- [ ] Task 4: Run Spark GCN for all partition counts (AC: #4)
  - [ ] Loop over N in {1, 2, 4, 8, 16}
  - [ ] Call `python benchmarks/run_spark.py --workers N` for each
  - [ ] Verify output CSVs created
- [ ] Task 5: Generate comparison results (AC: #5, #6)
  - [ ] Call `python benchmarks/compare_results.py`
  - [ ] Call visualization script to generate all charts
  - [ ] Print summary table to console
- [ ] Task 6: Idempotency and error handling (AC: #7)
  - [ ] Use set -e for fail-fast
  - [ ] Create `results/` directory if not present
  - [ ] Clean previous results or append timestamps

## Dev Notes

- **Target file:** `run_all.sh` (project root)
- **Shell:** Bash. Use `#!/usr/bin/env bash` for portability.
- **JAVA_HOME:** Export `JAVA_HOME=/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home` at the top.
- **PYSPARK_PYTHON:** Export `PYSPARK_PYTHON=$(which python3)` to ensure Spark uses the correct Python.
- **Timing:** Print start/end timestamps and total elapsed time for the full run.
- **Logging:** Redirect stdout/stderr to a log file (`results/run_all.log`) while also printing to console (use `tee`).

### Project Structure Notes

- File location: `run_all.sh` (project root)
- Depends on: All benchmark scripts (Story 3.3), visualization (Story 3.4), data loader (Story 1.1)

### References

- [Source: _bmad-output/planning-artifacts/requirements.md#REQ-P3-004]
- [Source: _bmad-output/planning-artifacts/requirements.md#Environment Setup]
