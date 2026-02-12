#!/usr/bin/env bash
#
# Story 3.5: Master demonstration script.
# Runs the complete GCN benchmark end-to-end on Reddit dataset.
#
# Usage: bash run_all.sh
#
set -e

echo "============================================================"
echo "  ML SysOps Assignment 1 - Distributed GCN Benchmarks"
echo "  $(date)"
echo "============================================================"

# --- Environment Setup ---
export JAVA_HOME="${JAVA_HOME:-/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home}"
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-$(which python3)}"
export OMP_NUM_THREADS=1  # Single-threaded NumPy for fair benchmarks
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[Setup] Activated virtual environment"
fi

# --- Step 1: Check Prerequisites ---
echo ""
echo "[Step 1] Checking prerequisites..."

# Java
if ! java -version 2>&1 | grep -q "11\|17\|21"; then
    echo "ERROR: Java 11+ required."
    java -version 2>&1
    exit 1
fi
echo "  Java: OK"

# Python
PYTHON_VERSION=$(python3 --version 2>&1)
echo "  Python: $PYTHON_VERSION"

# PySpark
if ! python3 -c "import pyspark" 2>/dev/null; then
    echo "ERROR: PySpark not installed. Run: pip install -r requirements.txt"
    exit 1
fi
echo "  PySpark: OK"

# Other libraries
for lib in numpy scipy sklearn networkx matplotlib pandas; do
    if ! python3 -c "import $lib" 2>/dev/null; then
        echo "ERROR: $lib not installed. Run: pip install -r requirements.txt"
        exit 1
    fi
done
echo "  All Python libraries: OK"

# --- Step 2: Run full benchmark suite ---
echo ""
echo "[Step 2] Running full benchmark suite..."
python3 benchmarks/run_all_benchmarks.py 2>results/benchmark_stderr.log

# --- Done ---
echo ""
echo "============================================================"
echo "  BENCHMARK COMPLETE"
echo "  $(date)"
echo "============================================================"
echo ""
echo "Results saved in: results/"
echo "  - Performance report: results/performance_report.txt"
echo "  - Summary CSV:        results/summary_table.csv"
echo "  - Charts:             results/*.png"
echo ""
