"""
Configuration for GCN training experiments.
Hyperparameters, Spark config, and paths.
"""
import os

# --- Project Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --- Dataset ---
DATASET_NAME = "reddit"
REDDIT_URL = "https://data.dgl.ai/dataset/reddit.zip"

# --- GCN Hyperparameters ---
NUM_LAYERS = 2          # L = 2 layers (per assignment)
HIDDEN_DIM = 128        # d = 128 hidden dimension
LEARNING_RATE = 0.01    # eta
NUM_EPOCHS = 3          # epochs for benchmarking (Reddit is large)
WEIGHT_DECAY = 0.0      # no regularization for simplicity
DROPOUT = 0.0           # no dropout for fair comparison

# --- Benchmark ---
# Skip P=1,2 for Spark (too large per partition for serialization on Reddit)
# Sequential baseline covers the single-worker case
PARTITION_COUNTS = [4, 8, 16]
NUM_REPEATS = 3         # repeated runs per configuration
SEEDS = [42, 123, 456]  # random seeds for repeated runs

# --- Spark ---
SPARK_DRIVER_MEMORY = "16g"
SPARK_EXECUTOR_MEMORY = "16g"

# --- Environment ---
JAVA_HOME = "/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home"
