"""
Story 2.3: Spark-Based Distributed GCN (MapReduce).

Distributed GCN training using PySpark. The graph is partitioned and
each partition processes its local sub-graph independently. Weights are
synchronized via broadcast (parameter server pattern).

Distributed Algorithm:
    Phase 1: Partition graph, distribute to workers
    Phase 2: For each epoch:
        2.1 Broadcast weights from driver to all workers
        2.2 Map: Each partition does local forward pass (message agg + transform)
        2.3 Reduce: Each partition computes local gradients
        2.4 Aggregate gradients on driver, update weights
"""
import time
import os
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

from src.model.layers import (
    gcn_forward, gcn_backward, softmax, cross_entropy_loss
)
from src.model.parameter_server import ParameterServer
from src.utils.timer import Timer
from src.utils.metrics import compute_f1, compute_accuracy
from src.data.partitioner import partition_graph


def _partition_forward_backward(partition_data, broadcast_weights, num_layers, num_classes):
    """
    Worker function: Run forward and backward pass on a single partition.
    This is the Map + local Reduce operation.

    Args:
        partition_data: Dict with local graph data
        broadcast_weights: Broadcast variable containing weights list
        num_layers: Number of GCN layers
        num_classes: Number of output classes

    Returns:
        Dict with gradients, loss, predictions, labels, masks, and local node count
    """
    weights = broadcast_weights.value

    # Reconstruct sparse adjacency from COO components (efficient deserialization)
    adj = sp.csr_matrix(
        (partition_data['adj_data'],
         (partition_data['adj_row'], partition_data['adj_col'])),
        shape=partition_data['adj_shape']
    )
    features = partition_data['features']
    labels = partition_data['labels']
    train_mask = partition_data['train_mask']

    # --- FORWARD PASS (Map + Reduce per layer) ---
    H = features.astype(np.float32)
    caches = []

    for l in range(num_layers):
        apply_relu = (l < num_layers - 1)
        # Map: For each node, aggregate neighbor features via sparse matmul
        # Reduce: Sum aggregated messages, apply weight transform + activation
        cache = gcn_forward(adj, H, weights[l], apply_relu=apply_relu)
        caches.append(cache)
        H = cache['output']

    logits = H

    # --- BACKWARD PASS ---
    loss = cross_entropy_loss(logits, labels, train_mask)

    # Compute softmax gradient
    probs = softmax(logits)
    n_train = train_mask.sum()

    d_output = np.zeros_like(logits, dtype=np.float32)
    if n_train > 0:
        one_hot = np.zeros_like(logits, dtype=np.float32)
        one_hot[train_mask, labels[train_mask]] = 1.0
        d_output[train_mask] = (probs[train_mask] - one_hot[train_mask]) / max(n_train, 1)

    # Backpropagate through layers
    gradients = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        result = gcn_backward(adj, caches[l], d_output)
        gradients[l] = result['dW']
        d_output = result['d_input']

    # Predictions
    preds = np.argmax(logits, axis=1)

    return {
        'gradients': gradients,
        'loss': loss,
        'n_train': int(n_train),
        'preds': preds,
        'labels': labels,
        'train_mask': train_mask,
        'val_mask': partition_data['val_mask'],
        'test_mask': partition_data['test_mask'],
    }


class SparkGCN:
    """
    Spark-based distributed Graph Convolutional Network.
    Uses MapReduce paradigm with parameter server pattern.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        learning_rate: float = 0.01,
        seed: int = 42,
        num_workers: int = 4,
        driver_memory: str = "8g",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.lr = learning_rate
        self.seed = seed
        self.num_workers = num_workers
        self.driver_memory = driver_memory

        self.sc = None
        self.spark = None
        self.param_server = None

        # Initialize weights (same as sequential for fair comparison)
        self._init_weights()

    def _init_weights(self):
        """Initialize weight matrices with Xavier init (same seed as sequential)."""
        rng = np.random.RandomState(self.seed)
        self.weights = []

        dims = [self.input_dim] + \
               [self.hidden_dim] * (self.num_layers - 1) + \
               [self.output_dim]

        for l in range(self.num_layers):
            fan_in = dims[l]
            fan_out = dims[l + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            W = rng.randn(fan_in, fan_out).astype(np.float32) * std
            self.weights.append(W)

    def _init_spark(self):
        """Initialize Spark session."""
        os.environ.setdefault(
            'JAVA_HOME',
            '/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home'
        )

        conf = SparkConf() \
            .setAppName("GCN-Spark") \
            .setMaster(f"local[{self.num_workers}]") \
            .set("spark.driver.memory", "16g") \
            .set("spark.executor.memory", "16g") \
            .set("spark.driver.maxResultSize", "8g") \
            .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .set("spark.kryoserializer.buffer.max", "1g") \
            .set("spark.ui.showConsoleProgress", "false") \
            .set("spark.log.level", "ERROR") \
            .set("spark.python.worker.memory", "4g")

        self.spark = SparkSession.builder.config(conf=conf).getOrCreate()
        self.sc = self.spark.sparkContext
        self.sc.setLogLevel("ERROR")

    def _stop_spark(self):
        """Stop Spark session."""
        if self.spark is not None:
            try:
                self.spark.stop()
            except Exception:
                pass
            self.spark = None
            self.sc = None

    def train(
        self,
        adj: sp.csr_matrix,
        features: np.ndarray,
        labels: np.ndarray,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray,
        num_epochs: int = 5,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Full distributed training loop.

        Returns list of per-epoch metrics.
        """
        # Initialize Spark
        self._init_spark()

        try:
            return self._train_loop(
                adj, features, labels,
                train_mask, val_mask, test_mask,
                num_epochs, verbose
            )
        finally:
            if self.param_server:
                self.param_server.cleanup()
            self._stop_spark()

    def _train_loop(
        self,
        adj, features, labels,
        train_mask, val_mask, test_mask,
        num_epochs, verbose
    ):
        """Internal training loop with Spark context active."""
        P = self.num_workers
        num_classes = self.output_dim

        # Phase 1: Partition the graph
        if verbose:
            print(f"[SparkGCN] Partitioning graph into {P} partitions...")
        partitions, quality = partition_graph(
            adj, features, labels,
            train_mask, val_mask, test_mask,
            num_partitions=P
        )

        # Create RDD from partitions
        partition_rdd = self.sc.parallelize(
            [(p['partition_id'], p) for p in partitions],
            numSlices=P
        ).cache()

        # Initialize parameter server with global weights
        self.param_server = ParameterServer(self.weights)

        epoch_metrics = []

        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            timer = Timer()

            # Step 2.1: Broadcast weights
            timer.start('broadcast')
            broadcast_weights, bcast_time = self.param_server.broadcast_weights(self.sc)
            timer.stop('broadcast')

            # Step 2.2 + 2.3: Map (forward) + local Reduce (backward) on each partition
            timer.start('forward')
            num_layers = self.num_layers
            results_rdd = partition_rdd.map(
                lambda x: (x[0], _partition_forward_backward(
                    x[1], broadcast_weights, num_layers, num_classes
                ))
            )
            # Force computation by collecting results
            results = results_rdd.collect()
            timer.stop('forward')

            # Step 2.4: Aggregate gradients
            timer.start('gradient_agg')
            # Extract gradients and aggregate
            all_gradients = [r[1]['gradients'] for r in results]
            aggregated = [
                sum(g[l] for g in all_gradients) / P
                for l in range(self.num_layers)
            ]
            timer.stop('gradient_agg')

            # Update weights on driver
            timer.start('update')
            self.param_server.update_weights(aggregated, self.lr)
            self.weights = self.param_server.get_weights()
            timer.stop('update')

            # Collect metrics from all partitions
            total_loss = 0.0
            total_train = 0
            all_preds = []
            all_labels = []
            all_train_masks = []
            all_val_masks = []
            all_test_masks = []

            for _, result in results:
                total_loss += result['loss'] * result['n_train']
                total_train += result['n_train']
                all_preds.append(result['preds'])
                all_labels.append(result['labels'])
                all_train_masks.append(result['train_mask'])
                all_val_masks.append(result['val_mask'])
                all_test_masks.append(result['test_mask'])

            avg_loss = total_loss / max(total_train, 1)
            preds = np.concatenate(all_preds)
            lbls = np.concatenate(all_labels)
            t_mask = np.concatenate(all_train_masks)
            v_mask = np.concatenate(all_val_masks)
            te_mask = np.concatenate(all_test_masks)

            train_acc = compute_accuracy(preds, lbls, t_mask)
            val_f1 = compute_f1(preds, lbls, v_mask)
            test_f1 = compute_f1(preds, lbls, te_mask)

            epoch_time = time.perf_counter() - epoch_start
            comm_time = timer.get('broadcast') + timer.get('gradient_agg')

            metrics = {
                'epoch': epoch + 1,
                'epoch_time': epoch_time,
                'loss': avg_loss,
                'train_acc': train_acc,
                'val_f1': val_f1,
                'test_f1': test_f1,
                'forward_time': timer.get('forward'),
                'backward_time': 0.0,  # Included in forward (map+reduce together)
                'comm_time': comm_time,
                'update_time': timer.get('update'),
                'broadcast_time': timer.get('broadcast'),
                'gradient_agg_time': timer.get('gradient_agg'),
            }
            epoch_metrics.append(metrics)

            if verbose:
                print(f"  Epoch {epoch+1}/{num_epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val F1: {val_f1:.4f} | "
                      f"Time: {epoch_time:.2f}s "
                      f"(comp: {timer.get('forward'):.2f}s, "
                      f"comm: {comm_time:.2f}s)")

        return epoch_metrics
