"""
Story 2.2: Parameter Server Simulation.

Uses Spark's driver as the central coordinator for weight broadcast
and gradient aggregation (Bulk Synchronous Parallel model).
"""
import time
import numpy as np
from typing import Dict, List, Optional


class ParameterServer:
    """
    Parameter server that manages global weights.
    Uses Spark broadcast for weight distribution and
    treeReduce for gradient aggregation.
    """

    def __init__(self, weights: List[np.ndarray]):
        """
        Initialize with global weight matrices.

        Args:
            weights: List of weight matrices W(1)...W(L)
        """
        self.weights = [w.copy() for w in weights]
        self._broadcast_var = None

    def broadcast_weights(self, sc) -> tuple:
        """
        Broadcast current weights to all workers via SparkContext.broadcast().

        Args:
            sc: SparkContext

        Returns:
            broadcast_variable, broadcast_time (seconds)
        """
        # Unpersist previous broadcast to prevent memory leaks
        if self._broadcast_var is not None:
            try:
                self._broadcast_var.unpersist(blocking=False)
            except Exception:
                pass

        start = time.perf_counter()
        # Broadcast weights as a list of numpy arrays
        self._broadcast_var = sc.broadcast(self.weights)
        broadcast_time = time.perf_counter() - start

        return self._broadcast_var, broadcast_time

    @staticmethod
    def aggregate_gradients(gradient_rdd, num_partitions: int):
        """
        Aggregate gradients from all partitions via treeReduce.

        Args:
            gradient_rdd: RDD of (partition_id, gradient_list) where
                          gradient_list is a List[np.ndarray]
            num_partitions: Number of partitions for averaging

        Returns:
            aggregated_gradients: List[np.ndarray], aggregation_time: float
        """
        start = time.perf_counter()

        def add_gradients(grads_a, grads_b):
            """Element-wise sum of two gradient lists."""
            return [a + b for a, b in zip(grads_a, grads_b)]

        # Use treeReduce for efficient aggregation (O(log P) rounds)
        total_gradients = gradient_rdd.map(lambda x: x[1]).treeReduce(
            add_gradients, depth=3
        )

        # Average by number of partitions
        avg_gradients = [g / num_partitions for g in total_gradients]

        agg_time = time.perf_counter() - start
        return avg_gradients, agg_time

    def update_weights(self, gradients: List[np.ndarray], lr: float):
        """
        SGD weight update on the driver.

        Args:
            gradients: Aggregated global gradients
            lr: Learning rate
        """
        for l in range(len(self.weights)):
            self.weights[l] -= lr * gradients[l]

    def get_weights(self) -> List[np.ndarray]:
        """Return current global weights."""
        return [w.copy() for w in self.weights]

    def cleanup(self):
        """Clean up broadcast variables."""
        if self._broadcast_var is not None:
            try:
                self._broadcast_var.unpersist(blocking=True)
            except Exception:
                pass
            self._broadcast_var = None
