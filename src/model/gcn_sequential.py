"""
Story 2.1: Sequential GCN Baseline.

Full GCN implementation in pure Python/NumPy following the assignment pseudocode.
No Spark, no PyTorch — pure NumPy/SciPy only.

Algorithm:
    for epoch:
        H(0) = X
        for layer l = 1 to L:
            for each node v:
                msg_agg = sum(H_u(l-1) for u in N(v))
                H_v(l) = sigma(msg_agg . W(l))
        Loss = CrossEntropy(H(L), Y)
        Compute gradients via backpropagation
        Update weights: W(l) -= lr * dW(l)
"""
import time
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple

from src.model.layers import (
    gcn_forward, gcn_backward, softmax, cross_entropy_loss
)
from src.utils.timer import Timer
from src.utils.metrics import compute_f1, compute_accuracy


class SequentialGCN:
    """
    Sequential (non-parallel) Graph Convolutional Network.

    Architecture: L-layer GCN with ReLU activations and softmax output.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        learning_rate: float = 0.01,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.lr = learning_rate
        self.seed = seed

        # Initialize weights with Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weight matrices W(1)...W(L) with Xavier init."""
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

    def forward(self, adj: sp.csr_matrix, features: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Forward pass through all GCN layers.

        Args:
            adj: Normalized adjacency matrix (N, N)
            features: Input features (N, F)

        Returns:
            logits: Output logits (N, C)
            caches: List of per-layer caches for backprop
        """
        H = features
        caches = []

        for l in range(self.num_layers):
            apply_relu = (l < self.num_layers - 1)  # No ReLU on last layer
            cache = gcn_forward(adj, H, self.weights[l], apply_relu=apply_relu)
            caches.append(cache)
            H = cache['output']

        return H, caches

    def backward(
        self,
        adj: sp.csr_matrix,
        logits: np.ndarray,
        labels: np.ndarray,
        train_mask: np.ndarray,
        caches: List[dict],
    ) -> Tuple[List[np.ndarray], float]:
        """
        Backward pass: compute gradients for all layers.

        Args:
            adj: Normalized adjacency matrix
            logits: Output from forward pass (N, C)
            labels: Ground truth labels (N,)
            train_mask: Boolean mask for training nodes
            caches: Per-layer caches from forward pass

        Returns:
            gradients: List of dW for each layer
            loss: Scalar loss value
        """
        # Compute loss
        loss = cross_entropy_loss(logits, labels, train_mask)

        # Gradient of softmax + cross-entropy: d_logits = (P - Y_onehot) / N_train
        probs = softmax(logits)
        n_train = train_mask.sum()

        d_output = np.zeros_like(logits)
        if n_train > 0:
            one_hot = np.zeros_like(logits)
            one_hot[train_mask, labels[train_mask]] = 1.0
            d_output[train_mask] = (probs[train_mask] - one_hot[train_mask]) / n_train

        # Backpropagate through layers (reverse order)
        gradients = [None] * self.num_layers
        for l in range(self.num_layers - 1, -1, -1):
            result = gcn_backward(adj, caches[l], d_output)
            gradients[l] = result['dW']
            d_output = result['d_input']

        return gradients, loss

    def update_weights(self, gradients: List[np.ndarray]):
        """SGD weight update: W(l) -= lr * dW(l)."""
        for l in range(self.num_layers):
            self.weights[l] -= self.lr * gradients[l]

    def predict(self, adj: sp.csr_matrix, features: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        logits, _ = self.forward(adj, features)
        return np.argmax(logits, axis=1)

    def train_epoch(
        self,
        adj: sp.csr_matrix,
        features: np.ndarray,
        labels: np.ndarray,
        train_mask: np.ndarray,
    ) -> dict:
        """
        Run one training epoch with timing instrumentation.

        Returns dict with timing, metric information, and logits for eval.
        """
        timer = Timer()

        # Forward pass
        timer.start('forward')
        logits, caches = self.forward(adj, features)
        timer.stop('forward')

        # Backward pass
        timer.start('backward')
        gradients, loss = self.backward(adj, logits, labels, train_mask, caches)
        timer.stop('backward')

        # Weight update
        timer.start('update')
        self.update_weights(gradients)
        timer.stop('update')

        # Compute training accuracy using training forward pass logits
        preds = np.argmax(logits, axis=1)
        train_acc = compute_accuracy(preds, labels, train_mask)

        return {
            'loss': loss,
            'train_acc': train_acc,
            'preds': preds,
            'forward_time': timer.get('forward'),
            'backward_time': timer.get('backward'),
            'update_time': timer.get('update'),
            'comm_time': 0.0,  # No communication in sequential version
        }

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
        Full training loop.

        Returns list of per-epoch metrics.
        """
        epoch_metrics = []

        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()

            # Train one epoch
            metrics = self.train_epoch(adj, features, labels, train_mask)

            # Use training-time predictions for val/test metrics
            # (avoids extra forward pass; uses pre-update logits which is standard practice)
            preds = metrics.pop('preds')
            val_f1 = compute_f1(preds, labels, val_mask)
            test_f1 = compute_f1(preds, labels, test_mask)

            epoch_time = time.perf_counter() - epoch_start

            metrics.update({
                'epoch': epoch + 1,
                'epoch_time': epoch_time,
                'val_f1': val_f1,
                'test_f1': test_f1,
            })
            epoch_metrics.append(metrics)

            if verbose:
                print(f"  Epoch {epoch+1}/{num_epochs} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Train Acc: {metrics['train_acc']:.4f} | "
                      f"Val F1: {val_f1:.4f} | "
                      f"Time: {epoch_time:.2f}s "
                      f"(fwd: {metrics['forward_time']:.2f}s, "
                      f"bwd: {metrics['backward_time']:.2f}s)")

        return epoch_metrics
