"""
Story 2.1: GCN Layer Logic.

Implements the GCN layer forward and backward passes.
Forward:  H(l) = sigma(A_hat @ H(l-1) @ W(l))

Optimization: Since (A @ H) @ W = A @ (H @ W) by matrix associativity,
we compute H @ W first (reducing dimension from F_in to F_out), then
the sparse matmul A @ (H@W) is much cheaper when F_out < F_in.

Shared by both sequential and Spark implementations.
"""
import numpy as np
import scipy.sparse as sp


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along axis=1."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 where x > 0, else 0."""
    return (x > 0).astype(np.float32)


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray,
                       mask: np.ndarray) -> float:
    """
    Compute cross-entropy loss on masked nodes.

    Args:
        logits: (N, C) raw scores (pre-softmax)
        labels: (N,) integer class labels
        mask: (N,) boolean mask for training nodes

    Returns:
        Scalar loss value
    """
    probs = softmax(logits)
    n_masked = mask.sum()
    if n_masked == 0:
        return 0.0
    probs_masked = probs[mask]
    labels_masked = labels[mask]
    log_probs = -np.log(np.clip(probs_masked[np.arange(n_masked), labels_masked], 1e-15, 1.0))
    return float(log_probs.mean())


def gcn_forward(adj: sp.csr_matrix, H: np.ndarray, W: np.ndarray,
                apply_relu: bool = True) -> dict:
    """
    Single GCN layer forward pass (optimized order).

    Computes: H_out = sigma(A_hat @ (H @ W))
    Mathematically equivalent to: sigma((A_hat @ H) @ W)
    But cheaper when F_out < F_in since sparse matmul on smaller matrix.

    The algorithm implements the assignment pseudocode:
        msg_agg = sum(H_u for u in N(v))  ->  A_hat @ (H @ W) combines
        H_v = sigma(msg_agg . W)               aggregation + transform

    Args:
        adj: Normalized sparse adjacency (N, N)
        H: Input features (N, F_in)
        W: Weight matrix (F_in, F_out)
        apply_relu: Whether to apply ReLU (False for last layer)

    Returns:
        Dict with 'output', 'pre_activation' and cache for backprop
    """
    # Step 1: Dense transform (project to lower dimension first)
    HW = H @ W  # (N, F_out) — dense matmul, reduces dimension

    # Step 2: Message passing on the lower-dimensional features
    # This is the neighbor aggregation: for each node, sum neighbors' projected features
    pre_activation = adj @ HW  # (N, F_out) — sparse matmul, much cheaper now

    # Step 3: Activation
    if apply_relu:
        output = relu(pre_activation)
    else:
        output = pre_activation

    return {
        'output': output,
        'pre_activation': pre_activation,
        'HW': HW,
        'input': H,
        'weights': W,
        'apply_relu': apply_relu,
    }


def gcn_backward(adj: sp.csr_matrix, cache: dict, d_output: np.ndarray) -> dict:
    """
    Single GCN layer backward pass (optimized order).

    For forward: Z = adj @ (H @ W), output = relu(Z)
    Backward:
        dZ = d_output * relu'(Z)
        dHW = adj^T @ dZ = adj @ dZ  (adj is symmetric)
        dW = H^T @ dHW
        dH = dHW @ W^T

    But dH needs to propagate through adj too:
    Actually: Z = adj @ M where M = H @ W
    dM = adj^T @ dZ
    dW = H^T @ dM
    dH = dM @ W^T

    Args:
        adj: Normalized sparse adjacency (symmetric, so A^T = A)
        cache: Dict from forward pass
        d_output: Gradient of loss w.r.t. layer output (N, F_out)

    Returns:
        Dict with 'dW' (gradient w.r.t. weights) and 'd_input' (gradient w.r.t. input)
    """
    H = cache['input']  # (N, F_in)
    W = cache['weights']  # (F_in, F_out)
    pre_activation = cache['pre_activation']  # (N, F_out)
    apply_relu = cache['apply_relu']

    # Gradient through activation
    if apply_relu:
        dZ = d_output * relu_derivative(pre_activation)  # (N, F_out)
    else:
        dZ = d_output  # (N, F_out)

    # Gradient through sparse matmul: Z = adj @ M, so dM = adj^T @ dZ
    # adj is symmetric -> adj^T = adj
    dM = adj @ dZ  # (N, F_out) — sparse matmul

    # Gradient w.r.t. weights: M = H @ W, so dW = H^T @ dM
    dW = H.T @ dM  # (F_in, F_out) — dense matmul

    # Gradient w.r.t. input: M = H @ W, so dH = dM @ W^T
    d_input = dM @ W.T  # (N, F_in) — dense matmul

    return {
        'dW': dW,
        'd_input': d_input,
    }
