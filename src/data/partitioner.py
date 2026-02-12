"""
Story 1.2: Graph Partitioning.

Splits a full graph into P balanced sub-graphs using hash-based partitioning.
Output is compatible with Spark RDD creation.
"""
import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Any, Tuple

from src.data.loader import _normalize_adjacency


def partition_graph(
    adj: sp.csr_matrix,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    num_partitions: int
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Hash-based graph partitioning: partition_id = node_id % P.

    Args:
        adj: Sparse adjacency matrix (normalized or raw)
        features: Node feature matrix (N, F)
        labels: Node labels (N,)
        train_mask, val_mask, test_mask: Boolean masks (N,)
        num_partitions: P - number of partitions

    Returns:
        partitions: List of dicts, one per partition:
            {
                'partition_id': int,
                'node_ids': np.ndarray,
                'features': np.ndarray,
                'labels': np.ndarray,
                'adj': sp.csr_matrix (local, normalized),
                'train_mask': np.ndarray,
                'val_mask': np.ndarray,
                'test_mask': np.ndarray,
            }
        quality_metrics: Dict with 'balance_ratio' and 'edge_cut_percentage'
    """
    num_nodes = features.shape[0]
    P = num_partitions

    # Assign each node to a partition via hash
    partition_ids = np.arange(num_nodes) % P

    # Build per-partition data
    partitions = []
    partition_sizes = []

    # Get the raw adjacency (without normalization) for edge counting
    # We need to build local adjacency and re-normalize per partition
    # First, extract edges from the adjacency
    adj_coo = adj.tocoo()
    rows, cols = adj_coo.row, adj_coo.col

    total_edges = len(rows)
    cross_partition_edges = 0

    for pid in range(P):
        # Nodes in this partition
        node_mask = (partition_ids == pid)
        local_node_ids = np.where(node_mask)[0]
        n_local = len(local_node_ids)
        partition_sizes.append(n_local)

        # Create node ID mapping: global -> local
        global_to_local = np.full(num_nodes, -1, dtype=np.int64)
        global_to_local[local_node_ids] = np.arange(n_local)

        # Find intra-partition edges
        src_in_partition = node_mask[rows]
        dst_in_partition = node_mask[cols]
        intra_mask = src_in_partition & dst_in_partition

        local_rows = global_to_local[rows[intra_mask]]
        local_cols = global_to_local[cols[intra_mask]]

        # Build local adjacency (already inherits normalization values from parent)
        if len(local_rows) > 0:
            local_adj = sp.csr_matrix(
                (adj_coo.data[intra_mask], (local_rows, local_cols)),
                shape=(n_local, n_local)
            )
        else:
            local_adj = sp.csr_matrix((n_local, n_local), dtype=np.float32)

        # Re-normalize local adjacency (add self-loops + symmetric norm)
        local_raw_adj = sp.csr_matrix(
            (np.ones(len(local_rows), dtype=np.float32), (local_rows, local_cols)),
            shape=(n_local, n_local)
        )
        # Make symmetric
        local_raw_adj = local_raw_adj + local_raw_adj.T
        local_raw_adj[local_raw_adj > 1] = 1
        local_adj_normalized = _normalize_adjacency(local_raw_adj)

        # Store adjacency as COO components for efficient serialization
        # (scipy CSR pickle can be slow for very large matrices)
        local_coo = local_adj_normalized.tocoo()

        partitions.append({
            'partition_id': pid,
            'node_ids': local_node_ids,
            'features': features[local_node_ids].copy(),
            'labels': labels[local_node_ids].copy(),
            'adj_data': local_coo.data.astype(np.float32),
            'adj_row': local_coo.row.astype(np.int32),
            'adj_col': local_coo.col.astype(np.int32),
            'adj_shape': local_coo.shape,
            'train_mask': train_mask[local_node_ids].copy(),
            'val_mask': val_mask[local_node_ids].copy(),
            'test_mask': test_mask[local_node_ids].copy(),
        })

    # Count cross-partition edges (edges where src and dst in different partitions)
    src_partitions = partition_ids[rows]
    dst_partitions = partition_ids[cols]
    cross_partition_edges = int(np.sum(src_partitions != dst_partitions))

    # Quality metrics
    avg_size = np.mean(partition_sizes)
    max_size = np.max(partition_sizes)
    balance_ratio = max_size / avg_size if avg_size > 0 else float('inf')
    edge_cut_pct = (cross_partition_edges / total_edges * 100) if total_edges > 0 else 0.0

    quality_metrics = {
        'balance_ratio': balance_ratio,
        'edge_cut_percentage': edge_cut_pct,
        'partition_sizes': partition_sizes,
        'total_edges': total_edges,
        'cross_partition_edges': cross_partition_edges,
        'intra_partition_edges': total_edges - cross_partition_edges,
    }

    print(f"[Partitioner] {P} partitions created.")
    print(f"[Partitioner] Sizes: {partition_sizes}")
    print(f"[Partitioner] Balance ratio: {balance_ratio:.4f}")
    print(f"[Partitioner] Edge cut: {edge_cut_pct:.2f}% "
          f"({cross_partition_edges}/{total_edges})")

    return partitions, quality_metrics
