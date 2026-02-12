"""
Story 1.1: Dataset Loading & Preprocessing.

Loads and preprocesses graph datasets (Reddit, PubMed, Cora, Citeseer, Karate Club).
Returns NumPy/SciPy objects consumable by both sequential and Spark GCN.
"""
import os
import pickle
import zipfile
import urllib.request
import numpy as np
import scipy.sparse as sp
import networkx as nx

from src.config import DATA_DIR, REDDIT_URL


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to zero-mean, unit-variance per feature column."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    return (features - mean) / std


def _normalize_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    """
    Symmetric normalization: D^{-1/2} (A + I) D^{-1/2}.
    Add self-loops, compute degree, normalize.
    """
    adj_with_self = adj + sp.eye(adj.shape[0], format='csr')
    rowsum = np.array(adj_with_self.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj_with_self @ d_mat_inv_sqrt
    return adj_normalized.tocsr()


# ---------------------------------------------------------------------------
# Reddit dataset loader
# ---------------------------------------------------------------------------

def _download_reddit(data_dir: str) -> None:
    """Download and extract Reddit dataset from DGL if not present."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "reddit.zip")

    # Check if already extracted
    data_file = os.path.join(data_dir, "reddit_data.npz")
    graph_file = os.path.join(data_dir, "reddit_graph.npz")
    if os.path.exists(data_file) and os.path.exists(graph_file):
        print("[Loader] Reddit dataset already present.")
        return

    # Download
    if not os.path.exists(zip_path):
        print(f"[Loader] Downloading Reddit dataset from {REDDIT_URL} ...")
        urllib.request.urlretrieve(REDDIT_URL, zip_path)
        print(f"[Loader] Download complete: {zip_path}")

    # Extract
    print("[Loader] Extracting Reddit dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_dir)
    print("[Loader] Extraction complete.")


def load_reddit_dataset(data_dir: str = None, normalize: bool = True):
    """
    Load the Reddit dataset.
    232,965 nodes, ~114M edges, 602 features, 41 classes.

    Returns:
        adj: scipy.sparse.csr_matrix - normalized adjacency matrix
        features: np.ndarray (N, 602) - node features
        labels: np.ndarray (N,) - node labels (0-40)
        train_mask: np.ndarray (N,) bool
        val_mask: np.ndarray (N,) bool
        test_mask: np.ndarray (N,) bool
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # Download if needed
    _download_reddit(data_dir)

    data_path = os.path.join(data_dir, "reddit_data.npz")
    graph_path = os.path.join(data_dir, "reddit_graph.npz")

    # Load node data
    print("[Loader] Loading Reddit node data...")
    data = np.load(data_path)
    features = data['feature'].astype(np.float32)
    labels = data['label'].astype(np.int64)
    node_types = data['node_types'].astype(np.int64)

    # Build masks from node_types: 1=train, 2=val, 3=test (DGL convention)
    train_mask = (node_types == 1)
    val_mask = (node_types == 2)
    test_mask = (node_types == 3)

    # Normalize features
    if normalize:
        features = _normalize_features(features)

    # Load graph
    print("[Loader] Loading Reddit graph...")
    graph_data = np.load(graph_path)

    # Build sparse adjacency matrix
    if 'row' in graph_data and 'col' in graph_data:
        row = graph_data['row']
        col = graph_data['col']
        num_nodes = features.shape[0]
        adj = sp.csr_matrix(
            (np.ones(len(row), dtype=np.float32), (row, col)),
            shape=(num_nodes, num_nodes)
        )
    elif 'data' in graph_data and 'indices' in graph_data and 'indptr' in graph_data:
        adj = sp.csr_matrix(
            (graph_data['data'], graph_data['indices'], graph_data['indptr']),
            shape=(features.shape[0], features.shape[0])
        )
    else:
        raise ValueError(
            f"Unexpected graph file format. Keys: {list(graph_data.keys())}"
        )

    # Make symmetric (undirected)
    adj = adj + adj.T
    adj[adj > 1] = 1
    adj = adj.tocsr()

    # Normalize adjacency
    if normalize:
        adj = _normalize_adjacency(adj)

    num_nodes = features.shape[0]
    num_edges = adj.nnz
    num_classes = len(np.unique(labels))
    print(f"[Loader] Reddit dataset loaded: {num_nodes} nodes, {num_edges} edges, "
          f"{features.shape[1]} features, {num_classes} classes")
    print(f"[Loader] Train: {train_mask.sum()}, Val: {val_mask.sum()}, "
          f"Test: {test_mask.sum()}")

    return adj, features, labels, train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Planetoid loader (PubMed / Cora / Citeseer)
# ---------------------------------------------------------------------------

PLANETOID_BASE_URL = "https://github.com/kimiyoung/planetoid/raw/master/data"


def _download_planetoid(dataset_name: str, data_dir: str) -> None:
    """Download Planetoid dataset files if not already present."""
    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    suffixes = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index']
    for suffix in suffixes:
        filename = f"ind.{dataset_name}.{suffix}"
        filepath = os.path.join(dataset_dir, filename)
        if not os.path.exists(filepath):
            url = f"{PLANETOID_BASE_URL}/ind.{dataset_name}.{suffix}"
            print(f"[Loader] Downloading {filename} ...")
            urllib.request.urlretrieve(url, filepath)

    print(f"[Loader] {dataset_name.capitalize()} dataset files ready.")


def _load_planetoid_file(filepath: str):
    """Load a single Planetoid pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def load_planetoid_dataset(dataset_name: str = "pubmed",
                           data_dir: str = None,
                           normalize: bool = True):
    """
    Load a Planetoid dataset (PubMed, Cora, or Citeseer).

    Returns:
        adj, features, labels, train_mask, val_mask, test_mask
    """
    if data_dir is None:
        data_dir = DATA_DIR

    _download_planetoid(dataset_name, data_dir)
    dataset_dir = os.path.join(data_dir, dataset_name)
    prefix = os.path.join(dataset_dir, f"ind.{dataset_name}")

    x = _load_planetoid_file(f"{prefix}.x")
    y = _load_planetoid_file(f"{prefix}.y")
    tx = _load_planetoid_file(f"{prefix}.tx")
    ty = _load_planetoid_file(f"{prefix}.ty")
    allx = _load_planetoid_file(f"{prefix}.allx")
    ally = _load_planetoid_file(f"{prefix}.ally")
    graph = _load_planetoid_file(f"{prefix}.graph")

    with open(f"{prefix}.test.index", 'r') as f:
        test_idx = [int(line.strip()) for line in f]
    test_idx = np.array(test_idx)
    test_idx_sorted = np.sort(test_idx)

    if dataset_name == 'citeseer':
        test_idx_range = np.arange(min(test_idx), max(test_idx) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range), tx.shape[1]))
        ty_extended = np.zeros((len(test_idx_range), ty.shape[1]))
        tx_extended[test_idx_sorted - min(test_idx)] = tx
        ty_extended[test_idx_sorted - min(test_idx)] = ty
        tx = tx_extended.tocsr()
        ty = ty_extended

    if sp.issparse(allx):
        allx = allx.toarray()
    if sp.issparse(tx):
        tx = tx.toarray()
    if sp.issparse(x):
        x = x.toarray()

    features = np.vstack([allx, tx]).astype(np.float32)
    labels_onehot = np.vstack([ally, ty])
    labels = np.argmax(labels_onehot, axis=1).astype(np.int64)

    num_nodes = features.shape[0]
    features_reordered = np.copy(features)
    labels_reordered = np.copy(labels)

    num_allx = allx.shape[0]
    for i, idx in enumerate(test_idx_sorted):
        if idx < num_nodes:
            features_reordered[idx] = tx[i]
            labels_reordered[idx] = np.argmax(ty[i])

    features = features_reordered
    labels = labels_reordered

    edges_src = []
    edges_dst = []
    for src, neighbors in graph.items():
        for dst in neighbors:
            if src < num_nodes and dst < num_nodes:
                edges_src.append(src)
                edges_dst.append(dst)

    adj = sp.csr_matrix(
        (np.ones(len(edges_src), dtype=np.float32),
         (edges_src, edges_dst)),
        shape=(num_nodes, num_nodes)
    )
    adj = adj + adj.T
    adj[adj > 1] = 1
    adj = adj.tocsr()

    num_train = x.shape[0]
    num_val = 500
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[:num_train] = True
    val_mask[num_train:num_train + num_val] = True
    test_mask[test_idx_sorted] = True

    if normalize:
        features = _normalize_features(features)
    if normalize:
        adj = _normalize_adjacency(adj)

    num_classes = len(np.unique(labels))
    print(f"[Loader] {dataset_name.capitalize()} loaded: "
          f"{num_nodes} nodes, {adj.nnz} edges, "
          f"{features.shape[1]} features, {num_classes} classes")
    print(f"[Loader] Train: {train_mask.sum()}, Val: {val_mask.sum()}, "
          f"Test: {test_mask.sum()}")

    return adj, features, labels, train_mask, val_mask, test_mask


# Convenience aliases
def load_pubmed_dataset(data_dir=None, normalize=True):
    """Load PubMed dataset."""
    return load_planetoid_dataset("pubmed", data_dir, normalize)


def load_cora_dataset(data_dir=None, normalize=True):
    """Load Cora dataset."""
    return load_planetoid_dataset("cora", data_dir, normalize)


# ---------------------------------------------------------------------------
# Karate Club (for unit testing)
# ---------------------------------------------------------------------------

def load_karate_dataset(normalize: bool = True):
    """
    Load Zachary's Karate Club graph for testing.
    34 nodes, 78 edges, 2 classes (community labels).
    """
    G = nx.karate_club_graph()
    num_nodes = G.number_of_nodes()

    adj = nx.adjacency_matrix(G).astype(np.float32)
    features = np.eye(num_nodes, dtype=np.float32)
    labels = np.array([
        0 if G.nodes[i]['club'] == 'Mr. Hi' else 1
        for i in range(num_nodes)
    ], dtype=np.int64)

    rng = np.random.RandomState(42)
    indices = rng.permutation(num_nodes)
    n_train = int(0.6 * num_nodes)
    n_val = int(0.2 * num_nodes)

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[indices[:n_train]] = True
    val_mask[indices[n_train:n_train + n_val]] = True
    test_mask[indices[n_train + n_val:]] = True

    if normalize:
        features = _normalize_features(features)
        adj = _normalize_adjacency(adj)

    print(f"[Loader] Karate Club loaded: {num_nodes} nodes, {adj.nnz} edges, 2 classes")
    return adj, features, labels, train_mask, val_mask, test_mask
