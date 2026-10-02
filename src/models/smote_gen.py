"""
Model 1: SMOTE — Sparse k-NN interpolation for PANCAN.

Adapted from scRNA-seq pipeline with all Run 1 fixes:
- Sparse-SMOTE: re-zero genes zero in BOTH anchor and neighbor
- Zero-threshold post-processing (adapted for bulk RNA-Seq sparsity)
"""
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def zero_threshold_postprocess(X_syn, X_train):
    """Match per-gene sparsity of training data.
    Only applied to genes with >50% zeros in training data."""
    P = X_train.shape[1]
    for j in range(P):
        real_zero_frac = np.mean(X_train[:, j] == 0)
        if real_zero_frac > 0.5:
            threshold = np.percentile(np.abs(X_syn[:, j]), real_zero_frac * 100)
            X_syn[:, j] = np.where(np.abs(X_syn[:, j]) < threshold, 0.0, X_syn[:, j])
    return X_syn


def generate(
    X_train: np.ndarray,
    n_samples: int,
    seed: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate synthetic data using Sparse-SMOTE interpolation.
    """
    rng = np.random.RandomState(seed)
    N, P = X_train.shape

    # Guard: if only 1 sample, use jittered replication
    if N <= 1:
        logger.warning(f"SMOTE: Only {N} training sample(s). Using jittered replication.")
        X_syn = np.tile(X_train, (n_samples, 1)) if N == 1 else np.zeros((n_samples, P), dtype=np.float32)
        if N == 1:
            noise = rng.normal(0, 1e-6, size=(n_samples, P)).astype(np.float32)
            X_syn = (X_syn + noise).astype(np.float32)
            X_syn = np.clip(X_syn, 0.0, None)
        return X_syn

    k = min(5, N - 1)
    logger.info(f"PANCAN Sparse-SMOTE: N={N}, P={P}, k={k}, generating {n_samples} samples")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_train)

    X_syn = np.empty((n_samples, P), dtype=np.float32)
    for i in range(n_samples):
        idx = rng.randint(0, N)
        nn_idx = indices[idx, rng.randint(1, k + 1)]

        lam = rng.uniform(0.0, 1.0)
        X_syn[i] = X_train[idx] + lam * (X_train[nn_idx] - X_train[idx])

        # Sparse-SMOTE: re-zero genes zero in BOTH anchor and neighbor
        both_zero = (X_train[idx] == 0) & (X_train[nn_idx] == 0)
        X_syn[i, both_zero] = 0.0

    X_syn = np.clip(X_syn, 0.0, None)
    X_syn = zero_threshold_postprocess(X_syn, X_train)

    logger.info(f"PANCAN SMOTE: Generated {n_samples} samples, "
                f"sparsity={np.mean(X_syn == 0) * 100:.1f}%")
    return X_syn
