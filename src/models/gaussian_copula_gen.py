"""
Model 2: Gaussian Copula with Ledoit-Wolf Shrinkage for PANCAN.

Adapted from scRNA-seq pipeline with all Run 1 fixes:
- Ledoit-Wolf shrinkage for singular covariance (N < P)
- Cholesky decomposition with jitter fallback
- Round (not truncate) for inverse CDF quantile mapping
- Zero-threshold post-processing
"""
import logging
import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def generate(
    X_train: np.ndarray,
    n_samples: int,
    seed: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate synthetic data using Gaussian Copula + Ledoit-Wolf shrinkage.
    """
    rng = np.random.RandomState(seed)
    N, P = X_train.shape
    logger.info(f"PANCAN GaussianCopula: N={N}, P={P}, generating {n_samples} samples")

    # Identify constant columns
    col_std = np.std(X_train, axis=0)
    variable_mask = col_std > 0
    n_variable = np.sum(variable_mask)
    n_constant = P - n_variable
    logger.info(f"  Variable genes: {n_variable}, Constant genes: {n_constant}")

    X_var = X_train[:, variable_mask]

    # Step 1: Empirical CDF → Uniform [0, 1]
    U = np.zeros_like(X_var)
    for j in range(n_variable):
        col = X_var[:, j]
        ranks = stats.rankdata(col, method="average")
        U[:, j] = ranks / (N + 1)

    # Step 2: Uniform → Normal
    Z = stats.norm.ppf(U)
    Z = np.clip(Z, -5.0, 5.0)

    # Step 3: Ledoit-Wolf shrinkage covariance
    logger.info(f"  Fitting Ledoit-Wolf covariance ({n_variable}×{n_variable})...")
    lw = LedoitWolf()
    lw.fit(Z)
    cov_shrunk = lw.covariance_
    shrinkage = lw.shrinkage_
    logger.info(f"  Shrinkage coefficient: {shrinkage:.4f}")

    # Step 4: Sample via Cholesky
    mean = np.mean(Z, axis=0)
    logger.info(f"  Sampling {n_samples} via Cholesky decomposition...")
    try:
        L = np.linalg.cholesky(cov_shrunk)
        standard_normal = rng.standard_normal((n_samples, n_variable))
        Z_syn = mean[np.newaxis, :] + standard_normal @ L.T
    except np.linalg.LinAlgError:
        logger.warning("  Cholesky failed, adding diagonal jitter...")
        jitter = 1e-6 * np.eye(n_variable)
        L = np.linalg.cholesky(cov_shrunk + jitter)
        standard_normal = rng.standard_normal((n_samples, n_variable))
        Z_syn = mean[np.newaxis, :] + standard_normal @ L.T

    # Step 5: Normal → Uniform
    U_syn = stats.norm.cdf(Z_syn)

    # Step 6: Uniform → Original marginals (inverse empirical CDF)
    X_var_syn = np.zeros((n_samples, n_variable), dtype=np.float32)
    for j in range(n_variable):
        sorted_vals = np.sort(X_var[:, j])
        indices = np.clip(
            np.round(U_syn[:, j] * N).astype(int),
            0, N - 1
        )
        X_var_syn[:, j] = sorted_vals[indices]

    # Step 7: Reconstruct full matrix
    X_syn = np.zeros((n_samples, P), dtype=np.float32)
    X_syn[:, variable_mask] = X_var_syn
    if n_constant > 0:
        const_values = X_train[0, ~variable_mask]
        X_syn[:, ~variable_mask] = const_values[np.newaxis, :]

    X_syn = np.clip(X_syn, 0.0, None)

    # Zero-threshold post-processing
    for j in range(P):
        real_zero_frac = np.mean(X_train[:, j] == 0)
        if real_zero_frac > 0.5:
            threshold = np.percentile(np.abs(X_syn[:, j]), real_zero_frac * 100)
            X_syn[:, j] = np.where(np.abs(X_syn[:, j]) < threshold, 0.0, X_syn[:, j])

    logger.info(f"PANCAN GaussianCopula: Generated {n_samples} samples, "
                f"sparsity={np.mean(X_syn == 0) * 100:.1f}%")
    return X_syn
