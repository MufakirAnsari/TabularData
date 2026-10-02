"""
Model 7: scGFT — Generative Fourier Transform for PANCAN.

Adapted from scRNA-seq pipeline with all Run 1 fixes:
- ncpmnts=100 (prevents memorization)
- Genes sorted by mean expression before DFT
- Zero-threshold post-processing
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def zero_threshold_postprocess(X_syn, X_train):
    """Match per-gene sparsity of training data."""
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
    scGFT (Fourier-based) generator for PANCAN data.
    """
    np.random.seed(seed)
    N, D = X_train.shape
    ncpmnts = 100

    logger.info(f"PANCAN scGFT: N={N}, D={D}, ncpmnts={ncpmnts}, generating {n_samples} samples")

    # Sort genes by mean expression for meaningful DFT signal
    gene_means = np.mean(X_train, axis=0)
    sort_order = np.argsort(gene_means)[::-1]
    unsort_order = np.argsort(sort_order)

    X_sorted = X_train[:, sort_order]

    synthetic_samples = np.zeros((n_samples, D), dtype=np.float32)

    for i in range(n_samples):
        real_idx = i % N
        x_i = X_sorted[real_idx]

        F = np.fft.rfft(x_i)
        A = np.abs(F)
        phi = np.angle(F)

        for k in range(1, min(ncpmnts + 1, len(F))):
            phi[k] += np.random.uniform(-np.pi/4, np.pi/4)
            A[k] *= np.random.uniform(0.9, 1.1)

        F_perturbed = A * np.exp(1j * phi)
        x_synthetic = np.fft.irfft(F_perturbed, n=D)
        x_synthetic = np.maximum(0, x_synthetic)

        synthetic_samples[i] = x_synthetic[unsort_order].astype(np.float32)

    synthetic_samples = zero_threshold_postprocess(synthetic_samples, X_train)

    logger.info(f"PANCAN scGFT: Generated {n_samples} samples, "
                f"sparsity={np.mean(synthetic_samples == 0) * 100:.1f}%")
    return synthetic_samples
