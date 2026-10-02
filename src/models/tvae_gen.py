"""
Model 4: TVAE — Tabular Variational Autoencoder for PANCAN (ALL GENES).

Operates on ALL 20,531 genes directly (no HVG subsetting).
Uses MinMaxScaler + VGM bypass for the full gene space.

Run 1 fixes preserved:
- VGM bypass (no categorical encoding / Gaussian mixture bloat)
- Increased compress/decompress dims to (256, 256)
- embedding_dim=256
- MinMaxScaler normalization
- Zero-threshold post-processing
"""
import logging
import warnings
import numpy as np
import pandas as pd

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
    device: str = "cuda",
) -> np.ndarray:
    """
    Generate synthetic data using TVAE on ALL genes.
    VGM bypass ensures no column expansion (20,531 stays 20,531).
    """
    from ctgan import TVAE
    from sklearn.preprocessing import MinMaxScaler

    N, P = X_train.shape
    logger.info(f"PANCAN TVAE (all genes): N={N}, P={P}, generating {n_samples} samples")

    # Scale ALL genes to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)

    col_names = [f"g{i}" for i in range(P)]
    df = pd.DataFrame(X_scaled, columns=col_names)

    batch_size = min(500, N)

    # Set seed for reproducibility
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    from src.config import HPARAMS
    hp = HPARAMS.get("tvae", {})

    logger.info(f"  Training TVAE on {P} features... (batch={batch_size})")

    model = TVAE(
        epochs=hp.get("epochs", 300),
        batch_size=batch_size,
        compress_dims=hp.get("compress_dims", (256, 256)),
        decompress_dims=hp.get("decompress_dims", (256, 256)),
        embedding_dim=hp.get("embedding_dim", 256),
        cuda=device.startswith("cuda"),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df, discrete_columns=[])

    # Generate
    logger.info(f"  Generating {n_samples} samples...")
    df_syn = model.sample(n_samples)
    X_syn_scaled = df_syn.values.astype(np.float32)

    # Inverse transform back to original scale
    X_syn = scaler.inverse_transform(X_syn_scaled).astype(np.float32)

    # Clamp to non-negative
    X_syn = np.clip(X_syn, 0.0, None)

    # Zero-threshold post-processing
    X_syn = zero_threshold_postprocess(X_syn, X_train)

    logger.info(f"PANCAN TVAE: Generated {n_samples} samples, "
                f"sparsity={np.mean(X_syn == 0) * 100:.1f}%")
    return X_syn
