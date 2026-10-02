"""
Model 3: CTGAN — Conditional Tabular GAN for PANCAN (ALL GENES).

Operates on ALL 20,531 genes directly (no HVG subsetting).
Uses MinMaxScaler + VGM bypass for the full gene space.

Run 1 fixes preserved:
- VGM bypass (no categorical encoding / Gaussian mixture bloat)
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
    Generate synthetic data using CTGAN on ALL genes.
    VGM bypass ensures no column expansion (20,531 stays 20,531).
    """
    from ctgan import CTGAN
    from sklearn.preprocessing import MinMaxScaler

    N, P = X_train.shape
    logger.info(f"PANCAN CTGAN (all genes): N={N}, P={P}, generating {n_samples} samples")

    # Scale ALL genes to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)

    col_names = [f"g{i}" for i in range(P)]
    df = pd.DataFrame(X_scaled, columns=col_names)

    # Configure CTGAN — adjust batch/pac for small per-class N
    batch_size = min(500, N)
    pac = min(10, max(1, N // 4))
    batch_size = (batch_size // pac) * pac
    if batch_size < pac:
        batch_size = pac

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
    hp = HPARAMS.get("ctgan", {})

    logger.info(f"  Training CTGAN on {P} features... (batch={batch_size}, pac={pac})")

    model = CTGAN(
        epochs=hp.get("epochs", 300),
        batch_size=batch_size,
        generator_dim=hp.get("generator_dim", (256, 256)),
        discriminator_dim=hp.get("discriminator_dim", (256, 256)),
        generator_lr=hp.get("generator_lr", 2e-4),
        discriminator_lr=hp.get("discriminator_lr", 2e-4),
        discriminator_steps=1,
        pac=pac,
        cuda=device.startswith("cuda"),
        verbose=True,
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

    logger.info(f"PANCAN CTGAN: Generated {n_samples} samples, "
                f"sparsity={np.mean(X_syn == 0) * 100:.1f}%")
    return X_syn
