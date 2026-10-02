"""
Data loading and splitting utilities for the PANCAN TCGA pipeline.
Handles CSV loading, stratified train/eval splitting, per-class extraction.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    data_path = data_dir / "data.csv"
    labels_path = data_dir / "labels.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    logger.info(f"Loading data from {data_dir}...")

    # Load expression data
    df_data = pd.read_csv(data_path, index_col=0)
    X = df_data.values.astype(np.float32)
    gene_names = list(df_data.columns)
    sample_ids = list(df_data.index)

    # Load labels (assume last column or 'Class' column)
    df_labels = pd.read_csv(labels_path, index_col=0)
    if "Class" in df_labels.columns:
        y = df_labels["Class"].values.astype(str)
    else:
        y = df_labels.iloc[:, 0].values.astype(str)

    # Verify alignment
    assert len(X) == len(y), f"Sample count mismatch: X={len(X)}, y={len(y)}"

    logger.info(f"  Shape: {X.shape[0]} samples x {X.shape[1]} features")
    logger.info(f"  dtype: {X.dtype}, range: [{X.min():.4f}, {X.max():.4f}]")
    logger.info(f"  Sparsity: {np.mean(X == 0) * 100:.1f}% zeros")
    logger.info(f"  Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, sample_ids, gene_names


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/eval split preserving class proportions.

    Returns
    -------
    X_train, X_eval, y_train, y_eval
    """
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, train_size=train_ratio, random_state=seed, stratify=y
    )
    logger.info(f"  Train: {X_train.shape}, Eval: {X_eval.shape}")
    logger.info(f"  Train classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logger.info(f"  Eval  classes: {dict(zip(*np.unique(y_eval, return_counts=True)))}")
    return X_train, X_eval, y_train, y_eval


def get_class_data(
    X: np.ndarray,
    y: np.ndarray,
    class_name: str,
) -> np.ndarray:
    """
    Extract samples belonging to a specific cancer class.

    Returns
    -------
    X_class : np.ndarray, shape (n_class, P)
    """
    mask = y == class_name
    X_class = X[mask]
    logger.info(f"  Class '{class_name}': {X_class.shape[0]} samples")
    return X_class


def save_synthetic(X_syn: np.ndarray, y_syn: np.ndarray, data_path: Path, labels_path: Path) -> None:
    """
    Save synthetic data and labels as numpy files.
    """
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(data_path, X_syn.astype(np.float32))
    np.save(labels_path, y_syn)
    logger.info(f"  Saved synthetic data: {data_path} (shape={X_syn.shape})")
    logger.info(f"  Saved synthetic labels: {labels_path} (n={len(y_syn)})")


def load_synthetic(data_path: Path, labels_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load saved synthetic data and labels."""
    X_syn = np.load(data_path).astype(np.float32)
    y_syn = np.load(labels_path, allow_pickle=True)
    return X_syn, y_syn
