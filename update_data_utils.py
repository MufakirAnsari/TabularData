import os

with open('src/pancan/data_utils.py', 'r') as f:
    content = f.read()

# Make sure it can load generic classes
new_func = '''def load_pancan_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
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
        y = df_labels["Class"].values
    else:
        y = df_labels.iloc[:, 0].values

    # Verify alignment
    assert len(X) == len(y), f"Sample count mismatch: X={len(X)}, y={len(y)}"

    logger.info(f"  Shape: {X.shape[0]} samples x {X.shape[1]} features")
    logger.info(f"  dtype: {X.dtype}, range: [{X.min():.4f}, {X.max():.4f}]")
    logger.info(f"  Sparsity: {np.mean(X == 0) * 100:.1f}% zeros")
    logger.info(f"  Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, sample_ids, gene_names'''

import re
# Replace the original load_pancan_data function
content = re.sub(r'def load_pancan_data.*?return X, y, sample_ids, gene_names', new_func, content, flags=re.DOTALL)

with open('src/pancan/data_utils.py', 'w') as f:
    f.write(content)
