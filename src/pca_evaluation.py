"""
PCA-Based Evaluation of Synthetic Data Quality for HDLSS.

Methodology (Advisor's spec):
1. Center real data (subtract mean)
2. Compute covariance matrix & eigen decomposition
3. Project real data onto top 2/3 eigenvectors
4. Project synthetic data using the SAME mean and eigenvectors from real data
5. Compare projections visually (colored by cancer type)

Usage:
    python src/HDLSS/pca_evaluation.py
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import src.config as cfg
MODELS = ["smote", "gaussian_copula", "ctgan", "tvae", "tabddpm", "flow_matching", "scgft"]
from src.config import synthetic_file, synthetic_labels_file, TRAIN_RATIO, SEED
from src.data_utils import load_dataset, stratified_split, load_synthetic



plt.style.use("seaborn-v0_8-whitegrid")

def get_colors():
    colors = plt.cm.tab10.colors
    return {c_type: colors[i % len(colors)] for i, c_type in enumerate(cfg.CANCER_TYPES)}


def pca_from_scratch(X_real):
    """Learn PCA basis from real data only."""
    N, P = X_real.shape
    mean_real = np.mean(X_real, axis=0)
    X_centered = X_real - mean_real
    
    # Gram matrix trick since N (640) < P (20531)
    C_small = X_centered @ X_centered.T / (N - 1)
    eigenvalues_small, eigenvectors_small = np.linalg.eigh(C_small)
    
    idx = np.argsort(eigenvalues_small)[::-1]
    eigenvalues_small = eigenvalues_small[idx]
    eigenvectors_small = eigenvectors_small[:, idx]
    
    pos_mask = eigenvalues_small > 1e-10
    eigenvalues = eigenvalues_small[pos_mask]
    eigenvectors = X_centered.T @ eigenvectors_small[:, pos_mask]
    
    norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
    eigenvectors = eigenvectors / norms
    
    total_var = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_var if total_var > 0 else eigenvalues
    
    return mean_real, eigenvectors, eigenvalues, explained_variance_ratio


def project_data(X, mean_real, eigenvectors, n_components=2):
    """Project data onto the top eigenvectors learned from real data."""
    X_centered = X - mean_real
    return X_centered @ eigenvectors[:, :n_components]


def plot_2d(Z_real, y_real, Z_syn, y_syn, model_name, explained_var, output_path):
    """Create 2D PCA scatter plot overlaying real and synthetic data."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Real (in background)
    for c_type in cfg.CANCER_TYPES:
        mask = (y_real == c_type)
        ax.scatter(Z_real[mask, 0], Z_real[mask, 1], color=get_colors()[c_type], 
                    alpha=0.3, s=60, marker='o', edgecolors='white', linewidths=0.5, label=f'Real {c_type}')
    
    # Plot Synthetic (on top)
    for c_type in cfg.CANCER_TYPES:
        mask = (y_syn == c_type)
        ax.scatter(Z_syn[mask, 0], Z_syn[mask, 1], color=get_colors()[c_type], 
                    alpha=0.8, s=50, marker='^', edgecolors='white', linewidths=0.5, label=f'Syn {c_type}')
    
    ax.set_title(f"Real vs Synthetic Data Overlap ({model_name})", fontsize=15, fontweight='bold')
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=12)
    
    # Move legend outside the plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_3d(Z_real, y_real, Z_syn, y_syn, model_name, explained_var, output_path):
    """Create 3D PCA scatter plot overlaying real and synthetic data."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Real (in background)
    for c_type in cfg.CANCER_TYPES:
        mask = (y_real == c_type)
        ax.scatter(Z_real[mask, 0], Z_real[mask, 1], Z_real[mask, 2], color=get_colors()[c_type], 
                    alpha=0.2, s=60, marker='o', edgecolors='white', linewidths=0.5, label=f'Real {c_type}')
                    
    # Plot Synthetic (on top)
    for c_type in cfg.CANCER_TYPES:
        mask = (y_syn == c_type)
        ax.scatter(Z_syn[mask, 0], Z_syn[mask, 1], Z_syn[mask, 2], color=get_colors()[c_type], 
                    alpha=0.8, s=50, marker='^', edgecolors='white', linewidths=0.5, label=f'Syn {c_type}')
                    
    ax.set_title(f"Real vs Synthetic Data Overlap 3D ({model_name})", fontsize=15, fontweight='bold')
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({explained_var[2]*100:.1f}%)')
    
    # Move legend outside the plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    (cfg.RESULTS_DIR / "pca_plots").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HDLSS PCA-BASED EVALUATION")
    print("=" * 60)
    
    # ── 1. Load Real Data ──
    try:
        X, y, _, _ = load_dataset(cfg.DATA_DIR)
        X_real, _, y_real, _ = stratified_split(X, y, TRAIN_RATIO, SEED)
        y_real = y_real.astype(str)
    except FileNotFoundError:
        print(f"ERROR: Could not find HDLSS data in {cfg.DATA_DIR}")
        return
        
    print(f"  Real data loaded: {X_real.shape}")
    
    # ── 2. Learn PCA from REAL data only ──
    mean_real, eigenvectors, eigenvalues, explained_var = pca_from_scratch(X_real)
    
    print(f"  Top 3 eigenvalues: {eigenvalues[:3]}")
    print(f"  Explained variance (PC1): {explained_var[0]*100:.1f}%")
    print(f"  Explained variance (PC1+PC2): {sum(explained_var[:2])*100:.1f}%")
    
    Z_real_2d = project_data(X_real, mean_real, eigenvectors, n_components=2)
    Z_real_3d = project_data(X_real, mean_real, eigenvectors, n_components=3)
    
    pca_metrics = []
    
    # ── 3. Project Synthetic Data ──
    models = ["smote", "gaussian_copula", "ctgan", "tvae", "tabddpm", "flow_matching", "scgft"]
    for model in models:
        syn_path = synthetic_file(model)
        lbl_path = synthetic_labels_file(model)
        
        if not syn_path.exists():
            print(f"  SKIP: {model} (synthetic data not found)")
            continue
            
        X_syn, y_syn = load_synthetic(syn_path, lbl_path)
        y_syn = y_syn.astype(str)
        print(f"  {model}: loaded {X_syn.shape}")
        
        Z_syn_2d = project_data(X_syn, mean_real, eigenvectors, n_components=2)
        Z_syn_3d = project_data(X_syn, mean_real, eigenvectors, n_components=3)
        
        # Calculate centroid metrics per class
        class_metrics = {}
        for c_type in cfg.CANCER_TYPES:
            r_mask = (y_real == c_type)
            s_mask = (y_syn == c_type)
            if np.any(r_mask) and np.any(s_mask):
                r_cent = np.mean(Z_real_2d[r_mask], axis=0)
                s_cent = np.mean(Z_syn_2d[s_mask], axis=0)
                dist = float(np.linalg.norm(r_cent - s_cent))
                class_metrics[c_type] = round(dist, 4)
                
        pca_metrics.append({
            "model": model,
            "centroid_distances_pc12": class_metrics
        })
        
        # Plots
        plot_2d(Z_real_2d, y_real, Z_syn_2d, y_syn, model, explained_var, 
                (cfg.RESULTS_DIR / "pca_plots") / f"pca_2d_{model}.png")
        plot_3d(Z_real_3d, y_real, Z_syn_3d, y_syn, model, explained_var, 
                (cfg.RESULTS_DIR / "pca_plots") / f"pca_3d_{model}.png")
                
    # Save metrics
    out_dir = cfg.RESULTS_DIR / "pca_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "pca_HDLSS_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(pca_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Plots saved to: {cfg.RESULTS_DIR / 'pca_plots'}")
    print("DONE!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="colon")
    args = parser.parse_args()
    
    # Dynamic config initialization
    import pandas as pd
    import numpy as np
    temp_data_dir = cfg.PROJECT_ROOT / "Data" / args.dataset
    df_l = pd.read_csv(temp_data_dir / "labels.csv", index_col=0)
    col = "Class" if "Class" in df_l.columns else df_l.columns[0]
    discovered_classes = list(np.unique(df_l[col].astype(str).values))
    cfg.set_dataset(args.dataset, discovered_classes)
    
    main()

