"""
Evaluation framework for HDLSS TCGA synthetic data.

Compares synthetic data against held-out evaluation set using:
1. Statistical fidelity (Wasserstein, KS)
2. Sparsity preservation
3. Gene-gene correlation structure
4. Mean vector similarity
5. Distance to Closest Record (DCR) — privacy/memorization
6. UMAP manifold coherence
7. Distribution diagnostics
8. TSTR: Train-on-Synthetic, Test-on-Real (5-class cancer classification)
"""
import json
import logging
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cdist
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ── Classifier definitions for TSTR ──
CLASSIFIER_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "gradient_boosting",
    "naive_bayes",
    "svm",
    "knn",
]


def _make_classifier(name: str, device: str = "cpu"):
    """Instantiate a classifier by name with HDLSS-appropriate defaults."""
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            max_iter=2000, random_state=42, n_jobs=-1,
            multi_class="multinomial",
        )
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200, max_features="sqrt",
            random_state=42, n_jobs=-1,
        )
    elif name == "xgboost":
        from xgboost import XGBClassifier
        use_gpu = device.startswith("cuda")
        return XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.3,
            device="cuda" if use_gpu else "cpu",
            random_state=42, eval_metric="logloss", verbosity=0,
        )
    elif name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, max_features="sqrt", random_state=42,
        )
    elif name == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    elif name == "svm":
        from sklearn.svm import SVC
        return SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=42, cache_size=1000,
            decision_function_shape="ovr",
        )
    elif name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(
            n_neighbors=5, metric="euclidean", n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


# ══════════════════════════════════════════════════════════════════════
# STATISTICAL FIDELITY EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_fidelity(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    X_syn: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Compute statistical fidelity metrics between synthetic and eval data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating fidelity: {model_name}...")
    logger.info(f"  Train: {X_train.shape}, Eval: {X_eval.shape}, Syn: {X_syn.shape}")

    P = X_train.shape[1]
    metrics = {
        "model": model_name,
        "n_train": int(X_train.shape[0]),
        "n_eval": int(X_eval.shape[0]),
        "n_syn": int(X_syn.shape[0]),
        "n_genes": int(P),
    }

    # ── 1. Basic statistics ──
    logger.info("  Computing basic statistics...")
    metrics["train_mean"] = float(np.mean(X_train))
    metrics["eval_mean"] = float(np.mean(X_eval))
    metrics["syn_mean"] = float(np.mean(X_syn))
    metrics["train_std"] = float(np.std(X_train))
    metrics["eval_std"] = float(np.std(X_eval))
    metrics["syn_std"] = float(np.std(X_syn))

    # ── 2. Sparsity preservation ──
    logger.info("  Computing sparsity...")
    metrics["train_sparsity"] = float(np.mean(X_train == 0) * 100)
    metrics["eval_sparsity"] = float(np.mean(X_eval == 0) * 100)
    metrics["syn_sparsity"] = float(np.mean(X_syn == 0) * 100)
    metrics["sparsity_ratio"] = float(
        metrics["syn_sparsity"] / max(metrics["eval_sparsity"], 1e-10)
    )
    logger.info(f"    Train: {metrics['train_sparsity']:.1f}%, "
                f"Eval: {metrics['eval_sparsity']:.1f}%, "
                f"Syn: {metrics['syn_sparsity']:.1f}%")

    # ── 3. Per-gene Wasserstein distance ──
    logger.info("  Computing Wasserstein distances...")
    wd_list = []
    for j in range(P):
        wd = stats.wasserstein_distance(X_eval[:, j], X_syn[:, j])
        wd_list.append(wd)
    wd_arr = np.array(wd_list, dtype=np.float64)
    metrics["wasserstein_mean"] = float(np.mean(wd_arr))
    metrics["wasserstein_median"] = float(np.median(wd_arr))
    metrics["wasserstein_std"] = float(np.std(wd_arr))
    metrics["wasserstein_p95"] = float(np.percentile(wd_arr, 95))
    logger.info(f"    Mean WD: {metrics['wasserstein_mean']:.6f}")

    # ── 4. KS-based likeness ──
    logger.info("  Computing KS likeness...")
    ks_list = []
    for j in range(P):
        ks_stat, _ = stats.ks_2samp(X_eval[:, j], X_syn[:, j])
        ks_list.append(ks_stat)
    ks_arr = np.array(ks_list, dtype=np.float64)
    metrics["ks_likeness"] = float(1.0 - np.mean(ks_arr))
    metrics["ks_stat_mean"] = float(np.mean(ks_arr))
    metrics["ks_stat_median"] = float(np.median(ks_arr))
    logger.info(f"    KS Likeness: {metrics['ks_likeness']:.4f}")

    # ── 5. Mean vector similarity ──
    logger.info("  Computing mean vector similarity...")
    eval_mean_vec = np.mean(X_eval, axis=0)
    syn_mean_vec = np.mean(X_syn, axis=0)
    metrics["mean_l2_distance"] = float(np.linalg.norm(eval_mean_vec - syn_mean_vec))
    cos_sim = np.dot(eval_mean_vec, syn_mean_vec) / (
        np.linalg.norm(eval_mean_vec) * np.linalg.norm(syn_mean_vec) + 1e-10
    )
    metrics["mean_cosine_similarity"] = float(cos_sim)
    logger.info(f"    L2: {metrics['mean_l2_distance']:.4f}, "
                f"Cosine: {metrics['mean_cosine_similarity']:.6f}")

    # ── 6. Gene-gene correlation preservation ──
    logger.info("  Computing gene-gene correlation preservation...")
    try:
        corr_metrics = _gene_correlation_preservation(X_eval, X_syn, top_k=500)
        metrics.update(corr_metrics)
        logger.info(f"    Meta-correlation: {metrics.get('meta_correlation', 'N/A')}")
    except Exception as e:
        logger.warning(f"    Gene correlation failed: {e}")
        metrics["meta_correlation"] = None

    # ── 7. DCR (Distance to Closest Record) ──
    logger.info("  Computing DCR metrics...")
    try:
        nn_metrics = _nearest_neighbor_metrics(X_train, X_eval, X_syn)
        metrics.update(nn_metrics)
        logger.info(f"    DCR ratio: {metrics.get('dcr_ratio', 'N/A'):.4f}")
    except Exception as e:
        logger.warning(f"    DCR failed: {e}")

    # ── 8. UMAP visualization ──
    logger.info("  Generating UMAP plot...")
    try:
        _plot_umap(X_eval, X_syn, model_name, output_dir)
        metrics["umap_plot"] = str(output_dir / "umap.png")
    except Exception as e:
        logger.warning(f"    UMAP failed: {e}")

    # ── 9. Distribution plots ──
    logger.info("  Generating distribution plots...")
    try:
        _plot_distributions(X_eval, X_syn, model_name, output_dir)
    except Exception as e:
        logger.warning(f"    Distribution plots failed: {e}")

    # Save metrics JSON
    metrics_path = output_dir / "fidelity_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"  Fidelity metrics saved to {metrics_path}")

    return metrics


# ══════════════════════════════════════════════════════════════════════
# TSTR EVALUATION (5-class cancer type classification)
# ══════════════════════════════════════════════════════════════════════

def evaluate_tstr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    model_name: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run TRTR + TSTR 5-class cancer classification comparison.

    TRTR: Train on real, test on real (baseline)
    TSTR: Train on synthetic, test on real (quality measure)

    If TSTR ≈ TRTR, synthetic data preserves discriminative structure.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
    )
    from sklearn.preprocessing import LabelEncoder

    logger.info(f"TSTR Evaluation: {model_name}")
    logger.info(f"  Train: {X_train.shape}, labels={np.unique(y_train, return_counts=True)}")
    logger.info(f"  Eval:  {X_eval.shape}, labels={np.unique(y_eval, return_counts=True)}")
    logger.info(f"  Syn:   {X_syn.shape}, labels={np.unique(y_syn, return_counts=True)}")

    # Encode labels to integers for classifiers
    # Normalize all labels to strings for consistent LabelEncoder behavior
    y_train = y_train.astype(str)
    y_eval = y_eval.astype(str)
    y_syn = y_syn.astype(str)

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_eval, y_syn]))
    y_train_enc = le.transform(y_train)
    y_eval_enc = le.transform(y_eval)
    y_syn_enc = le.transform(y_syn)

    results = {
        "model": model_name,
        "n_classes": int(len(le.classes_)),
        "class_names": list(le.classes_),
        "n_train": int(len(y_train)),
        "n_eval": int(len(y_eval)),
        "n_syn": int(len(y_syn)),
        "train_class_dist": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "eval_class_dist": {str(k): int(v) for k, v in zip(*np.unique(y_eval, return_counts=True))},
        "syn_class_dist": {str(k): int(v) for k, v in zip(*np.unique(y_syn, return_counts=True))},
        "classifiers": {},
    }

    for clf_name in CLASSIFIER_NAMES:
        logger.info(f"    Running {clf_name}...")

        # ── TRTR: Train Real, Test Real ──
        t0 = time.time()
        try:
            clf_trtr = _make_classifier(clf_name, device)
            clf_trtr.fit(X_train, y_train_enc)
            y_pred_trtr = clf_trtr.predict(X_eval)
            trtr_acc = float(accuracy_score(y_eval_enc, y_pred_trtr))
            trtr_f1 = float(f1_score(y_eval_enc, y_pred_trtr, average="weighted", zero_division=0))
            trtr_prec = float(precision_score(y_eval_enc, y_pred_trtr, average="weighted", zero_division=0))
            trtr_rec = float(recall_score(y_eval_enc, y_pred_trtr, average="weighted", zero_division=0))
            trtr_time = round(time.time() - t0, 2)
            trtr = {"accuracy": trtr_acc, "f1": trtr_f1, "precision": trtr_prec,
                     "recall": trtr_rec, "train_time_s": trtr_time}
        except Exception as e:
            logger.warning(f"      TRTR {clf_name} failed: {e}")
            trtr = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                     "train_time_s": 0.0, "error": str(e)}

        # ── TSTR: Train Synthetic, Test Real ──
        t0 = time.time()
        try:
            clf_tstr = _make_classifier(clf_name, device)
            clf_tstr.fit(X_syn, y_syn_enc)
            y_pred_tstr = clf_tstr.predict(X_eval)
            tstr_acc = float(accuracy_score(y_eval_enc, y_pred_tstr))
            tstr_f1 = float(f1_score(y_eval_enc, y_pred_tstr, average="weighted", zero_division=0))
            tstr_prec = float(precision_score(y_eval_enc, y_pred_tstr, average="weighted", zero_division=0))
            tstr_rec = float(recall_score(y_eval_enc, y_pred_tstr, average="weighted", zero_division=0))
            tstr_time = round(time.time() - t0, 2)
            tstr = {"accuracy": tstr_acc, "f1": tstr_f1, "precision": tstr_prec,
                     "recall": tstr_rec, "train_time_s": tstr_time}
        except Exception as e:
            logger.warning(f"      TSTR {clf_name} failed: {e}")
            tstr = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                     "train_time_s": 0.0, "error": str(e)}

        results["classifiers"][clf_name] = {
            "trtr": trtr,
            "tstr": tstr,
            "tstr_trtr_accuracy_ratio": round(
                tstr["accuracy"] / max(trtr["accuracy"], 1e-10), 4
            ),
            "tstr_trtr_f1_ratio": round(
                tstr["f1"] / max(trtr["f1"], 1e-10), 4
            ),
        }

        logger.info(f"      TRTR: acc={trtr['accuracy']:.3f} f1={trtr['f1']:.3f}")
        logger.info(f"      TSTR: acc={tstr['accuracy']:.3f} f1={tstr['f1']:.3f}")

    return results


# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def _gene_correlation_preservation(
    X_eval: np.ndarray,
    X_syn: np.ndarray,
    top_k: int = 500,
) -> Dict[str, float]:
    """Compare gene-gene correlation structure."""
    gene_var = np.var(X_eval, axis=0)
    top_genes = np.argsort(gene_var)[-top_k:]

    X_e = X_eval[:, top_genes]
    X_s = X_syn[:, top_genes]

    corr_eval = np.corrcoef(X_e.T)
    corr_syn = np.corrcoef(X_s.T)

    mask = ~(np.isnan(corr_eval) | np.isnan(corr_syn))
    triu = np.triu_indices(top_k, k=1)
    mask_triu = mask[triu]

    corr_e_flat = corr_eval[triu][mask_triu]
    corr_s_flat = corr_syn[triu][mask_triu]

    if len(corr_e_flat) < 10:
        return {"meta_correlation": None, "n_correlation_pairs": 0}

    meta_corr = np.corrcoef(corr_e_flat, corr_s_flat)[0, 1]

    return {
        "meta_correlation": float(meta_corr),
        "n_correlation_pairs": int(len(corr_e_flat)),
        "corr_rmse": float(np.sqrt(np.mean((corr_e_flat - corr_s_flat) ** 2))),
    }


def _nearest_neighbor_metrics(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    X_syn: np.ndarray,
    sample_size: int = 200,
) -> Dict[str, float]:
    """Compute DCR metrics for memorization/privacy check."""
    N = X_syn.shape[0]
    if N > sample_size:
        idx = np.random.choice(N, sample_size, replace=False)
        X_syn_sub = X_syn[idx]
    else:
        X_syn_sub = X_syn

    N_e = X_eval.shape[0]
    if N_e > sample_size:
        idx = np.random.choice(N_e, sample_size, replace=False)
        X_eval_sub = X_eval[idx]
    else:
        X_eval_sub = X_eval

    dist_syn_train = cdist(X_syn_sub, X_train, metric="euclidean")
    dcr_syn = np.min(dist_syn_train, axis=1)

    dist_eval_train = cdist(X_eval_sub, X_train, metric="euclidean")
    dcr_eval = np.min(dist_eval_train, axis=1)

    return {
        "dcr_syn_to_train_mean": float(np.mean(dcr_syn)),
        "dcr_syn_to_train_median": float(np.median(dcr_syn)),
        "dcr_eval_to_train_mean": float(np.mean(dcr_eval)),
        "dcr_eval_to_train_median": float(np.median(dcr_eval)),
        "dcr_ratio": float(np.mean(dcr_syn) / (np.mean(dcr_eval) + 1e-10)),
    }


def _plot_umap(
    X_eval: np.ndarray,
    X_syn: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> None:
    """Generate UMAP visualization of eval vs synthetic."""
    import umap

    combined = np.vstack([X_eval, X_syn])
    labels = ["Eval"] * len(X_eval) + ["Synthetic"] * len(X_syn)

    reducer = umap.UMAP(n_neighbors=min(15, len(combined) - 1), random_state=42)
    embedding = reducer.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(8, 6))
    n_eval = len(X_eval)

    ax.scatter(
        embedding[:n_eval, 0], embedding[:n_eval, 1],
        c="steelblue", alpha=0.7, s=30, label="Eval (real)", edgecolors="none"
    )
    ax.scatter(
        embedding[n_eval:, 0], embedding[n_eval:, 1],
        c="coral", alpha=0.7, s=30, label="Synthetic", edgecolors="none"
    )
    ax.set_title(f"HDLSS {model_name}")
    ax.legend()
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(output_dir / "umap.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_distributions(
    X_eval: np.ndarray,
    X_syn: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> None:
    """Generate distribution comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Per-sample mean expression
    eval_sample_means = np.mean(X_eval, axis=1)
    syn_sample_means = np.mean(X_syn, axis=1)
    axes[0].hist(eval_sample_means, bins=20, alpha=0.6, label="Eval", color="steelblue", density=True)
    axes[0].hist(syn_sample_means, bins=20, alpha=0.6, label="Synthetic", color="coral", density=True)
    axes[0].set_title("Per-sample Mean Expression")
    axes[0].legend()

    # Per-sample non-zero count
    eval_nnz = np.sum(X_eval > 0, axis=1)
    syn_nnz = np.sum(X_syn > 0, axis=1)
    axes[1].hist(eval_nnz, bins=20, alpha=0.6, label="Eval", color="steelblue", density=True)
    axes[1].hist(syn_nnz, bins=20, alpha=0.6, label="Synthetic", color="coral", density=True)
    axes[1].set_title("Per-sample Non-zero Gene Count")
    axes[1].legend()

    # Overall value distribution (non-zeros only)
    eval_nz = X_eval[X_eval > 0]
    syn_nz = X_syn[X_syn > 0]
    if len(eval_nz) > 5000:
        eval_nz = np.random.choice(eval_nz, 5000, replace=False)
    if len(syn_nz) > 5000:
        syn_nz = np.random.choice(syn_nz, 5000, replace=False)
    axes[2].hist(eval_nz, bins=50, alpha=0.6, label="Eval", color="steelblue", density=True)
    axes[2].hist(syn_nz, bins=50, alpha=0.6, label="Synthetic", color="coral", density=True)
    axes[2].set_title("Non-zero Expression Distribution")
    axes[2].legend()

    fig.suptitle(f"HDLSS {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
