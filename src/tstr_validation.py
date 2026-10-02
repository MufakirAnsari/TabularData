"""
TRTR vs TSTR Validation for HDLSS Synthetic scRNA-seq Data.

Train on Real, Test on Real (TRTR) — baseline
Train on Synthetic, Test on Real (TSTR) — synthetic data quality measure

If TSTR ≈ TRTR, the synthetic data preserves discriminative structure.

Label creation: binarize n_genes (number of expressed genes per cell)
at the median of the REAL training data. Same threshold applied everywhere.
"""
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# ── Classifier definitions ──
CLASSIFIER_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "gradient_boosting",
    "naive_bayes",
    "svm",
    "knn",
]


def _make_classifier(name: str, n_features: int, device: str = "cpu"):
    """Instantiate a classifier by name with HDLSS-appropriate defaults."""
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            max_iter=1000, random_state=42, n_jobs=-1,
        )
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100,
            max_features="sqrt",  # Important for HDLSS
            random_state=42, n_jobs=-1,
        )
    elif name == "xgboost":
        from xgboost import XGBClassifier
        # Use GPU if available
        use_gpu = device.startswith("cuda")
        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.3,  # Important: subsample features for 22K-D
            device="cuda" if use_gpu else "cpu",
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
    elif name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            max_features="sqrt",
            random_state=42,
        )
    elif name == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    elif name == "svm":
        from sklearn.svm import SVC
        return SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True,  # Needed for AUC-ROC
            random_state=42, cache_size=1000,
        )
    elif name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(
            n_neighbors=5, metric="euclidean", n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


def create_labels(X: np.ndarray, threshold: float) -> np.ndarray:
    """
    Create binary labels from gene count per cell.
    
    Parameters
    ----------
    X : np.ndarray, shape (N, P) — expression matrix
    threshold : float — median n_genes from real training data
    
    Returns
    -------
    y : np.ndarray, shape (N,) — binary labels (0 = low diversity, 1 = high)
    """
    n_genes = np.sum(X > 0, axis=1)  # Count non-zero genes per cell
    return (n_genes >= threshold).astype(np.int32)


def get_real_threshold(adata) -> float:
    """Get median n_genes from the real adata .obs."""
    if "n_genes" in adata.obs.columns:
        return float(np.median(adata.obs["n_genes"].values))
    else:
        # Fallback: compute from expression matrix
        import scipy.sparse as sp
        X = adata.X
        if sp.issparse(X):
            n_genes = np.array((X > 0).sum(axis=1)).flatten()
        else:
            n_genes = np.sum(X > 0, axis=1)
        return float(np.median(n_genes))


def evaluate_classifier(
    clf_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Train a classifier and evaluate on test set.
    
    Returns dict with accuracy, f1, auc_roc, precision, recall, and timing.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        precision_score, recall_score,
    )

    t0 = time.time()
    clf = _make_classifier(clf_name, X_train.shape[1], device)
    
    # Handle edge cases: if only one class in training data
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        logger.warning(f"    {clf_name}: Only 1 class in training data, skipping")
        return {
            "classifier": clf_name,
            "accuracy": 0.5,
            "f1": 0.0,
            "auc_roc": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "train_time_s": 0.0,
            "error": "single_class_in_train",
        }

    clf.fit(X_train, y_train)
    t_train = time.time() - t0

    y_pred = clf.predict(X_test)
    
    # AUC-ROC needs probability scores
    try:
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_prob = clf.decision_function(X_test)
        else:
            y_prob = y_pred.astype(float)
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.5  # Fallback if AUC can't be computed

    return {
        "classifier": clf_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "auc_roc": float(auc),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "train_time_s": round(t_train, 2),
    }


def run_tstr_for_one(
    X_train_real: np.ndarray,
    X_eval: np.ndarray,
    X_syn: np.ndarray,
    threshold: float,
    gen_model: str,
    n_samples: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run full TRTR + TSTR comparison for one (generative model, N) pair.
    
    Parameters
    ----------
    X_train_real : real training data
    X_eval : real evaluation data  
    X_syn : synthetic data
    threshold : median n_genes from real training data
    gen_model : name of the generative model
    n_samples : sample size N
    device : 'cuda' or 'cpu'
    
    Returns
    -------
    results : dict with TRTR and TSTR metrics per classifier
    """
    # Create labels
    y_train = create_labels(X_train_real, threshold)
    y_eval = create_labels(X_eval, threshold)
    y_syn = create_labels(X_syn, threshold)

    logger.info(f"  Labels — Train: {np.sum(y_train==1)}/{len(y_train)} high, "
                f"Eval: {np.sum(y_eval==1)}/{len(y_eval)} high, "
                f"Syn: {np.sum(y_syn==1)}/{len(y_syn)} high")

    results = {
        "gen_model": gen_model,
        "n_samples": n_samples,
        "threshold": float(threshold),
        "n_train": int(len(y_train)),
        "n_eval": int(len(y_eval)),
        "n_syn": int(len(y_syn)),
        "train_class_balance": float(np.mean(y_train)),
        "eval_class_balance": float(np.mean(y_eval)),
        "syn_class_balance": float(np.mean(y_syn)),
        "classifiers": {},
    }

    for clf_name in CLASSIFIER_NAMES:
        logger.info(f"    Running {clf_name}...")
        
        # TRTR: Train Real, Test Real
        trtr = evaluate_classifier(clf_name, X_train_real, y_train, X_eval, y_eval, device)
        
        # TSTR: Train Synthetic, Test Real
        tstr = evaluate_classifier(clf_name, X_syn, y_syn, X_eval, y_eval, device)

        results["classifiers"][clf_name] = {
            "trtr": trtr,
            "tstr": tstr,
            "tstr_trtr_accuracy_ratio": round(
                tstr["accuracy"] / max(trtr["accuracy"], 1e-10), 4
            ),
            "tstr_trtr_f1_ratio": round(
                tstr["f1"] / max(trtr["f1"], 1e-10), 4
            ),
            "tstr_trtr_auc_ratio": round(
                tstr["auc_roc"] / max(trtr["auc_roc"], 1e-10), 4
            ),
        }

        logger.info(
            f"      TRTR: acc={trtr['accuracy']:.3f} f1={trtr['f1']:.3f} auc={trtr['auc_roc']:.3f} "
            f"({trtr['train_time_s']:.1f}s)"
        )
        logger.info(
            f"      TSTR: acc={tstr['accuracy']:.3f} f1={tstr['f1']:.3f} auc={tstr['auc_roc']:.3f} "
            f"({tstr['train_time_s']:.1f}s)"
        )

    return results
