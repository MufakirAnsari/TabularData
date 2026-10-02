"""
TSTR Validation Plotting — comparison charts and heatmaps.

Generates:
1. Grouped bar charts: TRTR vs TSTR per classifier per model
2. Heatmaps: TSTR/TRTR ratio across models × classifiers
3. Summary CSV for all results
"""
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


def compile_tstr_results(tstr_dir: Path) -> pd.DataFrame:
    """
    Read all TSTR JSON files and compile into a single DataFrame.
    
    Returns DataFrame with columns:
    gen_model, n_samples, classifier, trtr_accuracy, tstr_accuracy,
    trtr_f1, tstr_f1, trtr_auc, tstr_auc, ratio_accuracy, ratio_f1, ratio_auc
    """
    rows = []
    for f in sorted(tstr_dir.glob("*_tstr.json")):
        with open(f) as fh:
            data = json.load(fh)
        
        gen_model = data["gen_model"]
        n_samples = data["n_samples"]
        
        for clf_name, clf_data in data["classifiers"].items():
            trtr = clf_data["trtr"]
            tstr = clf_data["tstr"]
            rows.append({
                "gen_model": gen_model,
                "n_samples": n_samples,
                "classifier": clf_name,
                "trtr_accuracy": trtr["accuracy"],
                "tstr_accuracy": tstr["accuracy"],
                "trtr_f1": trtr["f1"],
                "tstr_f1": tstr["f1"],
                "trtr_auc": trtr["auc_roc"],
                "tstr_auc": tstr["auc_roc"],
                "trtr_precision": trtr["precision"],
                "tstr_precision": tstr["precision"],
                "trtr_recall": trtr["recall"],
                "tstr_recall": tstr["recall"],
                "trtr_train_time_s": trtr["train_time_s"],
                "tstr_train_time_s": tstr["train_time_s"],
                "ratio_accuracy": clf_data["tstr_trtr_accuracy_ratio"],
                "ratio_f1": clf_data["tstr_trtr_f1_ratio"],
                "ratio_auc": clf_data["tstr_trtr_auc_ratio"],
            })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["n_samples", "gen_model", "classifier"])
    return df


def plot_tstr_bars(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create grouped bar charts: TRTR vs TSTR accuracy per classifier,
    one subplot per sample size, grouped by generative model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classifiers = df["classifier"].unique()
    models = sorted(df["gen_model"].unique())
    sizes = sorted(df["n_samples"].unique())
    
    # ── Per sample size: bar chart of all models × classifiers ──
    for n in sizes:
        df_n = df[df["n_samples"] == n]
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Accuracy comparison
        pivot_trtr = df_n.pivot(index="classifier", columns="gen_model", values="trtr_accuracy")
        pivot_tstr = df_n.pivot(index="classifier", columns="gen_model", values="tstr_accuracy")
        
        x = np.arange(len(classifiers))
        width = 0.1
        n_models = len(models)
        
        ax = axes[0]
        for i, model in enumerate(models):
            if model in pivot_tstr.columns:
                offset = (i - n_models/2 + 0.5) * width
                trtr_vals = pivot_trtr[model].values if model in pivot_trtr.columns else np.zeros(len(classifiers))
                tstr_vals = pivot_tstr[model].values
                ax.bar(x + offset, tstr_vals, width, label=model, alpha=0.8)
        
        # Add TRTR baseline as horizontal markers
        if not pivot_trtr.empty:
            trtr_mean = pivot_trtr.mean(axis=1)
            ax.scatter(x, trtr_mean.values, marker='_', color='red', s=200, linewidths=3, 
                      zorder=5, label='TRTR baseline')
        
        ax.set_xlabel("Classifier")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"TSTR Accuracy | N={n}")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in classifiers], fontsize=8)
        ax.legend(fontsize=7, loc="lower left")
        ax.set_ylim(0, 1.1)
        
        # F1 comparison
        pivot_tstr_f1 = df_n.pivot(index="classifier", columns="gen_model", values="tstr_f1")
        pivot_trtr_f1 = df_n.pivot(index="classifier", columns="gen_model", values="trtr_f1")
        
        ax = axes[1]
        for i, model in enumerate(models):
            if model in pivot_tstr_f1.columns:
                offset = (i - n_models/2 + 0.5) * width
                tstr_vals = pivot_tstr_f1[model].values
                ax.bar(x + offset, tstr_vals, width, label=model, alpha=0.8)
        
        if not pivot_trtr_f1.empty:
            trtr_mean = pivot_trtr_f1.mean(axis=1)
            ax.scatter(x, trtr_mean.values, marker='_', color='red', s=200, linewidths=3,
                      zorder=5, label='TRTR baseline')
        
        ax.set_xlabel("Classifier")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"TSTR F1 Score | N={n}")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in classifiers], fontsize=8)
        ax.legend(fontsize=7, loc="lower left")
        ax.set_ylim(0, 1.1)
        
        plt.suptitle(f"TRTR vs TSTR Comparison | N={n}", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"tstr_bars_n{n}.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    logger.info(f"  Saved bar charts to {output_dir}")


def plot_tstr_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create heatmaps: TSTR/TRTR ratio for each metric,
    with rows = generative models, columns = classifiers.
    One heatmap per sample size.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted(df["n_samples"].unique())
    
    for n in sizes:
        df_n = df[df["n_samples"] == n]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        for idx, (metric, title) in enumerate([
            ("ratio_accuracy", "Accuracy Ratio"),
            ("ratio_f1", "F1 Ratio"),
            ("ratio_auc", "AUC-ROC Ratio"),
        ]):
            pivot = df_n.pivot(index="gen_model", columns="classifier", values=metric)
            
            ax = axes[idx]
            sns.heatmap(
                pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                center=1.0, vmin=0.5, vmax=1.5,
                ax=ax, cbar_kws={"label": "TSTR/TRTR"},
                xticklabels=[c.replace("_", "\n") for c in pivot.columns],
            )
            ax.set_title(f"{title} | N={n}")
            ax.set_xlabel("")
            ax.set_ylabel("Generative Model")
        
        plt.suptitle(f"TSTR / TRTR Ratio (1.0 = perfect) | N={n}", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"tstr_heatmap_n{n}.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    logger.info(f"  Saved heatmaps to {output_dir}")


def plot_tstr_by_sample_size(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Line plots showing how TSTR/TRTR ratio changes with sample size,
    one subplot per classifier.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classifiers = sorted(df["classifier"].unique())
    models = sorted(df["gen_model"].unique())
    n_clf = len(classifiers)
    
    ncols = 4
    nrows = (n_clf + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if n_clf > 1 else [axes]
    
    for i, clf in enumerate(classifiers):
        ax = axes[i]
        df_clf = df[df["classifier"] == clf]
        
        for model in models:
            df_m = df_clf[df_clf["gen_model"] == model].sort_values("n_samples")
            if not df_m.empty:
                ax.plot(df_m["n_samples"], df_m["ratio_accuracy"],
                       marker="o", label=model, linewidth=1.5, markersize=4)
        
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="TRTR=TSTR")
        ax.set_title(clf.replace("_", " ").title())
        ax.set_xlabel("N (Sample Size)")
        ax.set_ylabel("TSTR/TRTR Accuracy Ratio")
        ax.set_ylim(0.3, 1.3)
        ax.legend(fontsize=6)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle("TSTR/TRTR Accuracy Ratio vs Sample Size", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "tstr_by_sample_size.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"  Saved sample size plot to {output_dir}")
