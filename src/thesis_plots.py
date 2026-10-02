import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
# Use basic seaborn styling for clean, publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Use relative paths so this works on both Windows and OSC
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "thesis_plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# Clean names for thesis legends
MODEL_NAMES = {
    "smote": "SMOTE",
    "scgft": "scGFT",
    "gaussian_copula": "Gaussian Copula",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "tabddpm": "TabDDPM",
    "flow_matching": "Flow Matching"
}

def load_all_metrics():
    rows = []
    for f in METRICS_DIR.glob("*_metrics.json"):
        with open(f, "r") as fh:
            data = json.load(fh)
            # Map keys exactly as provided in the user's JSON
            rows.append({
                "Model": MODEL_NAMES.get(data.get("model"), data.get("model")),
                "N": data.get("n_samples"),
                "Generation_Time_s": data.get("time_generation_s"),
                "KS_Likeness": data.get("ks_likeness"),
                "Wasserstein_Mean": data.get("wasserstein_mean"),
                "Cosine_Similarity": data.get("mean_cosine_similarity"),
                "Meta_Correlation": data.get("meta_correlation"),
                "Syn_Sparsity": data.get("syn_sparsity"),
                "Train_Sparsity": data.get("train_sparsity"),
                "DCR_Ratio": data.get("dcr_ratio"),
            })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["N", "Model"])
    return df

def plot_line_trend(df, metric, ylabel, title, filename, log_y=False):
    plt.figure(figsize=(10, 6))
    
    for model in df["Model"].unique():
        model_df = df[df["Model"] == model].sort_values("N")
        plt.plot(model_df["N"], model_df[metric], marker='o', linewidth=2.5, markersize=8, label=model)
    
    plt.xlabel("Sample Size (N)", fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.title(title, pad=15)
    plt.xticks(df["N"].unique())
    if log_y:
        plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sparsity_bars(df):
    plt.figure(figsize=(12, 6))
    
    # We will plot N=50 and N=500 for a side-by-side comparison if available
    target_ns = [n for n in [50, 500] if n in df["N"].unique()]
    if not target_ns:
        target_ns = [df["N"].max()]
        
    plot_df = df[df["N"].isin(target_ns)]
    
    ax = sns.barplot(data=plot_df, x="Model", y="Syn_Sparsity", hue="N", palette="Blues_d")
    
    # Add horizontal line for real data sparsity (usually ~90.8%)
    mean_real_sparsity = df["Train_Sparsity"].mean()
    plt.axhline(y=mean_real_sparsity, color='red', linestyle='--', linewidth=2, label=f"Real Data ({mean_real_sparsity:.1f}%)")
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Sparsity (% Zeros)", fontweight='bold')
    plt.xlabel("")
    plt.title("Zero-Inflation Preservation by Model", pad=15)
    
    # Fix legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sparsity_preservation.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_dcr_ratio(df):
    plt.figure(figsize=(10, 6))
    
    for model in df["Model"].unique():
        model_df = df[df["Model"] == model].sort_values("N")
        plt.plot(model_df["N"], model_df["DCR_Ratio"], marker='o', linewidth=2.5, markersize=8, label=model)
    
    # Line of perfect generalization
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label="Perfect Generalization (1.0)")
    # Fill memorization zone
    plt.axhspan(0, 1.0, alpha=0.1, color='red', label="Memorization Zone (< 1.0)")
    
    plt.xlabel("Sample Size (N)", fontweight='bold')
    plt.ylabel("DCR Ratio (Eval / Train)", fontweight='bold')
    plt.title("Privacy and Memorization (DCR Ratio)", pad=15)
    plt.xticks(df["N"].unique())
    plt.ylim(0.5, 1.5)  # Focus on the relevant ratio bounds
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dcr_memorization.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_heatmap(df):
    # Get the max N available for all models, or just use N=500
    N_target = df["N"].max()
    sub_df = df[df["N"] == N_target].set_index("Model")
    
    metrics = ["KS_Likeness", "Wasserstein_Mean", "Cosine_Similarity", "Meta_Correlation"]
    
    # We need to invert Wasserstein so higher is better for the heatmap
    sub_df["Wasserstein_Inv"] = 1.0 - sub_df["Wasserstein_Mean"]
    heat_metrics = ["KS_Likeness", "Wasserstein_Inv", "Cosine_Similarity", "Meta_Correlation"]
    
    heat_data = sub_df[heat_metrics].copy()
    
    # Normalize 0 to 1 column-wise for color scale
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(heat_norm, annot=heat_data, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Normalized Score'})
    plt.title(f"Model Performance Summary (N={N_target})", pad=15)
    plt.ylabel("")
    
    # Rename columns for display
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=["KS Likeness", "1 - Wasserstein", "Cosine Sim", "Meta-Corr"], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    df = load_all_metrics()
    if df.empty:
        print("No metrics JSON files found!")
        return
        
    print(f"Loaded {len(df)} metrics files.")
    
    # Save CSV
    csv_path = RESULTS_DIR / "compiled_thesis_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved compiled metrics to {csv_path}")
    
    # 1. Generation Time
    plot_line_trend(df, "Generation_Time_s", "Generation Time (Seconds)", "Computational Cost (Log Scale)", "time_vs_n.png", log_y=True)
    
    # 2. Distribution (KS Likeness)
    plot_line_trend(df, "KS_Likeness", "KS Likeness (Higher is Better)", "1D Distribution Fidelity", "ks_likeness_vs_n.png")
    
    # 3. Global Manifold (Cosine Sim)
    plot_line_trend(df, "Cosine_Similarity", "Cosine Similarity (Higher is Better)", "Global High-Dimensional Manifold", "cosine_sim_vs_n.png")
    
    # 4. Gene Relationships (Meta-Correlation)
    plot_line_trend(df, "Meta_Correlation", "Meta-Correlation (R²)", "Gene-Gene Covariance Preservation", "meta_corr_vs_n.png")
    
    # 5. Sparsity / Drop-outs
    plot_sparsity_bars(df)
    
    # 6. Memorization
    plot_dcr_ratio(df)
    
    # 7. Summary Heatmap
    plot_heatmap(df)
    
    print(f"Successfully generated 7 thesis plots in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
