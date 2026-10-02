import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import dynamic config
import src.config as cfg

import argparse

def plot_tstr(dataset: str):
    import pandas as pd
    temp_data = cfg.PROJECT_ROOT / "Data" / dataset
    df = pd.read_csv(temp_data / "labels.csv", index_col=0)
    col = "Class" if "Class" in df.columns else df.columns[0]
    classes = list(np.unique(df[col].astype(str).values))
    cfg.set_dataset(dataset, classes)
    
    TSTR_DIR = cfg.TSTR_DIR
    OUTPUT_DIR = cfg.TSTR_PLOTS_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Rest of the plot code adapted dynamically
    models = []
    classifiers = ["logistic_regression", "random_forest", "xgboost", "gradient_boosting", "naive_bayes", "svm", "knn"]
    trtr_accs = {clf: [] for clf in classifiers}
    tstr_accs = {clf: [] for clf in classifiers}
    
    for f in TSTR_DIR.glob("*_hdlss_tstr.json"):
        model_name = f.stem.replace("_hdlss_tstr", "")
        models.append(model_name)
        with open(f, 'r') as file:
            data = json.load(file)
            for clf in classifiers:
                if clf in data["classifiers"]:
                    trtr_accs[clf].append(data["classifiers"][clf]["trtr"]["accuracy"])
                    tstr_accs[clf].append(data["classifiers"][clf]["tstr"]["accuracy"])
                else:
                    trtr_accs[clf].append(0)
                    tstr_accs[clf].append(0)
                    
    if not models:
        print(f"No TSTR data found in {TSTR_DIR}")
        return

    # Plot 1: Bar chart grouped by classifier
    x = np.arange(len(classifiers))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, model in enumerate(models):
        offset = (i - len(models)/2) * width + width/2
        accs = [tstr_accs[clf][i] for clf in classifiers]
        ax.bar(x + offset, accs, width, label=model)
        
    ax.set_ylabel('TSTR Accuracy')
    ax.set_title(f'TSTR Accuracy by Classifier ({dataset})')
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tstr_by_classifier.png", dpi=300)
    plt.close()
    
    # Plot 2: Average across all classifiers
    avg_trtr = np.mean([np.mean(trtr_accs[clf]) for clf in classifiers]) # TRTR is same for all models usually
    avg_tstr = [np.mean([tstr_accs[clf][i] for clf in classifiers]) for i in range(len(models))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(models, avg_tstr, color='skyblue', edgecolor='black')
    ax.axhline(y=avg_trtr, color='r', linestyle='--', label=f'Avg TRTR (Real Data Baseline): {avg_trtr:.3f}')
    ax.set_ylabel('Average TSTR Accuracy')
    ax.set_title(f'Average TSTR Classification Accuracy ({dataset})')
    for i, v in enumerate(avg_tstr):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tstr_avg_accuracy.png", dpi=300)
    plt.close()
    
    print(f"Saved TSTR plots to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="colon")
    args = parser.parse_args()
    plot_tstr(args.dataset)
