import sys

with open('src/pancan/pca_evaluation.py', 'r') as f:
    content = f.read()

# Replace static OUTPUT_DIR references inside functions to dynamic ones
content = content.replace(
'''    metrics_path = OUTPUT_DIR / "pca_pancan_metrics.json"''',
'''    out_dir = cfg.RESULTS_DIR / "pca_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "pca_pancan_metrics.json"'''
)

content = content.replace(
'''                OUTPUT_DIR / f"pca_2d_{model}.png")''',
'''                (cfg.RESULTS_DIR / "pca_plots") / f"pca_2d_{model}.png")'''
)

content = content.replace(
'''                OUTPUT_DIR / f"pca_3d_{model}.png")''',
'''                (cfg.RESULTS_DIR / "pca_plots") / f"pca_3d_{model}.png")'''
)

content = content.replace(
'''    print(f"Plots saved to: {OUTPUT_DIR}")''',
'''    print(f"Plots saved to: {cfg.RESULTS_DIR / 'pca_plots'}")'''
)

with open('src/pancan/pca_evaluation.py', 'w') as f:
    f.write(content)
