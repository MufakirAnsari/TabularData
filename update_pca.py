import os

with open('src/pancan/pca_evaluation.py', 'r') as f:
    content = f.read()

# Replace hardcoded imports with dynamic cfg
content = content.replace(
'''from src.pancan.config import DATA_DIR, RESULTS_DIR
MODELS = ["smote", "gaussian_copula", "ctgan", "tvae", "tabddpm", "flow_matching", "scgft"]
from src.pancan.config import synthetic_file, synthetic_labels_file, CANCER_TYPES, TRAIN_RATIO, SEED''',
'''import argparse
import src.pancan.config as cfg
MODELS = ["smote", "gaussian_copula", "ctgan", "tvae", "tabddpm", "flow_matching", "scgft"]
from src.pancan.config import synthetic_file, synthetic_labels_file, TRAIN_RATIO, SEED'''
)

content = content.replace('OUTPUT_DIR = RESULTS_DIR / "pca_plots"', 'OUTPUT_DIR = None')

content = content.replace('DATA_DIR', 'cfg.DATA_DIR')
content = content.replace('RESULTS_DIR', 'cfg.RESULTS_DIR')
content = content.replace('CANCER_TYPES', 'cfg.CANCER_TYPES')

# Fix OUTPUT_DIR
content = content.replace('OUTPUT_DIR.mkdir', 'cfg.RESULTS_DIR.joinpath("pca_plots").mkdir')
content = content.replace('OUTPUT_DIR', 'cfg.RESULTS_DIR.joinpath("pca_plots")')

# Wrap the main block
content = content.replace(
'''if __name__ == "__main__":
    print("==================================================")
    print(" PANCAN PCA EVALUATION: TRUE BASIS PROJECTION")
    print("==================================================")
''',
'''if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="colon")
    args = parser.parse_args()
    
    # Dynamic config initialization
    import pandas as pd
    temp_data_dir = cfg.PROJECT_ROOT / "Data" / args.dataset
    df_l = pd.read_csv(temp_data_dir / "labels.csv", index_col=0)
    col = "Class" if "Class" in df_l.columns else df_l.columns[0]
    discovered_classes = list(np.unique(df_l[col].astype(str).values))
    cfg.set_dataset(args.dataset, discovered_classes)

    print("==================================================")
    print(f" PCA EVALUATION: {args.dataset}")
    print("==================================================")
'''
)

with open('src/pancan/pca_evaluation.py', 'w') as f:
    f.write(content)
