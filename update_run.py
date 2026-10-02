import os

with open('src/pancan/run_experiment.py', 'r') as f:
    content = f.read()

# Replace the top imports
content = content.replace(
'''from src.pancan.config import (
    DATA_DIR, RESULTS_DIR, SYNTHETIC_DIR, METRICS_DIR, PLOTS_DIR,
    TSTR_DIR, TSTR_PLOTS_DIR, LOGS_DIR,
    CANCER_TYPES, TRAIN_RATIO, SYNTH_RATIO, SEED, MODEL_NAMES,
    synthetic_file, synthetic_labels_file, metrics_file,
    tstr_file, plot_dir
)''',
'''import src.pancan.config as cfg
from src.pancan.config import (
    TRAIN_RATIO, SYNTH_RATIO, SEED, MODEL_NAMES,
    synthetic_file, synthetic_labels_file, metrics_file,
    tstr_file, plot_dir
)''')

# Replace globals with cfg.globals
for var in ["DATA_DIR", "RESULTS_DIR", "SYNTHETIC_DIR", "METRICS_DIR", "PLOTS_DIR", "TSTR_DIR", "TSTR_PLOTS_DIR", "LOGS_DIR", "CANCER_TYPES"]:
    content = content.replace(f" {var}", f" cfg.{var}")
    content = content.replace(f"({var}", f"(cfg.{var}")
    content = content.replace(f"[{var}", f"[cfg.{var}")
    content = content.replace(f"{var}.", f"cfg.{var}.")

# Update argparse to include --dataset
content = content.replace(
'''parser.add_argument("--cancer_type", type=str, choices=cfg.CANCER_TYPES + ["all"], default="all")''',
'''parser.add_argument("--dataset", type=str, default="TCGA-PANCAN-HiSeq-801x20531")
    parser.add_argument("--cancer_type", type=str, default="all")'''
)

# Call cfg.set_dataset(args.dataset, classes) right after parsing args
setup_code = '''
    args = parser.parse_args()

    # Pre-load data once to get classes
    temp_data_dir = cfg.PROJECT_ROOT / "Data" / args.dataset
    if not temp_data_dir.exists():
        print(f"Error: Dataset {args.dataset} not found in {temp_data_dir}")
        sys.exit(1)
        
    import pandas as pd
    import numpy as np
    labels_path = temp_data_dir / "labels.csv"
    df_l = pd.read_csv(labels_path, index_col=0)
    if "Class" in df_l.columns:
        discovered_classes = list(np.unique(df_l["Class"].values))
    else:
        discovered_classes = list(np.unique(df_l.iloc[:, 0].values))
        
    cfg.set_dataset(args.dataset, discovered_classes)
'''

content = content.replace("args = parser.parse_args()", setup_code)

with open('src/pancan/run_experiment.py', 'w') as f:
    f.write(content)
