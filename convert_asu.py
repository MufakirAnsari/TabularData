import os
import scipy.io
import pandas as pd
from pathlib import Path

datasets = ["ALLAML", "CLL_SUB_111", "colon", "Prostate_GE", "SMK_CAN_187", "GLI_85"]
base_dir = Path("Data")
asu_dir = base_dir / "ASU"

for ds in datasets:
    mat_path = asu_dir / f"{ds}.mat"
    if not mat_path.exists():
        print(f"Skipping {ds}, not found.")
        continue
    
    mat = scipy.io.loadmat(mat_path)
    X = mat['X']
    Y = mat['Y']
    
    out_dir = base_dir / ds
    out_dir.mkdir(exist_ok=True)
    
    # Create DataFrames
    df_data = pd.DataFrame(X, columns=[f"gene_{i}" for i in range(X.shape[1])])
    df_data.index = [f"sample_{i}" for i in range(X.shape[0])]
    
    df_labels = pd.DataFrame(Y, columns=["Class"])
    df_labels.index = df_data.index
    
    df_data.to_csv(out_dir / "data.csv")
    df_labels.to_csv(out_dir / "labels.csv")
    
    print(f"Converted {ds}: X {X.shape}, Y {Y.shape}")
