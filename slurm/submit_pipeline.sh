#!/bin/bash
# Master Pipeline Dispatcher for HDLSS Datasets

# Define the datasets and models you want to run
DATASETS=("ALLAML" "CLL_SUB_111" "colon" "Prostate_GE" "SMK_CAN_187" "GLI_85")
MODELS=("smote" "gaussian_copula" "ctgan" "tvae" "tabddpm" "flow_matching" "scgft")

export TABULAR_ROOT=$HOME/TabularData
cd $TABULAR_ROOT

# Ensure venv is activated for Python commands
source $HOME/TabularData/venv/bin/activate

for DATASET in "${DATASETS[@]}"; do
    echo "======================================================="
    echo "Processing Dataset: $DATASET"
    echo "======================================================="
    
    # 1. Dynamically extract the exact class labels for this dataset
    CLASSES=$(python -c "
import pandas as pd, numpy as np
try:
    df = pd.read_csv('Data/$DATASET/labels.csv', index_col=0)
    col = 'Class' if 'Class' in df.columns else df.columns[0]
    classes = np.unique(df[col].astype(str).values)
    print(' '.join(classes))
except Exception as e:
    print('')
")

    if [ -z "$CLASSES" ]; then
        echo "ERROR: Could not extract classes from Data/$DATASET/labels.csv. Skipping."
        continue
    fi
    
    echo "Discovered Classes: $CLASSES"
    
    # 2. Submit SLURM Jobs for each model
    for MODEL in "${MODELS[@]}"; do
        if [[ "$MODEL" == "ctgan" || "$MODEL" == "tvae" ]]; then
            # CTGAN & TVAE take longer, so we submit one parallel GPU job PER CLASS
            for CLASS in $CLASSES; do
                JOB_NAME="p_${MODEL}_${CLASS}"
                LOG_DIR="logs/$DATASET"
                mkdir -p $LOG_DIR
                
                sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --account=pwsu0516
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/slurm_${MODEL}_${CLASS}_%j.out

source \$HOME/TabularData/venv/bin/activate
export TABULAR_ROOT=\$HOME/TabularData
cd \$HOME/TabularData

python -m src.run_experiment --model $MODEL --dataset $DATASET --cancer_type="$CLASS"
EOF
                echo "  -> Submitted $MODEL (Class: $CLASS)"
            done
        else
            # Fast models (TabDDPM, Flow Matching, etc.) run in one single job for the whole dataset
            JOB_NAME="p_${MODEL}_all"
            LOG_DIR="logs/$DATASET"
            mkdir -p $LOG_DIR
            
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --account=pwsu0516
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/slurm_${MODEL}_all_%j.out

source \$HOME/TabularData/venv/bin/activate
export TABULAR_ROOT=\$HOME/TabularData
cd \$HOME/TabularData

python -m src.run_experiment --model $MODEL --dataset $DATASET
EOF
            echo "  -> Submitted $MODEL (Full Dataset)"
        fi
    done
done
