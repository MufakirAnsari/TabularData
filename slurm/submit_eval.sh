#!/bin/bash
# Master Pipeline Dispatcher for Evaluation
# Run this once generation jobs finish to evaluate safely without SSH timeouts

DATASETS=("ALLAML" "CLL_SUB_111" "colon" "Prostate_GE" "SMK_CAN_187" "GLI_85")
MODELS=("smote" "gaussian_copula" "ctgan" "tvae" "tabddpm" "flow_matching" "scgft")

export TABULAR_ROOT=$HOME/TabularData
cd $TABULAR_ROOT

for DATASET in "${DATASETS[@]}"; do
    echo "======================================================="
    echo "Evaluating Dataset: $DATASET"
    echo "======================================================="
    
    for MODEL in "${MODELS[@]}"; do
        JOB_NAME="eval_${MODEL}_${DATASET}"
        LOG_DIR="logs/$DATASET"
        mkdir -p $LOG_DIR
        
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --account=pwsu0516
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=$LOG_DIR/slurm_eval_${MODEL}_%j.out

source \$HOME/TabularData/venv/bin/activate
export TABULAR_ROOT=\$HOME/TabularData
cd \$HOME/TabularData

# The merge_eval flag automatically gathers class chunks (for CTGAN/TVAE) or loads the full synthetic data (for fast models) and evaluates it.
python -m src.run_experiment --model $MODEL --dataset $DATASET --merge_eval
EOF
        echo "  -> Submitted Eval: $MODEL"
    done
done
