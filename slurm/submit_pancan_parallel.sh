#!/bin/bash
###############################################################################
# PANCAN TCGA — Submit Parallel GPU Models
#
# Submits 5 separate SLURM jobs (one per cancer class) for CTGAN and TVAE.
# This runs them in parallel to drastically cut down training time (20,531 genes).
#
# Usage:
#   cd GenAI/
#   bash slurm/submit_pancan_parallel.sh
###############################################################################

set -e

# ── Configuration ──
ACCOUNT="pwsu0516"
PARTITION="gpu"
GPUS=1
CPUS=8
MEM="64G"
WALLTIME="24:00:00"
VENV_PATH="$HOME/TabularData/venv/bin/activate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$PROJECT_ROOT/pancan_logs"
mkdir -p "$PROJECT_ROOT/pancan_results"

echo "============================================"
echo "PANCAN TCGA — Submitting CTGAN & TVAE (Parallel)"
echo "Project root: $PROJECT_ROOT"
echo "============================================"

# Only run models that need parallelization (ctgan and tvae)
MODELS=("ctgan" "tvae")
CLASSES=("BRCA" "KIRC" "LUAD" "PRAD" "COAD")

for MODEL in "${MODELS[@]}"; do
    for CLASS in "${CLASSES[@]}"; do
        
        JOB_NAME="p_${MODEL}_${CLASS}"
        LOG_FILE="$PROJECT_ROOT/pancan_logs/slurm_${MODEL}_${CLASS}_%j.out"

        echo "Submitting: $MODEL for class $CLASS (partition=$PARTITION)"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --time=$WALLTIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --gres=gpu:$GPUS
#SBATCH --output=$LOG_FILE
#SBATCH --export=ALL

echo "=========================================="
echo "PANCAN Experiment: \$JOB_NAME"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node:   \$(hostname)"
echo "Start:  \$(date)"
echo "=========================================="

module load python/3.9 2>/dev/null || true
module load cuda/11.8 2>/dev/null || true

source "$VENV_PATH" 2>/dev/null || true

export TABULAR_ROOT="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Run parallel generation for just this class
echo "Running: python -m src.pancan.run_experiment --model $MODEL --cancer_type $CLASS"
python -m src.pancan.run_experiment --model $MODEL --cancer_type $CLASS

echo "=========================================="
echo "End:    \$(date)"
echo "Status: \$?"
echo "=========================================="
EOF
        echo "  -> Submitted $JOB_NAME"
    done
done

echo ""
echo "============================================"
echo "Submitted 10 parallel GPU jobs."
echo "When all 10 are finished, run the merge command to evaluate:"
echo "  python -m src.pancan.run_experiment --model ctgan --merge_eval"
echo "  python -m src.pancan.run_experiment --model tvae --merge_eval"
echo "============================================"
