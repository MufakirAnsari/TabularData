#!/bin/bash
###############################################################################
# PANCAN TCGA — Submit All 7 Generative Models
#
# Submits one SLURM job per model. Each job runs class-conditional generation
# on the TCGA-PANCAN-HiSeq dataset (801 samples × 20,531 genes, 5 cancer types)
# and evaluates both statistical fidelity and TSTR classification.
#
# Usage:
#   cd GenAI/
#   bash slurm/submit_pancan_all.sh
###############################################################################

set -e

# ── Configuration ──
ACCOUNT="pwsu0516"
PARTITION="gpu"
GPUS=1
CPUS=8
MEM="64G"
VENV_PATH="$HOME/TabularData/venv/bin/activate"

# Project root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Output directories
mkdir -p "$PROJECT_ROOT/pancan_logs"
mkdir -p "$PROJECT_ROOT/pancan_results"

echo "============================================"
echo "PANCAN TCGA — Submitting all 7 models"
echo "Project root: $PROJECT_ROOT"
echo "============================================"

# ── Model configurations ──
# Format: model_name  wall_time  gpu_needed
declare -a MODELS=(
    "smote          01:00:00  no"
    "gaussian_copula 04:00:00 no"
    "ctgan           24:00:00 yes"
    "tvae            24:00:00 yes"
    "tabddpm         24:00:00 yes"
    "flow_matching   24:00:00 yes"
    "scgft           01:00:00 no"
)

for entry in "${MODELS[@]}"; do
    read -r MODEL WALLTIME NEEDS_GPU <<< "$entry"

    if [ "$NEEDS_GPU" = "yes" ]; then
        PART="$PARTITION"
        GRES="#SBATCH --gres=gpu:$GPUS"
    else
        PART="batch"
        GRES=""
    fi

    JOB_NAME="pancan_${MODEL}"
    LOG_FILE="$PROJECT_ROOT/pancan_logs/slurm_${MODEL}_%j.out"

    echo "Submitting: $MODEL (walltime=$WALLTIME, partition=$PART)"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PART
#SBATCH --time=$WALLTIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
$GRES
#SBATCH --output=$LOG_FILE
#SBATCH --export=ALL

echo "=========================================="
echo "PANCAN Experiment: $MODEL"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node:   \$(hostname)"
echo "Start:  \$(date)"
echo "=========================================="

# Load modules
module load python/3.9 2>/dev/null || true
module load cuda/11.8 2>/dev/null || true

# Activate virtual environment
source "$VENV_PATH" 2>/dev/null || {
    echo "Warning: Could not activate venv at $VENV_PATH"
    echo "Attempting to use system Python..."
}

# Set project root
export TABULAR_ROOT="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Run experiment
echo "Running: python -m src.pancan.run_experiment --model $MODEL"
python -m src.pancan.run_experiment --model $MODEL

echo "=========================================="
echo "End:    \$(date)"
echo "Status: \$?"
echo "=========================================="
EOF

    echo "  -> Submitted $JOB_NAME"
done

echo ""
echo "============================================"
echo "All 7 PANCAN jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Results dir:  $PROJECT_ROOT/pancan_results/"
echo "Logs dir:     $PROJECT_ROOT/pancan_logs/"
echo "============================================"
