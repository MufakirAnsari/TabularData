#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# HDLSS Synthetic scRNA-seq Generation — SLURM Job Template
# Usage: sbatch --export=MODEL=smote,N=50 slurm/run_job.sbatch
# ──────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Model: ${MODEL}"
echo "N: ${N}"
echo "============================================================"

# Navigate to project
cd $HOME/TabularData

# Activate virtual environment (already has all packages)
source venv/bin/activate

# Set project root for config.py
export TABULAR_ROOT=$HOME/TabularData

# Ensure output directories exist
mkdir -p logs results/synthetic results/metrics results/plots

# Print environment info
echo "Python: $(python --version)"
nvidia-smi 2>/dev/null || echo "No GPU on this node (CPU job)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch check skipped"

# Run experiment
echo ""
echo "Starting experiment: model=${MODEL}, N=${N}"
echo ""
python -m src.run_experiment --model ${MODEL} --n ${N}
EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Job finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
