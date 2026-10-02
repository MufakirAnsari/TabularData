#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# TSTR Validation — SLURM Job Template
# Usage: sbatch --export=MODEL=smote,N=50 slurm/run_tstr.sbatch
# ──────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "TSTR Validation Job started: $(date)"
echo "Node: $(hostname)"
echo "Model: ${MODEL}"
echo "N: ${N}"
echo "============================================================"

cd $HOME/TabularData
source venv/bin/activate
export TABULAR_ROOT=$HOME/TabularData
mkdir -p logs results/tstr results/tstr_plots

echo "Python: $(python --version)"
nvidia-smi 2>/dev/null || echo "No GPU (CPU job)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')" 2>/dev/null || true

echo ""
python -m src.run_tstr --model ${MODEL} --n ${N}
EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Job finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
