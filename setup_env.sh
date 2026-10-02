#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# One-command environment setup for OSC Pitzer
#
# Usage: bash setup_env.sh
#
# This script:
# 1. Loads required modules
# 2. Installs missing Python packages into existing venv
# 3. Creates output directories
# 4. Verifies everything works
# ──────────────────────────────────────────────────────────────────────

set -e  # Exit on any error

echo "============================================================"
echo "Setting up HDLSS Synthetic scRNA-seq Pipeline"
echo "============================================================"

cd $HOME/TabularData

# Load modules
echo "Loading modules..."
module load python/3.9-2022.05
module load cuda/11.8.0

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install numpy scipy pandas scikit-learn scanpy anndata matplotlib seaborn umap-learn
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install sdv torchdiffeq

# Create directory structure
echo "Creating directories..."
mkdir -p src/models
mkdir -p slurm
mkdir -p logs
mkdir -p results/synthetic
mkdir -p results/metrics
mkdir -p results/plots
mkdir -p Data/hdlss_subsets

# Set project root
export TABULAR_ROOT=$HOME/TabularData

# Verify installation
echo ""
echo "============================================================"
echo "Verification"
echo "============================================================"
python -c "
import numpy as np; print(f'numpy:       {np.__version__}')
import scipy; print(f'scipy:       {scipy.__version__}')
import pandas as pd; print(f'pandas:      {pd.__version__}')
import sklearn; print(f'sklearn:     {sklearn.__version__}')
import scanpy as sc; print(f'scanpy:      {sc.__version__}')
import anndata; print(f'anndata:     {anndata.__version__}')
import torch; print(f'torch:       {torch.__version__}')
print(f'CUDA avail:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:         {torch.cuda.get_device_name(0)}')
import sdv; print(f'sdv:         {sdv.__version__}')
import torchdiffeq; print(f'torchdiffeq: OK')
import umap; print(f'umap-learn:  {umap.__version__}')
print()
print('All packages installed successfully!')
"

# Verify data files
echo ""
echo "Verifying data files..."
python -c "
import scanpy as sc
import os
data_dir = os.path.join(os.environ.get('TABULAR_ROOT', '.'), 'Data', 'hdlss_subsets')
sizes = [50, 100, 200, 500, 1000]
for n in sizes:
    train_f = os.path.join(data_dir, f'neurons_n{n}_p22787.h5ad')
    eval_f = os.path.join(data_dir, f'neurons_n{n}_p22787_eval.h5ad')
    if os.path.exists(train_f):
        a = sc.read_h5ad(train_f)
        print(f'  ✓ Train N={n}: {a.shape}')
    else:
        print(f'  ✗ MISSING: {train_f}')
    if os.path.exists(eval_f):
        a = sc.read_h5ad(eval_f)
        print(f'  ✓ Eval  N={n}: {a.shape}')
    else:
        print(f'  ✗ MISSING: {eval_f}')
"

echo ""
echo "============================================================"
echo "Setup complete! You can now run experiments:"
echo "  python -m src.run_experiment --model smote --n 50"
echo "  bash slurm/submit_all.sh"
echo "============================================================"
