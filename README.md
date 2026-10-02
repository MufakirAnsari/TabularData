# Generative AI for HDLSS Tabular Data

A comprehensive benchmarking pipeline for evaluating Generative AI models on **High-Dimensional, Low-Sample-Size (HDLSS)** tabular data, primarily focusing on bioinformatics and genomics datasets (e.g., transcriptomics arrays where $P \gg N$).

## Overview

Generating synthetic data in the HDLSS regime (e.g., 20,000 genes for 100 patients) breaks many core assumptions of standard generative models. This repository implements a full end-to-end pipeline to train, generate, and evaluate **7 diverse generative models** across multiple HDLSS datasets to understand their stability, fidelity, and privacy/novelty preservation.

### Supported Models
1. **SMOTE** (Sparse-aware variant) - Nearest-neighbor interpolation baseline
2. **Gaussian Copula** (SDV) - Statistical generative modeling with Ledoit-Wolf shrinkage
3. **CTGAN** - Conditional Tabular GAN 
4. **TVAE** - Tabular Variational Autoencoder
5. **TabDDPM** - Tabular Denoising Diffusion Probabilistic Model
6. **Flow Matching (OT-CFM)** - Optimal Transport Conditional Flow Matching via Neural ODEs
7. **scGFT** - Single-cell Generative Fourier Transformer

### Supported Datasets
* **ALLAML**: Leukemia ($N=72, P=7,129$, 2-class)
* **CLL_SUB_111**: Chronic Lymphocytic Leukemia ($N=111, P=11,340$, 3-class)
* **colon**: Colon Cancer ($N=62, P=2,000$, 2-class)
* **GLI_85**: Glioma ($N=85, P=22,283$, 2-class)
* **Prostate_GE**: Prostate Cancer ($N=102, P=5,966$, 2-class)
* **SMK_CAN_187**: Smokers vs Non-Smokers ($N=187, P=19,993$, 2-class)

### Evaluation Metrics
* **Fidelity**: Wasserstein Distance (Marginal), Kolmogorov-Smirnov (KS) Likeness, Mean L2 Distance.
* **Correlation**: Meta-Correlation (Pearson correlation of feature-feature correlation matrices).
* **Privacy/Novelty**: **DCR** (Distance to Closest Record).
* **Utility**: **TSTR** (Train on Synthetic, Test on Real) & TRTR (Train on Real, Test on Real) across 7 downstream classifiers: Logistic Regression, Random Forest, XGBoost, Gradient Boosting, Naive Bayes, SVM, and KNN.

---

## Directory Structure

```text
├── src/                      # Source code
│   ├── models/               # Generative model implementations (CTGAN, TVAE, etc.)
│   ├── config.py             # Global hyperparameters and dataset configurations
│   ├── data_utils.py         # Data loading, splitting, and scaling
│   ├── evaluate.py           # Evaluation framework (Fidelity, DCR, TSTR)
│   ├── run_experiment.py     # Main entry point for generation pipeline
│   ├── run_tstr.py           # Main entry point for standalone TSTR evaluation
│   ├── tstr_validation.py    # TSTR logic and cross-validation
│   └── *_plots.py            # Plotting scripts (UMAP, PCA, TSTR charts)
├── slurm/                    # SLURM batch scripts for HPC deployment
│   ├── submit_pipeline.sh    # Submits array jobs for all datasets & models
│   ├── submit_tstr.sh        # Submits TSTR evaluation jobs
│   └── run_job.sbatch        # SBATCH configuration for generation
├── Data/                     # (Ignored in Git) Raw dataset storage (.mat / .csv)
└── results/                  # (Ignored in Git) Generated artifacts
    └── {dataset_name}/
        ├── metrics/          # .json files containing HDLSS fidelity and DCR metrics
        ├── synthetic/        # .npy generated datasets
        ├── plots/            # UMAP distributions
        ├── pca_plots/        # 2D and 3D PCA distributions
        └── tstr/             # .json classification reports
```

---

## Installation & Setup

### 1. Local Environment
It is recommended to use a Python virtual environment.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. HPC Environment (OSC / SLURM)
If deploying on the Ohio Supercomputer Center (OSC) or a similar HPC cluster:

```bash
# Execute the setup script to load modules and install dependencies
bash setup_env.sh
```

Ensure your data files (e.g., `ALLAML.mat`, `colon.mat`) are placed inside the `Data/` directory.

---

## How to Run

The pipeline is split into two phases: **Generation + Fidelity** and **TSTR Evaluation**.

### Option A: Running Locally

**1. Generate Data & Compute Fidelity:**
Generate synthetic data for a specific dataset and model. This will automatically compute Wasserstein distance, KS Likeness, and DCR.
```bash
python src/run_experiment.py --dataset ALLAML --model tvae
```
*Outputs will be saved to `results/ALLAML/synthetic/` and `results/ALLAML/metrics/`.*

**2. Run TSTR Evaluation:**
After generating data, run the downstream classification benchmarks.
```bash
python src/run_tstr.py --dataset ALLAML --model tvae
```
*Outputs will be saved to `results/ALLAML/tstr/`.*

### Option B: Running on SLURM (HPC)

The repository provides automated bash scripts to queue jobs for all datasets and models concurrently on a SLURM cluster.

**1. Submit Generation Jobs:**
```bash
# Submits jobs for 6 datasets x 7 models (42 jobs)
bash slurm/submit_pipeline.sh
```
*You can monitor progress using `squeue -u $USER`.*

**2. Submit TSTR Evaluation Jobs:**
Once all generation jobs complete, submit the evaluation suite:
```bash
bash slurm/submit_tstr.sh
```

**3. Generate TSTR Plots:**
```bash
bash slurm/generate_plots.sh
```

---

## Notes on the HDLSS Regime

* **TabDDPM Failure:** Because standard diffusion score-matching collapses in $P \gg N$ spaces without manifold projection, expect `TabDDPM` to diverge severely on $P > 5000$ (e.g., GLI_85).
* **CTGAN VGM Expansion:** CTGAN uses Variational Gaussian Mixture (VGM) preprocessing, which expands dimensionality $10\times$. For HDLSS genomics, VGM is aggressively bounded or bypassed to prevent Out-Of-Memory (OOM) errors.
* **scGFT and SMOTE:** Because these methods do not rely on iterative neural network gradient updates, they are highly robust in HDLSS spaces but suffer from high memorization (low DCR ratio).

## License
MIT License
