"""
Central configuration for the PANCAN TCGA RNA-Seq synthetic data pipeline.
Dataset: TCGA-PANCAN-HiSeq-801x20531 (UCI ML Repository #401)
  - 801 bulk RNA-Seq samples, 20,531 genes, 5 cancer types
  - BRCA (300), KIRC (146), LUAD (141), PRAD (136), COAD (78)
"""
import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get("TABULAR_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = PROJECT_ROOT / "Data" / "TCGA-PANCAN-HiSeq-801x20531"
RESULTS_DIR = PROJECT_ROOT / "results" / "TCGA-PANCAN-HiSeq-801x20531"
SYNTHETIC_DIR = RESULTS_DIR / "synthetic"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
TSTR_DIR = RESULTS_DIR / "tstr"
TSTR_PLOTS_DIR = RESULTS_DIR / "tstr_plots"
LOGS_DIR = PROJECT_ROOT / "logs" / "TCGA-PANCAN-HiSeq-801x20531"

def set_dataset(dataset_name: str, classes: list = None):
    global DATA_DIR, RESULTS_DIR, SYNTHETIC_DIR, METRICS_DIR, PLOTS_DIR, TSTR_DIR, TSTR_PLOTS_DIR, LOGS_DIR, CANCER_TYPES
    DATA_DIR = PROJECT_ROOT / "Data" / dataset_name
    RESULTS_DIR = PROJECT_ROOT / "results" / dataset_name
    SYNTHETIC_DIR = RESULTS_DIR / "synthetic"
    METRICS_DIR = RESULTS_DIR / "metrics"
    PLOTS_DIR = RESULTS_DIR / "plots"
    TSTR_DIR = RESULTS_DIR / "tstr"
    TSTR_PLOTS_DIR = RESULTS_DIR / "tstr_plots"
    LOGS_DIR = PROJECT_ROOT / "logs" / dataset_name
    if classes is not None:
        CANCER_TYPES = classes


# ──────────────────────────────────────────────────────────────────────
# DATA PROPERTIES
# ──────────────────────────────────────────────────────────────────────
N_GENES = 20531
N_SAMPLES = 801
DTYPE = "float32"
CANCER_TYPES = ["BRCA", "KIRC", "LUAD", "PRAD", "COAD"]

# ──────────────────────────────────────────────────────────────────────
# EXPERIMENT DESIGN
# ──────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8          # 80% train, 20% eval (stratified)
SYNTH_RATIO = 1            # Generate same N as training (1:1)
SEED = 42

# File naming conventions
def synthetic_file(model: str) -> Path:
    return SYNTHETIC_DIR / f"{model}_hdlss_syn.npy"

def synthetic_labels_file(model: str) -> Path:
    return SYNTHETIC_DIR / f"{model}_hdlss_syn_labels.npy"

def metrics_file(model: str) -> Path:
    return METRICS_DIR / f"{model}_hdlss_metrics.json"

def tstr_file(model: str) -> Path:
    return TSTR_DIR / f"{model}_hdlss_tstr.json"

def plot_dir(model: str) -> Path:
    return PLOTS_DIR / model

# ──────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ──────────────────────────────────────────────────────────────────────
MODEL_NAMES = [
    "smote",
    "gaussian_copula",
    "ctgan",
    "tvae",
    "tabddpm",
    "flow_matching",
    "scgft",
]

# ──────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS (per model)
# ──────────────────────────────────────────────────────────────────────
HPARAMS = {
    "smote": {
        "k_neighbors": 5,
    },
    "gaussian_copula": {
        # Ledoit-Wolf shrinkage is automatic
    },
    "ctgan": {
        "generator_dim": (256, 256),
        "discriminator_dim": (256, 256),
        "epochs": 300,
        "batch_size": 500,
        "generator_lr": 2e-4,
        "discriminator_lr": 2e-4,
        "pac": 10,
    },
    "tvae": {
        "compress_dims": (256, 256),
        "decompress_dims": (256, 256),
        "embedding_dim": 256,
        "epochs": 300,
        "batch_size": 500,
    },
    "tabddpm": {
        "hidden_dim": 1024,
        "n_blocks": 4,
        "time_emb_dim": 256,
        "dropout": 0.3,
        "T": 1000,
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "max_epochs": 2000,
        "patience": 200,
        "grad_clip": 1.0,
    },
    "flow_matching": {
        "hidden_dim": 1024,
        "n_blocks": 4,
        "time_emb_dim": 256,
        "dropout": 0.3,
        "sigma_min": 0.001,
        "ode_steps": 100,
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "max_epochs": 2000,
        "patience": 200,
        "grad_clip": 1.0,
    },
    "scgft": {
        "ncpmnts": 100,
        "phase_range": 0.7854,       # pi/4 radians
        "amp_scale_range": (0.9, 1.1),
    },
}
