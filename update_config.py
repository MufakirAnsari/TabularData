import os

with open('src/pancan/config.py', 'r') as f:
    content = f.read()

# Replace hardcoded paths with a dynamic function
content = content.replace(
'''PROJECT_ROOT = Path(os.environ.get("TABULAR_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = PROJECT_ROOT / "Data" / "TCGA-PANCAN-HiSeq-801x20531"
RESULTS_DIR = PROJECT_ROOT / "pancan_results"
SYNTHETIC_DIR = RESULTS_DIR / "synthetic"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
TSTR_DIR = RESULTS_DIR / "tstr"
TSTR_PLOTS_DIR = RESULTS_DIR / "tstr_plots"
LOGS_DIR = PROJECT_ROOT / "pancan_logs"''',
'''PROJECT_ROOT = Path(os.environ.get("TABULAR_ROOT", Path(__file__).resolve().parent.parent.parent))

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
''')

# Also fix the `synthetic_file` etc. functions so they don't capture the old global, but read it dynamically.
# Wait, python globals are read dynamically anyway! If the function executes after `set_dataset`, it reads the new SYNTHETIC_DIR.

with open('src/pancan/config.py', 'w') as f:
    f.write(content)
