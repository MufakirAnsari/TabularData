"""
Master experiment runner for the HDLSS TCGA synthetic data pipeline.

Class-conditional generation strategy:
  For each generative model:
    1. For each cancer class: train model on class-specific training data
    2. Generate synthetic samples per class (matching class proportions)
    3. Combine into full synthetic dataset with labels
    4. Evaluate: statistical fidelity + TSTR 5-class classification

Usage:
    # Run all classes sequentially (best for fast models like SMOTE)
    python -m src.run_experiment --model ctgan

    # Parallel Mode (Step 1): Generate a specific class
    python -m src.run_experiment --model ctgan --cancer_type BRCA

    # Parallel Mode (Step 2): Merge classes and evaluate
    python -m src.run_experiment --model ctgan --merge_eval
"""
import argparse
import json
import logging
import sys
import time
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Setup project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as cfg
from src.config import (
    #
    #
    TRAIN_RATIO, SYNTH_RATIO, SEED, MODEL_NAMES,
    synthetic_file, synthetic_labels_file, metrics_file, tstr_file, plot_dir,
)
from src.data_utils import (
    load_dataset, stratified_split, get_class_data,
    save_synthetic, load_synthetic,
)
from src.evaluate import evaluate_fidelity, evaluate_tstr


def setup_logging(model: str, cancer_type: str = "all", merge: bool = False) -> None:
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    mode_str = "merge" if merge else cancer_type
    log_file = cfg.LOGS_DIR / f"{model}_{mode_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(ch)

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(fh)

    logging.info(f"Logging to: {log_file}")


def get_generator(model_name: str):
    if model_name == "smote":
        from src.models.smote_gen import generate
    elif model_name == "gaussian_copula":
        from src.models.gaussian_copula_gen import generate
    elif model_name == "ctgan":
        from src.models.ctgan_gen import generate
    elif model_name == "tvae":
        from src.models.tvae_gen import generate
    elif model_name == "tabddpm":
        from src.models.tabddpm_gen import generate
    elif model_name == "flow_matching":
        from src.models.flow_matching_gen import generate
    elif model_name == "scgft":
        from src.models.scgft_gen import generate
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return generate


def is_completed(model: str) -> bool:
    syn_path = synthetic_file(model)
    met_path = metrics_file(model)
    tstr_path = tstr_file(model)
    if syn_path.exists() and met_path.exists() and tstr_path.exists():
        try:
            with open(met_path) as f:
                data = json.load(f)
            if "model" in data and "wasserstein_mean" in data:
                return True
        except (json.JSONDecodeError, KeyError):
            return False
    return False


def run_single(model_name: str, cancer_type: str = "all", merge_eval: bool = False) -> dict:
    logger = logging.getLogger(__name__)

    # ── Check for resume ──
    if is_completed(model_name) and not merge_eval:
        logger.info(f"SKIP: {model_name} already completed. Delete files to re-run.")
        if metrics_file(model_name).exists():
            with open(metrics_file(model_name)) as f:
                return json.load(f)
        return {}

    logger.info("=" * 70)
    logger.info(f"HDLSS EXPERIMENT: model={model_name} | class={cancer_type} | merge={merge_eval}")
    logger.info("=" * 70)

    for d in [cfg.SYNTHETIC_DIR, cfg.METRICS_DIR, cfg.PLOTS_DIR, cfg.TSTR_DIR, cfg.TSTR_PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Detect device ──
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            try:
                logger.info(f"  GPU detected: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass
    except ImportError:
        pass

    # =========================================================================
    # PATH A: MERGE & EVALUATE
    # =========================================================================
    if merge_eval:
        logger.info("MODE: Merge & Evaluate. Loading synthetic data...")
        X, y, _, _ = load_dataset(cfg.DATA_DIR)
        X_train, X_eval, y_train, y_eval = stratified_split(X, y, TRAIN_RATIO, SEED)
        
        # Try loading per-class chunks first (CTGAN/TVAE parallel mode)
        all_chunks_exist = True
        for ct in cfg.CANCER_TYPES:
            f_data = synthetic_file(model_name).with_name(f"{model_name}_hdlss_syn_{ct}.npy")
            if not f_data.exists():
                all_chunks_exist = False
                break
        
        if all_chunks_exist:
            logger.info("  Found per-class chunk files. Merging...")
            syn_parts = []
            syn_labels = []
            for ct in cfg.CANCER_TYPES:
                f_data = synthetic_file(model_name).with_name(f"{model_name}_hdlss_syn_{ct}.npy")
                f_lbl = synthetic_labels_file(model_name).with_name(f"{model_name}_hdlss_syn_labels_{ct}.npy")
                syn_parts.append(np.load(f_data))
                syn_labels.append(np.load(f_lbl, allow_pickle=True))
                
            X_syn = np.vstack(syn_parts).astype(np.float32)
            y_syn = np.concatenate(syn_labels).astype(str)
        elif synthetic_file(model_name).exists():
            logger.info("  Found whole-dataset synthetic file. Loading...")
            X_syn, y_syn = load_synthetic(synthetic_file(model_name), synthetic_labels_file(model_name))
            y_syn = y_syn.astype(str)
        else:
            raise FileNotFoundError(
                f"No synthetic data found for {model_name}. "
                f"Expected either per-class chunks or {synthetic_file(model_name)}"
            )
        
        logger.info(f"Successfully loaded {X_syn.shape[0]} synthetic samples.")
        
        # Save full synthetic (in case of merge)
        save_synthetic(X_syn, y_syn, synthetic_file(model_name), synthetic_labels_file(model_name))
        
        # Phase 4 & 5 (Fidelity and TSTR)
        logger.info("Running Fidelity Evaluation...")
        t3 = time.time()
        fidelity_metrics = evaluate_fidelity(X_train, X_eval, X_syn, model_name, plot_dir(model_name))
        t_fidelity = time.time() - t3
        
        logger.info("Running TSTR Evaluation...")
        t4 = time.time()
        tstr_results = evaluate_tstr(X_train, y_train, X_eval, y_eval, X_syn, y_syn, model_name, device)
        t_tstr = time.time() - t4
        
        # Save TSTR
        tstr_path = tstr_file(model_name)
        with open(tstr_path, "w") as f:
            json.dump(tstr_results, f, indent=2, default=str)
            
        # Merge all metrics
        all_metrics = {**fidelity_metrics}
        all_metrics["tstr_summary"] = {
            clf: {
                "trtr_accuracy": d["trtr"]["accuracy"],
                "tstr_accuracy": d["tstr"]["accuracy"],
                "ratio": d["tstr_trtr_accuracy_ratio"],
            }
            for clf, d in tstr_results.get("classifiers", {}).items()
        }
        all_metrics["time_fidelity_s"] = round(t_fidelity, 1)
        all_metrics["time_tstr_s"] = round(t_tstr, 1)
        
        met_path = metrics_file(model_name)
        with open(met_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        logger.info("=" * 70)
        logger.info(f"MERGE & EVALUATE COMPLETE for {model_name}!")
        return all_metrics


    # =========================================================================
    # PATH B: GENERATE SPECIFIC CLASS (PARALLEL MODE)
    # =========================================================================
    if cancer_type != "all":
        f_data = synthetic_file(model_name).with_name(f"{model_name}_hdlss_syn_{cancer_type}.npy")
        f_lbl = synthetic_labels_file(model_name).with_name(f"{model_name}_hdlss_syn_labels_{cancer_type}.npy")
        if f_data.exists():
            logger.info(f"SKIP: Class {cancer_type} already generated. Delete {f_data.name} to re-run.")
            return {}
            
        logger.info(f"MODE: Single Class Generation for {cancer_type}")
        X, y, _, _ = load_dataset(cfg.DATA_DIR)
        X_train, X_eval, y_train, y_eval = stratified_split(X, y, TRAIN_RATIO, SEED)
        
        X_train_class = get_class_data(X_train, y_train, str(cancer_type))
        n_class = X_train_class.shape[0]
        n_syn_class = int(round(n_class * SYNTH_RATIO))
        
        generate_fn = get_generator(model_name)
        logger.info(f"  Generating {n_syn_class} samples for {cancer_type}...")
        
        X_syn_class = generate_fn(X_train_class, n_samples=n_syn_class, seed=SEED + hash(str(cancer_type)) % 10000, device=device)
        y_syn_class = np.array([str(cancer_type)] * n_syn_class)
        
        np.save(f_data, X_syn_class.astype(np.float32))
        np.save(f_lbl, y_syn_class)
        logger.info(f"Saved independent generation for {cancer_type}: {f_data.name}")
        return {}


    # =========================================================================
    # PATH C: FULL SEQUENTIAL (ORIGINAL MODE)
    # =========================================================================
    t0 = time.time()
    X, y, sample_ids, gene_names = load_dataset(cfg.DATA_DIR)
    X_train, X_eval, y_train, y_eval = stratified_split(X, y, TRAIN_RATIO, SEED)
    t_load = time.time() - t0

    t1 = time.time()
    generate_fn = get_generator(model_name)
    syn_parts = []
    syn_labels = []

    for c_type in cfg.CANCER_TYPES:
        X_train_class = get_class_data(X_train, y_train, str(c_type))
        n_syn_class = int(round(X_train_class.shape[0] * SYNTH_RATIO))
        X_syn_class = generate_fn(X_train_class, n_samples=n_syn_class, seed=SEED + hash(str(c_type)) % 10000, device=device)
        syn_parts.append(X_syn_class)
        syn_labels.extend([str(c_type)] * n_syn_class)

    X_syn = np.vstack(syn_parts).astype(np.float32)
    y_syn = np.array(syn_labels)
    t_gen = time.time() - t1

    save_synthetic(X_syn, y_syn, synthetic_file(model_name), synthetic_labels_file(model_name))

    t3 = time.time()
    fidelity_metrics = evaluate_fidelity(X_train, X_eval, X_syn, model_name, plot_dir(model_name))
    t_fidelity = time.time() - t3

    t4 = time.time()
    tstr_results = evaluate_tstr(X_train, y_train, X_eval, y_eval, X_syn, y_syn, model_name, device)
    tstr_path = tstr_file(model_name)
    with open(tstr_path, "w") as f:
        json.dump(tstr_results, f, indent=2, default=str)
    t_tstr = time.time() - t4

    # Merge metrics
    all_metrics = {**fidelity_metrics}
    all_metrics["tstr_summary"] = {clf: {"trtr_accuracy": d["trtr"]["accuracy"], "tstr_accuracy": d["tstr"]["accuracy"], "ratio": d["tstr_trtr_accuracy_ratio"]} for clf, d in tstr_results.get("classifiers", {}).items()}
    all_metrics["time_total_s"] = round(time.time() - t0, 1)
    
    with open(metrics_file(model_name), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    logger.info(f"DONE: {model_name} (Sequential Mode)")
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="HDLSS TCGA Synthetic Data Experiment Runner")
    parser.add_argument("--dataset", type=str, default="colon")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_NAMES + ["all"])
    parser.add_argument("--cancer_type", type=str, default="all")
    parser.add_argument("--merge_eval", action="store_true", help="Merge generated class files and run evaluations")
    
    args = parser.parse_args()

    # Pre-load data once to get classes
    temp_data_dir = cfg.PROJECT_ROOT / "Data" / args.dataset
    if not temp_data_dir.exists():
        print(f"Error: Dataset {args.dataset} not found in {temp_data_dir}")
        sys.exit(1)
        
    import pandas as pd
    import numpy as np
    labels_path = temp_data_dir / "labels.csv"
    df_l = pd.read_csv(labels_path, index_col=0)
    if "Class" in df_l.columns:
        discovered_classes = list(np.unique(df_l["Class"].values.astype(str)))
    else:
        discovered_classes = list(np.unique(df_l.iloc[:, 0].values.astype(str)))
        
    cfg.set_dataset(args.dataset, discovered_classes)


    models = MODEL_NAMES if args.model == "all" else [args.model]
    
    for model in models:
        setup_logging(model, args.cancer_type, args.merge_eval)
        try:
            run_single(model, args.cancer_type, args.merge_eval)
        except Exception as e:
            logging.getLogger(__name__).error(f"FAILED: {model} - {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
