"""
TSTR Validation Runner — Train on Synthetic, Test on Real.

Runs 7 classifiers in TRTR and TSTR mode across all generative models
and sample sizes. Resume-safe: skips completed experiments.

Usage:
    python -m src.run_tstr --model smote --n 50
    python -m src.run_tstr --model all --n all
"""
import argparse
import json
import logging
import sys
import time
import traceback
import numpy as np
import scanpy as sc
from pathlib import Path
from datetime import datetime

# ── Setup project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR, RESULTS_DIR, SYNTHETIC_DIR, LOGS_DIR,
    SAMPLE_SIZES, SEED, MODEL_NAMES,
    train_file, eval_file, synthetic_file,
)
from src.data_utils import load_h5ad_as_numpy
from src.tstr_validation import run_tstr_for_one, get_real_threshold
from src.tstr_plots import (
    compile_tstr_results, plot_tstr_bars,
    plot_tstr_heatmaps, plot_tstr_by_sample_size,
)


# Output directories
TSTR_DIR = RESULTS_DIR / "tstr"
TSTR_PLOTS_DIR = RESULTS_DIR / "tstr_plots"


def setup_logging(model: str, n: int) -> None:
    """Configure logging to both console and file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"tstr_{model}_n{n}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(ch)

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(fh)
    logging.info(f"Logging to: {log_file}")


def tstr_result_file(model: str, n: int) -> Path:
    """Path to the TSTR results JSON."""
    return TSTR_DIR / f"{model}_n{n}_tstr.json"


def is_tstr_completed(model: str, n: int) -> bool:
    """Check if TSTR for this (model, N) is already done."""
    f = tstr_result_file(model, n)
    if f.exists():
        try:
            with open(f) as fh:
                data = json.load(fh)
            # Verify it has classifier results
            return "classifiers" in data and len(data["classifiers"]) == 7
        except (json.JSONDecodeError, KeyError):
            return False
    return False


def run_single_tstr(model_name: str, n: int) -> dict:
    """Run TSTR validation for one (model, N) combination."""
    logger = logging.getLogger(__name__)

    # Resume check
    if is_tstr_completed(model_name, n):
        logger.info(f"SKIP: TSTR {model_name}/n={n} already completed.")
        with open(tstr_result_file(model_name, n)) as f:
            return json.load(f)

    logger.info("=" * 70)
    logger.info(f"TSTR VALIDATION: model={model_name}, N={n}")
    logger.info("=" * 70)

    t0 = time.time()

    # ── Load data ──
    logger.info("Loading data...")
    train_path = train_file(n)
    eval_path = eval_file(n)
    syn_path = synthetic_file(model_name, n)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")
    if not syn_path.exists():
        raise FileNotFoundError(f"Synthetic file not found: {syn_path}")

    # Load with adata to get n_genes from .obs
    adata_train = sc.read_h5ad(train_path)
    threshold = get_real_threshold(adata_train)
    logger.info(f"  n_genes threshold (median): {threshold:.0f}")

    X_train, _ = load_h5ad_as_numpy(train_path)
    X_eval, _ = load_h5ad_as_numpy(eval_path)
    X_syn = np.load(syn_path)

    logger.info(f"  Train: {X_train.shape}, Eval: {X_eval.shape}, Syn: {X_syn.shape}")

    # Detect device
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            try:
                logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            except Exception:
                logger.info("  GPU detected")
    except ImportError:
        pass
    logger.info(f"  Device: {device}")

    # ── Run TSTR ──
    results = run_tstr_for_one(
        X_train_real=X_train,
        X_eval=X_eval,
        X_syn=X_syn,
        threshold=threshold,
        gen_model=model_name,
        n_samples=n,
        device=device,
    )

    # ── Save results ──
    TSTR_DIR.mkdir(parents=True, exist_ok=True)
    out_path = tstr_result_file(model_name, n)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    t_total = time.time() - t0
    results["time_total_s"] = round(t_total, 1)

    # Re-save with timing
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"DONE: TSTR {model_name}/n={n} in {t_total:.1f}s ({t_total/60:.1f} min)")
    logger.info(f"  Saved to: {out_path}")

    return results


def generate_all_plots():
    """Generate all TSTR comparison plots from completed results."""
    logger = logging.getLogger(__name__)
    
    df = compile_tstr_results(TSTR_DIR)
    if df.empty:
        logger.warning("No TSTR results found to plot.")
        return

    logger.info(f"Compiling {len(df)} TSTR results for plotting...")

    # Save CSV
    csv_path = TSTR_DIR / "tstr_all_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"  CSV: {csv_path}")

    # Generate plots
    TSTR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_tstr_bars(df, TSTR_PLOTS_DIR)
    plot_tstr_heatmaps(df, TSTR_PLOTS_DIR)
    plot_tstr_by_sample_size(df, TSTR_PLOTS_DIR)

    logger.info(f"All plots saved to {TSTR_PLOTS_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="TSTR Validation Runner"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=MODEL_NAMES + ["all"],
        help="Generative model to validate (or 'all')"
    )
    parser.add_argument(
        "--n", type=str, required=True,
        help="Sample size (50, 100, 200, 500, 1000) or 'all'"
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Only generate plots from existing results (no classifier runs)"
    )
    args = parser.parse_args()

    if args.plots_only:
        setup_logging("plots", 0)
        generate_all_plots()
        return

    # Parse
    sample_sizes = SAMPLE_SIZES if args.n == "all" else [int(args.n)]
    models = MODEL_NAMES if args.model == "all" else [args.model]

    results_list = []
    errors = []

    for model in models:
        for n in sample_sizes:
            setup_logging(model, n)
            logger = logging.getLogger(__name__)
            try:
                result = run_single_tstr(model, n)
                results_list.append(result)
            except Exception as e:
                error_msg = f"FAILED: TSTR {model}/n={n}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                errors.append({"model": model, "n": n, "error": str(e)})
                continue

    # Final summary
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 70)
    logger.info("TSTR VALIDATION COMPLETE")
    logger.info(f"  Successful: {len(results_list)}")
    logger.info(f"  Failed:     {len(errors)}")
    if errors:
        for e in errors:
            logger.info(f"    {e['model']}/n={e['n']}: {e['error']}")

    # Generate plots if we ran everything
    if args.model == "all" and args.n == "all" and len(errors) == 0:
        logger.info("\nGenerating comparison plots...")
        generate_all_plots()


if __name__ == "__main__":
    main()
