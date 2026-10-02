"""
Smoke test: Run all 7 models on a tiny synthetic dataset (N=10, P=100)
to verify all imports, shapes, and post-processing work correctly.

Usage (on OSC):
    python smoke_test.py

Usage (local):
    python smoke_test.py --cpu
"""
import sys
import time
import logging
import numpy as np
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Tiny test data: 10 cells, 100 genes, ~80% zeros, values in [0, 5]
def make_test_data(N=10, P=100, sparsity=0.80, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.exponential(0.5, size=(N, P)).astype(np.float32)
    mask = rng.random((N, P)) < sparsity
    X[mask] = 0.0
    return X

def run_smoke_test(use_cpu=False):
    device = "cpu" if use_cpu else "cuda"
    
    # Try to detect GPU
    if not use_cpu:
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("No GPU detected, falling back to CPU")
                device = "cpu"
            else:
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            device = "cpu"

    X_train = make_test_data(N=10, P=100, sparsity=0.80)
    n_samples = 5
    
    logger.info(f"Test data: shape={X_train.shape}, sparsity={np.mean(X_train==0)*100:.1f}%, device={device}")
    
    models = [
        ("smote",           "src.models.smote_gen"),
        ("gaussian_copula", "src.models.gaussian_copula_gen"),
        ("ctgan",           "src.models.ctgan_gen"),
        ("tvae",            "src.models.tvae_gen"),
        ("tabddpm",         "src.models.tabddpm_gen"),
        ("flow_matching",   "src.models.flow_matching_gen"),
        ("scgft",           "src.models.scgft_gen"),
    ]
    
    results = []
    
    for name, module_path in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING: {name}")
        logger.info(f"{'='*60}")
        
        t0 = time.time()
        try:
            # Dynamic import
            mod = __import__(module_path, fromlist=["generate"])
            generate = mod.generate
            
            # Run generation
            X_syn = generate(X_train, n_samples=n_samples, seed=42, device=device)
            elapsed = time.time() - t0
            
            # Validate output
            checks = []
            
            # Shape check
            expected_shape = (n_samples, X_train.shape[1])
            shape_ok = X_syn.shape == expected_shape
            checks.append(("Shape", shape_ok, f"{X_syn.shape} == {expected_shape}"))
            
            # Dtype check
            dtype_ok = X_syn.dtype == np.float32
            checks.append(("Dtype", dtype_ok, f"{X_syn.dtype}"))
            
            # Non-negative check
            non_neg_ok = np.all(X_syn >= 0)
            checks.append(("Non-negative", non_neg_ok, f"min={X_syn.min():.4f}"))
            
            # No NaN/Inf check
            finite_ok = np.all(np.isfinite(X_syn))
            checks.append(("Finite", finite_ok, f"nan={np.isnan(X_syn).sum()}, inf={np.isinf(X_syn).sum()}"))
            
            # Sparsity check (should be > 0% — model is generating some zeros)
            syn_sparsity = np.mean(X_syn == 0) * 100
            sparsity_ok = syn_sparsity > 0
            checks.append(("Sparsity>0%", sparsity_ok, f"{syn_sparsity:.1f}%"))
            
            all_passed = all(c[1] for c in checks)
            status = "PASS" if all_passed else "FAIL"
            
            for check_name, ok, detail in checks:
                symbol = "✓" if ok else "✗"
                logger.info(f"  {symbol} {check_name}: {detail}")
            
            logger.info(f"  Time: {elapsed:.2f}s")
            results.append((name, status, elapsed, syn_sparsity))
            
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"  CRASH: {e}")
            logger.error(traceback.format_exc())
            results.append((name, "CRASH", elapsed, -1))
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("SMOKE TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Model':<20} {'Status':<8} {'Time(s)':<10} {'Sparsity':<10}")
    logger.info("-" * 50)
    
    all_ok = True
    for name, status, elapsed, sparsity in results:
        sp_str = f"{sparsity:.1f}%" if sparsity >= 0 else "N/A"
        logger.info(f"{name:<20} {status:<8} {elapsed:<10.2f} {sp_str:<10}")
        if status != "PASS":
            all_ok = False
    
    logger.info("-" * 50)
    if all_ok:
        logger.info("ALL 7 MODELS PASSED! Safe to submit full SLURM jobs.")
    else:
        logger.info("SOME MODELS FAILED! Fix errors before submitting.")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    use_cpu = "--cpu" in sys.argv
    sys.exit(run_smoke_test(use_cpu))
