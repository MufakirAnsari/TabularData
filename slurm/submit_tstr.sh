#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# Submit all TSTR validation jobs (7 gen models × 5 sample sizes = 35)
#
# All jobs request 1 GPU (for XGBoost acceleration).
# Each job runs 7 classifiers in both TRTR and TSTR mode (~5-15 min each).
# Resume-safe: skips completed experiments.
#
# After all jobs finish, run plots:
#   python -m src.run_tstr --model all --n all --plots-only
#
# Usage: bash slurm/submit_tstr.sh
# ──────────────────────────────────────────────────────────────────────

cd $HOME/TabularData
mkdir -p logs results/tstr results/tstr_plots

MODELS=("smote" "gaussian_copula" "ctgan" "tvae" "tabddpm" "flow_matching" "scgft")
SIZES=(50 100 200 500 1000)

echo "============================================================"
echo "Submitting TSTR Validation: $(date)"
echo "============================================================"

SUBMIT_COUNT=0
SKIP_COUNT=0

for MODEL in "${MODELS[@]}"; do
    for N in "${SIZES[@]}"; do
        JOB_NAME="tstr_${MODEL}_n${N}"

        # Check if synthetic data exists first
        SYN_FILE="results/synthetic/${MODEL}_n${N}.npy"
        if [ ! -f "$SYN_FILE" ]; then
            echo "  WAIT: ${JOB_NAME} — synthetic data not ready yet"
            continue
        fi

        # Resume check
        TSTR_FILE="results/tstr/${MODEL}_n${N}_tstr.json"
        if [ -f "$TSTR_FILE" ]; then
            echo "  SKIP: ${JOB_NAME} (completed)"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

        echo "  SUBMIT: ${JOB_NAME}"
        sbatch \
            --job-name=${JOB_NAME} \
            --account=pwsu0516 \
            --gpus-per-node=1 \
            --time=01:00:00 \
            --mem=32G \
            --nodes=1 \
            --ntasks-per-node=1 \
            --output=logs/slurm_%x_%j.out \
            --error=logs/slurm_%x_%j.err \
            --export=MODEL=${MODEL},N=${N} \
            slurm/run_tstr.sbatch

        SUBMIT_COUNT=$((SUBMIT_COUNT + 1))
        sleep 0.5
    done
done

echo ""
echo "============================================================"
echo "Submitted: ${SUBMIT_COUNT} TSTR jobs"
echo "Skipped:   ${SKIP_COUNT} (already completed)"
echo ""
echo "After all jobs finish, generate plots:"
echo "  python -m src.run_tstr --model all --n all --plots-only"
echo "============================================================"
