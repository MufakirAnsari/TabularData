#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# Submit all 35 experiments to SLURM
# CPU and GPU jobs run IN PARALLEL (different resource requests)
#
# Resume-safe: skips already-completed experiments.
# Usage: bash slurm/submit_all.sh
# ──────────────────────────────────────────────────────────────────────

cd $HOME/TabularData
mkdir -p logs

CPU_MODELS=("smote" "gaussian_copula" "scgft")
GPU_MODELS=("ctgan" "tvae" "tabddpm" "flow_matching")
SIZES=(50 100 200 500 1000)

echo "============================================================"
echo "Submitting HDLSS experiments: $(date)"
echo "============================================================"

CPU_COUNT=0
GPU_COUNT=0
SKIP_COUNT=0

# ── Submit CPU jobs (no GPU requested) ──
echo ""
echo "--- CPU JOBS ---"
for MODEL in "${CPU_MODELS[@]}"; do
    for N in "${SIZES[@]}"; do
        JOB_NAME="hdlss_${MODEL}_n${N}"

        SYN_FILE="results/synthetic/${MODEL}_n${N}.npy"
        MET_FILE="results/metrics/${MODEL}_n${N}_metrics.json"
        if [ -f "$SYN_FILE" ] && [ -f "$MET_FILE" ]; then
            echo "  SKIP: ${JOB_NAME} (completed)"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

        echo "  SUBMIT: ${JOB_NAME} (CPU, 1h)"
        sbatch \
            --job-name=${JOB_NAME} \
            --account=pwsu0516 \
            --time=01:00:00 \
            --mem=32G \
            --nodes=1 \
            --ntasks-per-node=1 \
            --output=logs/slurm_%x_%j.out \
            --error=logs/slurm_%x_%j.err \
            --export=MODEL=${MODEL},N=${N} \
            slurm/run_job.sbatch

        CPU_COUNT=$((CPU_COUNT + 1))
        sleep 0.5
    done
done

# ── Submit GPU jobs (1 GPU requested) ──
echo ""
echo "--- GPU JOBS ---"
for MODEL in "${GPU_MODELS[@]}"; do
    for N in "${SIZES[@]}"; do
        JOB_NAME="hdlss_${MODEL}_n${N}"

        SYN_FILE="results/synthetic/${MODEL}_n${N}.npy"
        MET_FILE="results/metrics/${MODEL}_n${N}_metrics.json"
        if [ -f "$SYN_FILE" ] && [ -f "$MET_FILE" ]; then
            echo "  SKIP: ${JOB_NAME} (completed)"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

        echo "  SUBMIT: ${JOB_NAME} (GPU, 6h)"
        sbatch \
            --job-name=${JOB_NAME} \
            --account=pwsu0516 \
            --gpus-per-node=1 \
            --time=06:00:00 \
            --mem=64G \
            --nodes=1 \
            --ntasks-per-node=1 \
            --output=logs/slurm_%x_%j.out \
            --error=logs/slurm_%x_%j.err \
            --export=MODEL=${MODEL},N=${N} \
            slurm/run_job.sbatch

        GPU_COUNT=$((GPU_COUNT + 1))
        sleep 0.5
    done
done

echo ""
echo "============================================================"
echo "Submitted: ${CPU_COUNT} CPU + ${GPU_COUNT} GPU = $((CPU_COUNT + GPU_COUNT)) jobs"
echo "Skipped:   ${SKIP_COUNT} (already completed)"
echo "CPU and GPU jobs run IN PARALLEL"
echo "Monitor:   squeue -u \$USER"
echo "============================================================"
