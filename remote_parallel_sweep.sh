#!/bin/bash
# Parallel boundary sweep: runs each (seed, ckpt, intervention) as a separate
# process so the GPU stays saturated. Each run writes to its own JSON file to
# avoid concurrent-write races. Merge with summarize_boundary_sweep.py after.
#
# Env vars (all optional):
#   SEEDS        comma-separated seeds            default: 42,43,44
#   NUM_STEPS    continuation steps per run       default: 30000
#   CHECKPOINTS  comma-separated ckpt steps       default: 7000,8000,9000,10000,11000
#   INTERVENTIONS comma-separated interventions   default: all 11
#   MAX_PARALLEL  max concurrent python jobs      default: 32
set -euo pipefail

SEEDS="${SEEDS:-42,43,44}"
NUM_STEPS="${NUM_STEPS:-30000}"
CHECKPOINTS="${CHECKPOINTS:-7000,8000,9000,10000,11000}"
INTERVENTIONS="${INTERVENTIONS:-baseline,no_weight_decay,freeze_head,freeze_embed,freeze_exit_layer,freeze_exit_attn,freeze_exit_mlp,freeze_entry_layer,freeze_middle_layers,freeze_attn_all,freeze_mlp_all}"
MAX_PARALLEL="${MAX_PARALLEL:-32}"

SEED_LIST="${SEEDS//,/ }"
CKPT_LIST="${CHECKPOINTS//,/ }"
IV_LIST="${INTERVENTIONS//,/ }"

echo "=== Parallel boundary sweep ==="
echo "Seeds: ${SEEDS}"
echo "Continuation steps per run: ${NUM_STEPS}"
echo "Checkpoints: ${CHECKPOINTS}"
echo "Max parallel jobs: ${MAX_PARALLEL}"

declare -a pids=()

# Throttle: wait for any one child to finish before launching more
throttle() {
    while [[ ${#pids[@]} -ge ${MAX_PARALLEL} ]]; do
        wait -n 2>/dev/null || true
        # Remove any finished pids from the list
        local alive=()
        for pid in "${pids[@]}"; do
            kill -0 "${pid}" 2>/dev/null && alive+=("${pid}") || true
        done
        pids=("${alive[@]+"${alive[@]}"}")
    done
}

for SEED in ${SEED_LIST}; do
    METRICS_OUTPUT="grokking_full_metrics_seed${SEED}.json"
    CKPTS_FILE="grokking_checkpoints_seed${SEED}.pt"

    # Baseline training must finish before we can launch ablations for this seed
    if [[ -f "${METRICS_OUTPUT}" && -f "${CKPTS_FILE}" ]]; then
        echo "Seed ${SEED}: baseline present, skipping training"
    else
        echo "Seed ${SEED}: training baseline (blocking)..."
        python3 grokking_full_metrics.py \
            --seed "${SEED}" \
            --num-steps 30000 \
            --metrics-output "${METRICS_OUTPUT}" \
            --checkpoints-output "${CKPTS_FILE}"
        echo "Seed ${SEED}: baseline done"
    fi

    for CKPT in ${CKPT_LIST}; do
        for IV in ${IV_LIST}; do
            OUTPUT="ablation_seed${SEED}_ckpt${CKPT}_${IV}.json"

            # Skip already-completed runs (resume support)
            if [[ -f "${OUTPUT}" ]]; then
                echo "  skip: seed=${SEED} ckpt=${CKPT} iv=${IV}"
                continue
            fi

            throttle

            echo "  launch: seed=${SEED} ckpt=${CKPT} iv=${IV}"
            python3 grokking_ablation.py \
                --checkpoint-path "${CKPTS_FILE}" \
                --checkpoints "${CKPT}" \
                --interventions "${IV}" \
                --num-steps "${NUM_STEPS}" \
                --seed "${SEED}" \
                --output "${OUTPUT}" \
                >> "ablation_seed${SEED}_ckpt${CKPT}_${IV}.log" 2>&1 &
            pids+=($!)
        done
    done
done

echo "Waiting for ${#pids[@]} remaining jobs..."
for pid in "${pids[@]+"${pids[@]}"}"; do
    wait "${pid}" || true
done

echo ""
echo "=== All runs complete. Merging results... ==="
python3 summarize_boundary_sweep.py ablation_seed*_ckpt*_*.json \
    --csv-output boundary_sweep_summary.csv \
    --md-output boundary_sweep_summary.md

echo "=== Done. Summary in boundary_sweep_summary.csv / .md ==="
