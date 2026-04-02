#!/bin/bash
# Run the full baseline + ablation sweep for multiple seeds on the remote machine.
set -euo pipefail

SEEDS="${SEEDS:-42,43,44}"
NUM_STEPS="${NUM_STEPS:-30000}"
CHECKPOINTS="${CHECKPOINTS:-7000,8000,9000,10000,11000}"
SEED_LIST="${SEEDS//,/ }"

echo "=== Boundary sweep ==="
echo "Seeds: ${SEEDS}"
echo "Continuation steps per ablation: ${NUM_STEPS}"
echo "Checkpoints: ${CHECKPOINTS}"

for SEED in ${SEED_LIST}; do
  METRICS_OUTPUT="grokking_full_metrics_seed${SEED}.json"
  CHECKPOINTS_OUTPUT="grokking_checkpoints_seed${SEED}.pt"
  ABLATION_OUTPUT="grokking_ablation_results_seed${SEED}.json"

  echo
  if [[ -f "${METRICS_OUTPUT}" && -f "${CHECKPOINTS_OUTPUT}" ]]; then
    echo "=== Seed ${SEED}: baseline already present, skipping ==="
  else
    echo "=== Seed ${SEED}: baseline training ==="
    python3 grokking_full_metrics.py \
      --seed "${SEED}" \
      --num-steps 30000 \
      --metrics-output "${METRICS_OUTPUT}" \
      --checkpoints-output "${CHECKPOINTS_OUTPUT}"
  fi

  echo
  echo "=== Seed ${SEED}: ablation sweep ==="
  python3 grokking_ablation.py \
    --seed "${SEED}" \
    --checkpoint-path "${CHECKPOINTS_OUTPUT}" \
    --num-steps "${NUM_STEPS}" \
    --checkpoints "${CHECKPOINTS}" \
    --output "${ABLATION_OUTPUT}"
done

echo
echo "=== Boundary sweep complete ==="
