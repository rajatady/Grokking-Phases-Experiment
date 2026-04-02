#!/bin/bash
# Download remote results from RunPod into results/<seed>/{json,logs}/.
set -euo pipefail
source "$(dirname "$0")/remote.env"

SSH_OPTS="-p ${REMOTE_PORT}"
INCLUDE_CHECKPOINTS=0

if [[ "${1:-}" == "--include-checkpoints" ]]; then
  INCLUDE_CHECKPOINTS=1
fi

SEEDS="${SEEDS:-42,43,44}"

for SEED in ${SEEDS//,/ }; do
  mkdir -p results/seed${SEED}/json results/seed${SEED}/logs

  # JSONs
  rsync -avz --no-owner --no-group \
    --include="ablation_seed${SEED}_*.json" \
    --include="grokking_full_metrics_seed${SEED}.json" \
    --exclude="*" \
    -e "ssh ${SSH_OPTS}" \
    "${REMOTE_HOST}:${REMOTE_DIR}/" \
    results/seed${SEED}/json/

  # Logs
  rsync -avz --no-owner --no-group \
    --include="ablation_seed${SEED}_*.log" \
    --exclude="*" \
    -e "ssh ${SSH_OPTS}" \
    "${REMOTE_HOST}:${REMOTE_DIR}/" \
    results/seed${SEED}/logs/

  # Checkpoints (optional)
  if [[ "${INCLUDE_CHECKPOINTS}" -eq 1 ]]; then
    mkdir -p results/checkpoints
    rsync -avz --no-owner --no-group \
      --include="grokking_checkpoints_seed${SEED}.pt" \
      --exclude="*" \
      -e "ssh ${SSH_OPTS}" \
      "${REMOTE_HOST}:${REMOTE_DIR}/" \
      results/checkpoints/
  fi
done

# Sweep-level logs
rsync -avz --no-owner --no-group \
  --include="boundary-sweep.log" \
  --include="*.csv" \
  --include="*.md" \
  --exclude="*" \
  -e "ssh ${SSH_OPTS}" \
  "${REMOTE_HOST}:${REMOTE_DIR}/" \
  results/logs/
