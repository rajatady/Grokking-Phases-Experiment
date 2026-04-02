#!/bin/bash
# Launch the multi-seed parallel boundary sweep on RunPod through run-remote.sh.
set -euo pipefail

source "$(dirname "$0")/remote.env"

SEEDS="${SEEDS:-42,43,44}"
NUM_STEPS="${NUM_STEPS:-30000}"
CHECKPOINTS="${CHECKPOINTS:-7000,8000,9000,10000,11000}"
MAX_PARALLEL="${MAX_PARALLEL:-32}"

SSH_OPTS="-o UserKnownHostsFile=/dev/null -p ${REMOTE_PORT}"
if ssh ${SSH_OPTS} "${REMOTE_HOST}" 'ps -ef | grep "[r]emote_parallel_sweep.sh" >/dev/null'; then
  echo "A boundary sweep is already running on the remote host. Refusing to launch a duplicate."
  exit 1
fi

./run-remote.sh boundary-sweep -- env \
  SEEDS="${SEEDS}" \
  NUM_STEPS="${NUM_STEPS}" \
  CHECKPOINTS="${CHECKPOINTS}" \
  MAX_PARALLEL="${MAX_PARALLEL}" \
  bash remote_parallel_sweep.sh
