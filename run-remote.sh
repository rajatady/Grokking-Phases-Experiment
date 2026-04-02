#!/bin/bash
# Sync to remote, run a command under nohup, poll the log, then download results.
#
# Usage:
#   ./run-remote.sh <run_name> -- <command...>
# Example:
#   ./run-remote.sh full-metrics -- python3 grokking_full_metrics.py --num-steps 30000
set -euo pipefail
source "$(dirname "$0")/remote.env"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${REMOTE_PORT}"
SSH="ssh ${SSH_OPTS} ${REMOTE_HOST}"

if [[ $# -lt 3 || "${2}" != "--" ]]; then
  echo "Usage: $0 <run_name> -- <command...>"
  exit 1
fi

RUN_NAME="$1"
shift 2

REMOTE_LOG="${REMOTE_DIR}/${RUN_NAME}.log"
REMOTE_CMD="$*"

echo "=== Syncing to remote ==="
./sync.sh 2>&1 | tail -5

echo "=== Launching ${RUN_NAME} ==="
PID=$($SSH "source /etc/environment 2>/dev/null; mkdir -p ${REMOTE_DIR} /workspace/tmp; export TMPDIR=\${TMPDIR:-/workspace/tmp}; cd ${REMOTE_DIR} && PYTHONUNBUFFERED=1 nohup ${REMOTE_CMD} > ${REMOTE_LOG} 2>&1 & echo \$!")
echo "Remote PID: ${PID}"
echo "Remote log: ${REMOTE_LOG}"

echo "=== Polling every 30s ==="
while $SSH "ps -p ${PID} >/dev/null 2>&1"; do
  PROGRESS=$($SSH "tail -1 ${REMOTE_LOG} 2>/dev/null" | tr -d '\r')
  if [[ -n "${PROGRESS}" ]]; then
    echo "$(date '+%H:%M:%S') ${PROGRESS}"
  else
    echo "$(date '+%H:%M:%S') waiting for log output"
  fi
  sleep 30
done

echo "=== Final log tail ==="
$SSH "tail -n 40 ${REMOTE_LOG}" || true

echo "=== Downloading results ==="
./download-results.sh

echo "=== Done ==="
