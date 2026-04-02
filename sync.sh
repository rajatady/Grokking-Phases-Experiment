#!/bin/bash
# Sync local repo to remote RunPod server.
set -euo pipefail
source "$(dirname "$0")/remote.env"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${REMOTE_PORT}"
SSH="ssh ${SSH_OPTS} ${REMOTE_HOST}"

$SSH "mkdir -p ${REMOTE_DIR}"

rsync -avz --delete --no-owner --no-group \
  --exclude '.venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude 'run.log' \
  --exclude '*.log' \
  --exclude '*.json' \
  --exclude '*.csv' \
  --exclude '*.md' \
  --exclude 'results/' \
  --exclude '.idea' \
  --exclude 'remote.env' \
  --exclude '*.pt' \
  --exclude '*.bin' \
  -e "ssh ${SSH_OPTS}" \
  ./ \
  "${REMOTE_HOST}:${REMOTE_DIR}/"
