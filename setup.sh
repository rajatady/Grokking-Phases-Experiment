#!/bin/bash
# One-time setup for fresh RunPod server.
# Run from LOCAL machine: ./setup.sh
set -euo pipefail
source "$(dirname "$0")/remote.env"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${REMOTE_PORT}"
SSH="ssh ${SSH_OPTS} ${REMOTE_HOST}"

echo "=== Step 1: Install rsync ==="
$SSH "
  if ! command -v rsync >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq rsync 2>/dev/null
  else
    echo 'rsync already installed'
  fi
"

echo "=== Step 2: Sync local files to remote ==="
./sync.sh

echo "=== Step 3: Prepare remote workspace ==="
$SSH "
  mkdir -p ${REMOTE_DIR} /workspace/tmp
  grep -q '^TMPDIR=' /etc/environment 2>/dev/null || echo 'TMPDIR=/workspace/tmp' >> /etc/environment
  echo 'TMPDIR set to /workspace/tmp'
"

echo "=== Step 4: Install Python deps ==="
$SSH "
  pip install --break-system-packages torch==2.4.0 torchvision==0.19.0 transformers datasets accelerate huggingface_hub 2>&1 | tail -5
"

echo "=== Step 5: Verify CUDA ==="
$SSH "python3 -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')\""

echo "=== Setup complete ==="
