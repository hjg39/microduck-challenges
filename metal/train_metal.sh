#!/usr/bin/env bash
# Train the MicroDuck sprint policy on the Apple GPU (Metal).
#
# Usage:
#   metal/train_metal.sh <run-name> [num-envs] [max-iterations]
#   metal/train_metal.sh first 2048 3000
#
# The bridge routes the Warp physics to metal:0; torch tensors stay on cpu
# (unified memory, zero-copy). Checkpoints land in logs/rsl_rl/sprint_2m/ as
# usual, so the workshop and `arena score` work unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="${1:-first}"
NUM_ENVS="${2:-2048}"
MAX_ITERS="${3:-3000}"
SEED="${4:-1}"

mkdir -p logs
LOG="logs/${RUN_NAME}.metal.log"

# Load the Metal bridge in-process, then hand off to the normal train CLI.
.venv-metal/bin/python -c "
import metal.bridge
from mjlab_microduck.train_cli import main
import sys
sys.argv = ['train', 'Mjlab-Sprint2m-MicroDuck',
            '--gpu-ids', 'None',
            '--env.scene.num-envs', '$NUM_ENVS',
            '--agent.max-iterations', '$MAX_ITERS',
            '--agent.seed', '$SEED',
            '--agent.logger', 'tensorboard',
            '--agent.run-name', '$RUN_NAME']
main()
" > "$LOG" 2>&1 &

echo "Metal training '$RUN_NAME' ($NUM_ENVS envs, $MAX_ITERS iters) started."
echo "  log: $LOG"
echo "  track: grep -oE 'Learning iteration [0-9]+/[0-9]+' $LOG | tail -1"
