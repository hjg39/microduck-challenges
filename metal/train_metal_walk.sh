#!/usr/bin/env bash
# Train the MicroDuck sprint WALKING variant on the Apple GPU (Metal).
#
# Usage:
#   metal/train_metal_walk.sh <run-name> [num-envs] [max-iterations] [seed]
#   metal/train_metal_walk.sh smoke 512 150 1
#
# Same bridge as train_metal.sh, but trains Mjlab-Sprint2m-MicroDuck with the
# reward re-shaped to translate forward (the walk fixes are folded into the
# sprint_2m env). Checkpoints land in logs/rsl_rl/sprint_2m/ as usual.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="${1:-walk}"
NUM_ENVS="${2:-2048}"
MAX_ITERS="${3:-3000}"
SEED="${4:-1}"

mkdir -p logs
LOG="logs/${RUN_NAME}.metal.log"

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
