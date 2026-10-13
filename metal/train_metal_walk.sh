#!/usr/bin/env bash
# Train the MicroDuck sprint WALKING variant on the Apple GPU (Metal).
#
# Usage:
#   metal/train_metal_walk.sh <run-name> [num-envs] [max-iterations] [seed] [resume-run-regex]
#   metal/train_metal_walk.sh smoke 512 150 1
#   metal/train_metal_walk.sh walk3 2048 3000 1 '2026-10-13_08-44-51_walk2'
#
# Same bridge as train_metal.sh, but trains Mjlab-Sprint2m-MicroDuck with the
# reward re-shaped to translate forward (the walk fixes are folded into the
# sprint_2m env). Checkpoints land in logs/rsl_rl/sprint_2m/ as usual.
#
# The optional 5th arg is a regex matched against the run directory names in
# logs/rsl_rl/sprint_2m/; when set, the run RESUMES from the latest checkpoint
# of the newest matching run (weights, normalizers, optimizer and the
# step-based curricula are all restored — this is a same-task continuation, so
# the curricula stay at their final stage rather than restarting). Leave it
# empty to train from scratch.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="${1:-walk}"
NUM_ENVS="${2:-2048}"
MAX_ITERS="${3:-3000}"
SEED="${4:-1}"
RESUME_RUN="${5:-}"

mkdir -p logs
LOG="logs/${RUN_NAME}.metal.log"

# Build the resume flags only when a resume-run regex is supplied.
# NOTE: tyro runs with FlagConversionOff, so the boolean needs an explicit
# value (--agent.resume true), not a bare flag.
RESUME_FLAGS=()
if [ -n "$RESUME_RUN" ]; then
  RESUME_FLAGS=(--agent.resume true --agent.load-run "$RESUME_RUN")
fi

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
            '--agent.run-name', '$RUN_NAME'] + $RESUME_FLAGS
main()
" > "$LOG" 2>&1 &

if [ -n "$RESUME_RUN" ]; then
  echo "Metal training '$RUN_NAME' ($NUM_ENVS envs, $MAX_ITERS iters) started, RESUMING from '$RESUME_RUN'."
else
  echo "Metal training '$RUN_NAME' ($NUM_ENVS envs, $MAX_ITERS iters) started (from scratch)."
fi
echo "  log: $LOG"
echo "  track: grep -oE 'Learning iteration [0-9]+/[0-9]+' $LOG | tail -1"
