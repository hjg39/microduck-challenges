"""Isolated PPO-update benchmark: time ONLY the CPU-side alg.update() at
different torch thread counts. The update is pure CPU (the GPU is idle during
it), so this isolates exactly what the thread count affects, free of the GPU
physics noise that dominates the full iteration.

Usage:
  PYTHONPATH=. .venv-metal/bin/python metal/bench_update.py --threads 6
  PYTHONPATH=. .venv-metal/bin/python metal/bench_update.py --threads 18
"""
import os
import sys
import tempfile
import time

import torch

argv = sys.argv
def _argval(flag):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None

nt = _argval("--threads")
nt = int(nt) if nt else None
if nt:
    torch.set_num_threads(nt)

os.environ["WP_KERNEL_CACHE_DIR"] = tempfile.mkdtemp(prefix="warp_cache_upd_")
import warp
warp.config.kernel_cache_dir = os.environ["WP_KERNEL_CACHE_DIR"]
import metal.bridge

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from dataclasses import asdict
import microduck_challenges.sprint_2m

NE = 2048
env_cfg = load_env_cfg("Mjlab-Sprint2m-MicroDuck")
env_cfg.scene.num_envs = NE
agent = load_rl_cfg("Mjlab-Sprint2m-MicroDuck")
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
agent_cfg = asdict(agent)
agent_cfg["logger"] = "tensorboard"
runner_cls = load_runner_cls("Mjlab-Sprint2m-MicroDuck")
from mjlab.rl.runner import MjlabOnPolicyRunner
if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner
log_dir = tempfile.mkdtemp(prefix="bench_upd_")
runner = runner_cls(wrapped, agent_cfg, log_dir, "cpu")

# warmup: fill the rollout buffer + compile
print(f"[upd] NE={NE} torch_threads={nt or torch.get_num_threads()}", flush=True)
runner.learn(3, init_at_random_ep_len=True)

# time ONLY the PPO update (the CPU-side half), several reps
import statistics
times = []
for _ in range(5):
    t0 = time.perf_counter()
    runner.alg.update()
    times.append((time.perf_counter() - t0) * 1000)
print(f"RESULT threads={nt or torch.get_num_threads()} "
      f"update_mean={statistics.mean(times):.0f} ms  "
      f"update_min={min(times):.0f}  update_max={max(times):.0f}  "
      f"reps={[f'{t:.0f}' for t in times]}")
