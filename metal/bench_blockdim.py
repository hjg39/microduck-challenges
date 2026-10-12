"""A/B benchmark: Metal block_dim (threadgroup size) and torch thread count.

Run in a FRESH process per block_dim (kernels are compiled per block_dim and
cached; a fresh cache dir per run avoids cross-contamination). Does NOT touch
the live training run.

Usage:
  PYTHONPATH=. .venv-metal/bin/python metal/bench_blockdim.py --block-dim 256
  PYTHONPATH=. .venv-metal/bin/python metal/bench_blockdim.py --block-dim 512
  PYTHONPATH=. .venv-metal/bin/python metal/bench_blockdim.py --block-dim 1024
  PYTHONPATH=. .venv-metal/bin/python metal/bench_blockdim.py --threads 6
  PYTHONPATH=. .venv-metal/bin/python metal/bench_blockdim.py --threads 18
"""
import os
import subprocess
import sys
import tempfile
import threading
import time

import torch

# --- parse args -------------------------------------------------------------
argv = sys.argv
def _argval(flag):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None

# --- patch the Metal block_dim BEFORE any kernel compiles -------------------
bd = _argval("--block-dim")
bd = int(bd) if bd else None
if bd:
    import warp._src.context as _c
    # The options dict (block_dim=256 default) is built inline in Module.__init__;
    # wrap it so every Module gets our block_dim before any kernel compiles.
    _orig_init = _c.Module.__init__
    def _patched_init(self, *a, **k):
        _orig_init(self, *a, **k)
        self.options["block_dim"] = bd
    _c.Module.__init__ = _patched_init

# fresh kernel cache per run so the right block_dim variant is built
os.environ["WP_KERNEL_CACHE_DIR"] = tempfile.mkdtemp(prefix=f"warp_cache_bd{bd or 'x'}_")
import warp
warp.config.kernel_cache_dir = os.environ["WP_KERNEL_CACHE_DIR"]

import metal.bridge  # routes Warp physics to metal:0 (after warp import)

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from dataclasses import asdict
import microduck_challenges.sprint_2m  # registers the task

# --- torch thread count -----------------------------------------------------
nt = _argval("--threads")
nt = int(nt) if nt else None
if nt:
    torch.set_num_threads(nt)

NE = 2048
env_cfg = load_env_cfg("Mjlab-Sprint2m-MicroDuck")
env_cfg.scene.num_envs = NE
agent = load_rl_cfg("Mjlab-Sprint2m-MicroDuck")
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
agent_cfg = asdict(agent)
agent_cfg["logger"] = "tensorboard"  # match the live run; avoids the wandb/pydantic mismatch
runner_cls = load_runner_cls("Mjlab-Sprint2m-MicroDuck") or None
from mjlab.rl.runner import MjlabOnPolicyRunner
if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner
log_dir = tempfile.mkdtemp(prefix="bench_run_")
runner = runner_cls(wrapped, agent_cfg, log_dir, "cpu")

# --- GPU utilisation sampler (ioreg, ~2s period) ----------------------------
util_samples = []
stop = threading.Event()
def sample_gpu():
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["ioreg", "-r", "-c", "AGXAccelerator"],
                capture_output=True, text=True, timeout=5)
            for line in out.stdout.splitlines():
                if '"Device Utilization %"' in line:
                    v = int(line.split("=")[1].strip().rstrip(","))
                    util_samples.append(v)
                    break
        except Exception:
            pass
        time.sleep(1.0)
t = threading.Thread(target=sample_gpu, daemon=True)
t.start()

# --- warmup (compiles kernels, settles) -------------------------------------
print(f"[bench] NE={NE} block_dim={bd or 'default(256)'} torch_threads={nt or torch.get_num_threads()}")
print("[bench] warming up (3 iters, compiles kernels)...", flush=True)
t0 = time.time()
runner.learn(3, init_at_random_ep_len=True)
warm = time.time() - t0
print(f"[bench] warmup done in {warm:.1f}s", flush=True)

# --- verify the block_dim patch took effect (kernels now compiled) ----------
if bd:
    import warp as _wp
    dims = {m.options.get("block_dim") for m in _wp._src.context.user_modules.values()
            if "block_dim" in getattr(m, "options", {})}
    print(f"[bench] verified active block_dims={sorted(d for d in dims if d)} (requested {bd})", flush=True)

# --- timed: 5 full iterations (collection + PPO update) ----------------------
N = 5
t0 = time.perf_counter()
runner.learn(N, init_at_random_ep_len=False)
dt = (time.perf_counter() - t0) / N
stop.set()
t.join(timeout=6)

import statistics
util = util_samples
um = f"{statistics.mean(util):.0f}%" if util else "n/a"
up = f"{statistics.median(util):.0f}%" if util else "n/a"
ux = f"{max(util)}%" if util else "n/a"
print(f"RESULT block_dim={bd or 'default'} threads={nt or torch.get_num_threads()} "
      f"iter_time={dt*1000:.0f} ms  "
      f"gpu_util_mean={um}  gpu_util_p50={up}  gpu_util_max={ux}  "
      f"samples={len(util)}")
