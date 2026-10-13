"""Decisive parity test: (1) initial-obs parity after reset, (2) trained-policy
behaviour on the requested backend.

Part 1 rules out reset non-determinism (if initial obs differ, everything
diverges for trivial reasons). Part 2 loads the finished checkpoint and runs
the ACTUAL policy for N steps, reporting translation + contact + foot height —
the thing that matters for the arena.

Usage:
  PYTHONPATH=. .venv/bin/python       metal/parity2.py --backend cpu   --ckpt .../model_2999.pt
  PYTHONPATH=. .venv-metal/bin/python metal/parity2.py --backend metal --ckpt .../model_2999.pt
"""
import os
import sys
import tempfile

import torch

argv = sys.argv
def _argval(flag, default=None):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default

backend = _argval("--backend", "cpu")
steps = int(_argval("--steps", "500"))
ne = int(_argval("--envs", "256"))
seed = int(_argval("--seed", "0"))
ckpt = _argval("--ckpt")

if backend == "metal":
    os.environ["WP_KERNEL_CACHE_DIR"] = tempfile.mkdtemp(prefix="warp_cache_par2_")
    import warp
    warp.config.kernel_cache_dir = os.environ["WP_KERNEL_CACHE_DIR"]
    import metal.bridge  # noqa: F401

torch.manual_seed(seed)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from dataclasses import asdict
import microduck_challenges.sprint_2m  # noqa: F401

env_cfg = load_env_cfg("Mjlab-Sprint2m-MicroDuck")
env_cfg.scene.num_envs = ne
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=None)

# ---- Part 1: initial-obs parity ----
obs, _ = wrapped.reset()
print(f"=== {backend} RESET-OBS ===")
for k in sorted(obs.keys()):
    t = obs[k].float()
    print(f"  {k}: shape={tuple(t.shape)} mean={t.mean().item():.5f} std={t.std().item():.5f} "
          f"min={t.min().item():.5f} max={t.max().item():.5f}")

# ---- Part 2: trained policy behaviour ----
if not ckpt:
    print("(no --ckpt; skipping policy run)")
    sys.exit(0)

agent = load_rl_cfg("Mjlab-Sprint2m-MicroDuck")
agent_cfg = asdict(agent)
agent_cfg["logger"] = "tensorboard"
runner_cls = load_runner_cls("Mjlab-Sprint2m-MicroDuck")
from mjlab.rl.runner import MjlabOnPolicyRunner
if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner
import tempfile as _tf
runner = runner_cls(wrapped, agent_cfg, _tf.mkdtemp(prefix="par2_"), "cpu")
runner.load(ckpt)
print(f"=== {backend} POLICY loaded iter={runner.current_learning_iteration} ===")

# reset and run the policy
obs, _ = wrapped.reset()
asset = env.scene["robot"]
contact = env.scene["feet_ground_contact"].data.found
foot_h = env.scene["foot_height_scan"].data.heights
base0 = asset.data.root_link_pos_w.clone()

actor = runner.alg.get_policy()
actor.eval()
for i in range(steps):
    with torch.no_grad():
        a = actor(obs, stochastic_output=False)  # deterministic (mean) action
    obs, rew, term, trunc = wrapped.step(a)

base = asset.data.root_link_pos_w
disp = (base - base0).norm(dim=-1)
print(f"=== {backend} POLICY after {steps} steps ===")
print(f"  base_disp: mean={disp.mean().item():.4f} max={disp.max().item():.4f} (m moved from start)")
print(f"  base_pos: x_mean={base[:,0].mean().item():.4f} y_mean={base[:,1].mean().item():.4f} z_mean={base[:,2].mean().item():.4f}")
print(f"  contact_found: mean={contact.float().mean().item():.4f} (feet in contact, of 2)")
print(f"  foot_height: mean={foot_h.mean().item():.5f} max={foot_h.max().item():.5f}")
print(f"  env0 base_pos: {[f'{v:.4f}' for v in base[0].tolist()]}")
print(f"  env0 disp: {disp[0].item():.4f} m")
print(f"  env0 contact: {contact[0].tolist()}")
print(f"  env0 foot_height: {[f'{v:.5f}' for v in foot_h[0].tolist()]}")
