"""CPU-vs-Metal physics parity check.

Builds the sprint env on the requested backend, feeds an IDENTICAL fixed
random action sequence (independent of observations, so any divergence is
purely physics), and reports the resulting robot state. Run it once with
--backend cpu (main .venv) and once with --backend metal (.venv-metal +
bridge), then diff the two outputs.

Usage:
  PYTHONPATH=. .venv/bin/python     metal/parity.py --backend cpu   --steps 200
  PYTHONPATH=. .venv-metal/bin/python metal/parity.py --backend metal --steps 200
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
steps = int(_argval("--steps", "200"))
ne = int(_argval("--envs", "256"))
seed = int(_argval("--seed", "0"))

if backend == "metal":
    os.environ["WP_KERNEL_CACHE_DIR"] = tempfile.mkdtemp(prefix="warp_cache_par_")
    import warp
    warp.config.kernel_cache_dir = os.environ["WP_KERNEL_CACHE_DIR"]
    import metal.bridge  # noqa: F401  (patches wp.get_device -> metal:0)

torch.manual_seed(seed)
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
import microduck_challenges.sprint_2m  # noqa: F401  (registers the task)

env_cfg = load_env_cfg("Mjlab-Sprint2m-MicroDuck")
env_cfg.scene.num_envs = ne
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=None)

# reset (deterministic given the same seed set above)
obs, _ = wrapped.reset()

# fixed action sequence, independent of observations
gen = torch.Generator().manual_seed(seed)
action_dim = env.action_space.shape[-1] if hasattr(env.action_space, "shape") else 12
actions = [torch.randn(env.num_envs, action_dim, generator=gen) for _ in range(steps)]

# run the fixed actions
for a in actions:
    obs, rew, term, trunc = wrapped.step(a)

# gather the resulting robot state
asset = env.scene["robot"]
base_pos = asset.data.root_link_pos_w
base_quat = asset.data.root_link_quat_w
joint_pos = asset.data.joint_pos
contact = env.scene["feet_ground_contact"].data.found
foot_h = env.scene["foot_height_scan"].data.heights

def summarize(name, t):
    t = t.float()
    print(f"{name}: mean={t.mean().item():.5f} std={t.std().item():.5f} "
          f"min={t.min().item():.5f} max={t.max().item():.5f}")

print(f"=== PARITY backend={backend} envs={ne} steps={steps} seed={seed} ===")
print(f"base_pos_x: mean={base_pos[:,0].mean().item():.5f} std={base_pos[:,0].std().item():.5f}")
print(f"base_pos_y: mean={base_pos[:,1].mean().item():.5f} std={base_pos[:,1].std().item():.5f}")
print(f"base_pos_z: mean={base_pos[:,2].mean().item():.5f} std={base_pos[:,2].std().item():.5f}")
print(f"base_quat: mean={base_quat.mean().item():.5f} std={base_quat.std().item():.5f}")
summarize("joint_pos", joint_pos)
print(f"contact_found: mean={contact.float().mean().item():.5f} (fraction of feet in contact)")
summarize("foot_height", foot_h)
# a few concrete envs to diff by hand
print("env0 base_pos:", [f"{v:.5f}" for v in base_pos[0].tolist()])
print("env0 base_quat:", [f"{v:.5f}" for v in base_quat[0].tolist()])
print("env0 joint_pos[:6]:", [f"{v:.5f}" for v in joint_pos[0, :6].tolist()])
print("env0 contact:", contact[0].tolist())
print("env0 foot_height:", [f"{v:.5f}" for v in foot_h[0].tolist()])
