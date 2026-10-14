"""Measure the trunk orientation + absolute head direction of a trained
sprint_2m checkpoint, to explain the 'extended neck' perception.

Usage:
  PYTHONPATH=. .venv-metal/bin/python metal/measure_trunk.py --ckpt <path> --vx 0.8
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


ckpt = _argval("--ckpt")
vx = float(_argval("--vx", "0.8"))
steps = int(_argval("--steps", "500"))
ne = int(_argval("--envs", "64"))
seed = int(_argval("--seed", "1"))
warmup = int(_argval("--warmup", "100"))

os.environ["WP_KERNEL_CACHE_DIR"] = tempfile.mkdtemp(prefix="warp_cache_trunk_")
import warp

warp.config.kernel_cache_dir = os.environ["WP_KERNEL_CACHE_DIR"]
import metal.bridge  # noqa: F401

torch.manual_seed(seed)
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from rsl_rl.models import MLPModel
from dataclasses import asdict
from scipy.spatial.transform import Rotation
import microduck_challenges.sprint_2m  # noqa: F401
from microduck_challenges.sprint_2m.env import RL_CFG

cfg = load_env_cfg("Mjlab-Sprint2m-MicroDuck")
cfg.scene.num_envs = ne
env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=None)

obs_td = wrapped.get_observations()
ac = asdict(RL_CFG.actor)
pol = MLPModel(
    obs_td, {"actor": ["actor"]}, "actor", env.action_space.shape[-1],
    hidden_dims=tuple(ac["hidden_dims"]), activation=ac["activation"],
    obs_normalization=ac["obs_normalization"],
    distribution_cfg=dict(ac["distribution_cfg"]),
).to("cpu")
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
pol.load_state_dict(ck["actor_state_dict"])
pol.eval()

asset = env.scene["robot"]
neck_ids, _ = asset.find_bodies("neck_pitch")
neck_idx = neck_ids[0]

obs, _ = wrapped.reset()
tp, tr, ty, hz, fwd = [], [], [], [], []
for t in range(steps):
    c = env.command_manager.get_command("twist")
    c[:, 0] = vx; c[:, 1] = 0; c[:, 2] = 0
    for cn in ("head_pose", "body_pose"):
        env.command_manager.get_command(cn).zero_()
    o = wrapped.get_observations()
    with torch.no_grad():
        a = pol(o, stochastic_output=False)
    obs, rew, dones, ex = wrapped.step(a)
    if dones.any():
        obs, _ = wrapped.reset()
        o = wrapped.get_observations()
    if t >= warmup:
        q = asset.data.root_link_quat_w.cpu().numpy()[:, [1, 2, 3, 0]]
        e = Rotation.from_quat(q).as_euler("xyz", degrees=True)
        tp.append(e[:, 1]); tr.append(e[:, 0]); ty.append(e[:, 2])
        hz.append(asset.data.body_link_pos_w[:, neck_idx, 2].cpu().numpy())
        # forward direction of the trunk (x-axis in world)
        R = Rotation.from_quat(q)
        fwd.append(R.apply(torch.tensor([1.0, 0.0, 0.0]).numpy()))

import numpy as np
tp = np.array(tp); tr = np.array(tr); ty = np.array(ty)
hz = np.array(hz); fwd = np.array(fwd)
print(f"=== TRUNK ORIENTATION  ckpt={ckpt}  vx={vx}  envs={ne}  steps={steps} (warmup {warmup}) ===")
print(f"trunk pitch (forward lean): mean={tp.mean():+.2f} std={tp.std():.2f} min={tp.min():+.2f} max={tp.max():+.2f} deg")
print(f"trunk roll:                 mean={tr.mean():+.2f} std={tr.std():.2f} deg")
print(f"trunk yaw:                  mean={ty.mean():+.2f} std={ty.std():.2f} deg")
print(f"head z (neck body):         mean={hz.mean():.4f} min={hz.min():.4f} max={hz.max():.4f} m  (home ~0.20)")
print(f"trunk forward vector:       x={fwd[:,0].mean():.3f} y={fwd[:,1].mean():.3f} z={fwd[:,2].mean():+.3f}  (z>0 = pointing up)")
print(f"  -> trunk forward axis elevation: {np.degrees(np.arcsin(fwd[:,2].mean())):+.2f} deg above horizontal")
