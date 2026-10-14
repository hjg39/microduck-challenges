"""Measure the actual joint pose of a trained sprint_2m checkpoint.

Loads a model_<N>.pt, runs the policy with the Arena's command (constant vx,
zero head/body commands) for a fixed number of steps, and reports the mean /
std / min / max of every joint angle over the rollout, plus the deviation from
the HOME pose. This grounds Harry's visual observations (extended neck,
outward knees) in real numbers.

Usage:
  PYTHONPATH=. .venv-metal/bin/python metal/measure_pose.py \
      --ckpt logs/rsl_rl/sprint_2m/<run>/model_5998.pt \
      --vx 0.8 --steps 500 --envs 64 --seed 1
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
warmup = int(_argval("--warmup", "100"))  # skip the first N steps (settle)

os.environ["WP_KERNEL_CACHE_DIR"] = tempfile.mkdtemp(prefix="warp_cache_pose_")
import warp

warp.config.kernel_cache_dir = os.environ["WP_KERNEL_CACHE_DIR"]
import metal.bridge  # noqa: F401  (patches wp.get_device -> metal:0)

torch.manual_seed(seed)
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from tensordict import TensorDict
import microduck_challenges.sprint_2m  # noqa: F401  (registers the task)

env_cfg = load_env_cfg("Mjlab-Sprint2m-MicroDuck")
env_cfg.scene.num_envs = ne
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=None)

# Build the actor model from the task's RL_CFG, then load the checkpoint's
# weights + observation normalizer into it.
from rsl_rl.models import MLPModel
from microduck_challenges.sprint_2m.env import RL_CFG

obs_td = wrapped.get_observations()
from dataclasses import asdict

actor_cfg = asdict(RL_CFG.actor)
policy = MLPModel(
    obs_td,
    {"actor": ["actor"]},
    "actor",
    env.action_space.shape[-1],
    hidden_dims=tuple(actor_cfg["hidden_dims"]),
    activation=actor_cfg["activation"],
    obs_normalization=actor_cfg["obs_normalization"],
    distribution_cfg=dict(actor_cfg["distribution_cfg"]),
).to("cpu")

ck = torch.load(ckpt, map_location="cpu", weights_only=False)
policy.load_state_dict(ck["actor_state_dict"])
# Restore the trained observation normalizer (mean/var/count) so inference
# matches the normalizer baked into the exported ONNX.
if "actor obs_normalizer._mean" in ck["infos"]:
    policy.obs_normalizer._mean.copy_(ck["infos"]["actor obs_normalizer._mean"])
    policy.obs_normalizer._var.copy_(ck["infos"]["actor obs_normalizer._var"])
    policy.obs_normalizer._std.copy_(ck["infos"]["actor obs_normalizer._std"])
    policy.obs_normalizer.count.copy_(ck["infos"]["actor obs_normalizer.count"])
policy.eval()

# The robot's joint order. The entity has 16 joints (14 servos + 2 passive jaw
# linkage); the data tensors index all of them, so keep the full list and only
# report the 14 actuated ones.
asset = env.scene["robot"]
all_joint_names = list(asset.joint_names)
print("ALL JOINTS:", all_joint_names)
actuated = [n for n in all_joint_names if not n.startswith("passive_")]
print("ACTUATED (14):", actuated)

# HOME pose (from microduck_constants.HOME_FRAME) for deviation reporting.
HOME = {
    "left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
    "left_knee": -0.0049, "left_ankle": 0.4530,
    "neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579,
    "right_knee": 0.0049, "right_ankle": -0.4530,
}

obs, _ = wrapped.reset()

# Force the Arena command: constant vx, zero head/body. The command obs terms
# read from the env's command manager; we overwrite the command buffer directly.
cmd = env.command_manager.get_command("twist")  # [ne, 3]
cmd[:, 0] = vx
cmd[:, 1] = 0.0
cmd[:, 2] = 0.0
for cname in ("head_pose", "body_pose"):
    c = env.command_manager.get_command(cname)
    c.zero_()

# Run the rollout, collecting joint positions.
qpos_hist = []
base_z_hist = []
for t in range(steps):
    # Re-assert the command each step (the command manager resamples on its
    # own timer; we want it pinned to the Arena value).
    cmd = env.command_manager.get_command("twist")
    cmd[:, 0] = vx
    cmd[:, 1] = 0.0
    cmd[:, 2] = 0.0
    for cname in ("head_pose", "body_pose"):
        c = env.command_manager.get_command(cname)
        c.zero_()

    obs_td = wrapped.get_observations()
    with torch.no_grad():
        actions = policy(obs_td, stochastic_output=False)
    obs, rew, dones, extras = wrapped.step(actions)
    # Reset any envs that terminated (fall) so the rollout stays populated.
    if dones.any():
        obs, _ = wrapped.reset()
        obs_td = wrapped.get_observations()

    qpos_hist.append(asset.data.joint_pos.clone())
    base_z_hist.append(asset.data.root_link_pos_w[:, 2].clone())

qpos = torch.stack(qpos_hist, dim=0)  # [steps, ne, 14]
base_z = torch.stack(base_z_hist, dim=0)  # [steps, ne]

# Measure over the steady-state window (skip warmup).
q = qpos[warmup:]
bz = base_z[warmup:]

idx = {n: all_joint_names.index(n) for n in actuated}
print(f"\n=== POSE MEASUREMENT  ckpt={ckpt}  vx={vx}  envs={ne}  steps={steps} (warmup {warmup}) ===")
print(f"base_z: mean={bz.mean().item():.4f} min={bz.min().item():.4f} max={bz.max().item():.4f}  (home ~0.125)")
print(f"{'joint':<16} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  {'home':>8} {'dev':>8}")
for name in actuated:
    col = q[:, :, idx[name]]
    home = HOME.get(name, 0.0)
    dev = col.mean().item() - home
    print(f"{name:<16} {col.mean().item():8.4f} {col.std().item():8.4f} "
          f"{col.min().item():8.4f} {col.max().item():8.4f}  {home:8.4f} {dev:+8.4f}")

# Head/neck extension summary (the "neck extended" observation).
neck = q[:, :, idx["neck_pitch"]]
headp = q[:, :, idx["head_pitch"]]
print(f"\nNECK+HEAD pitch (extension): neck mean={neck.mean().item():.4f} "
      f"head mean={headp.mean().item():.4f}  combined={ (neck.mean()+headp.mean()).item():.4f} rad "
      f"= {(neck.mean()+headp.mean()).item()*57.2958:.1f} deg")
print(f"  (HOME combined = {(0.3491+0.3491)*57.2958:.1f} deg)")

# Knee splay summary (the "knees angled outward" observation).
lr = q[:, :, idx["left_hip_roll"]]
rr = q[:, :, idx["right_hip_roll"]]
lk = q[:, :, idx["left_knee"]]
rk = q[:, :, idx["right_knee"]]
print(f"\nHIP ROLL (splay): left mean={lr.mean().item():.4f} right mean={rr.mean().item():.4f}  "
      f"(HOME: left -0.0873, right +0.0873)")
print(f"KNEE: left mean={lk.mean().item():.4f} right mean={rk.mean().item():.4f}  (HOME ~0)")
