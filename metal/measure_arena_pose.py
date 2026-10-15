"""Measure the trunk lean + head gaze of a policy in the ARENA.

Loads an exported .onnx, runs the arena's own step-through at a given vx, FKs
the trace against the walk-robot MJCF, and reports the trunk forward-pitch
(world) and the head gaze relative to the trunk, over the middle of the run.
Run with the LOCKED venv (.venv) — it has arena + mujoco 3.x.

Usage:
  cd ~/projects/microduck-challenges
  .venv/bin/python metal/measure_arena_pose.py --onnx <path> --vx 0.8 [--seed 1] [--label name]
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--vx", type=float, default=0.8)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--label", default="policy")
ap.add_argument("--skip", type=int, default=25, help="skip N frames at the start (settle/launch)")
args = ap.parse_args()

import arena.manifest
import arena.policy
import arena.runner
import arena.constants
import mujoco

m = arena.manifest.load_event("sprint-2m")
pol = arena.policy.load_policy(args.onnx)
out = arena.runner.run_once(m, pol, args.seed, None, np.array(m.command_at(args.vx)))

print(f"\n=== ARENA POSE  {args.label}  onnx={args.onnx}  vx={args.vx}  seed={args.seed}")
print(f"status={out.status}  time_s={out.time_s}  trace={np.asarray(out.trace).shape}")

trace = np.asarray(out.trace, dtype=np.float32)
assert trace.shape[1] == arena.constants.TRACE_WIDTH

# The walk robot's own MJCF (home frame: root at the floor).
import arena.assets.robot.legs as legs_pkg

xml_dir = list(legs_pkg.__path__)[0]
xml_path = None
for f in sorted(os.listdir(xml_dir)):
    if f.endswith(".xml"):
        xml_path = os.path.join(xml_dir, f)
        break
if xml_path is None:
    raise SystemExit("no legs xml found in " + xml_dir)

mj = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj)

trunk_body = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
neck_body = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "neck_pitch")
head_body = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "bottom_head_shell")
if head_body < 0:
    for cand in ("jaw_soft", "head_shell", "head"):
        h = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, cand)
        if h >= 0:
            head_body = h
            break
assert trunk_body >= 0 and neck_body >= 0 and head_body >= 0
print(f"bodies: trunk={trunk_body} neck={neck_body} head={head_body}  xml={os.path.basename(xml_path)}")


def world_y(xmat):
    """The body-frame FORWARD axis (local +y) in world, from the 9-elem body xmat."""
    R = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    return R @ np.array([0.0, 1.0, 0.0])


def pitch_of(v):
    """Angle of a world vector vs +y, positive = pitched DOWN (toward -z)."""
    return np.degrees(np.arctan2(-v[2], v[1]))


n = trace.shape[0]
start = min(args.skip, max(0, n - 5))
trunk_pitch, head_gaze, fwd_speed, x_progress = [], [], [], []
x0 = trace[start, 0]
for t in range(start, n):
    fr = trace[t]
    mj_data.qpos[:3] = fr[0:3]
    mj_data.qpos[3:7] = fr[3:7]
    mj_data.qpos[7:21] = fr[7:21]
    mujoco.mj_forward(mj, mj_data)
    v_trunk = world_y(mj_data.body(trunk_body).xmat)
    v_head = world_y(mj_data.body(head_body).xmat)
    trunk_pitch.append(pitch_of(v_trunk))
    # Signed: how far the head is pitched relative to the trunk's own forward
    # (nose-down positive) — the real "looking down" signal.
    head_gaze.append(pitch_of(v_head) - pitch_of(v_trunk))
    fwd_speed.append(float(fr[0]))
    x_progress.append(fr[0] - x0)

trunk_pitch = np.array(trunk_pitch)
head_gaze = np.array(head_gaze)
fwd_speed = np.array(fwd_speed)
x_progress = np.array(x_progress)

print(f"frames used: {start}..{n}  (dx={x_progress[-1]:.2f} m over {len(x_progress)} frames)")
print(f"trunk pitch (world, deg, + = nose DOWN): mean={trunk_pitch.mean():+.2f}  "
      f"p10={np.percentile(trunk_pitch, 10):+.2f}  p50={np.percentile(trunk_pitch, 50):+.2f}  "
      f"p90={np.percentile(trunk_pitch, 90):+.2f}")
print(f"head gaze vs trunk (deg, + = nose DOWN): mean={head_gaze.mean():+.2f}  "
      f"p10={np.percentile(head_gaze, 10):+.2f}  p50={np.percentile(head_gaze, 50):+.2f}  "
      f"p90={np.percentile(head_gaze, 90):+.2f}")
print(f"head world pitch (deg, + = DOWN): mean={(head_gaze + trunk_pitch).mean():+.2f}")
print(f"forward speed: mean={fwd_speed.mean():.3f} m/s  max={fwd_speed.max():.3f}")
if out.time_s:
    print(f"finish time: {out.time_s:.3f} s")

# --- leg kinematics: what is the stride actually doing ---
JN = arena.constants.JOINT_NAMES
def jcol(name):
    return np.stack([
        np.stack([t[7 + JN.index(name)] for t in trace[start:n]], axis=0)
        for _ in range(1)
    ]) if False else np.array([fr[7 + JN.index(name)] for fr in trace[start:n]])
print(f"\n--- LEG JOINT (rad, over {start}..{n}) ---")
for nm in ("left_hip_pitch", "left_knee", "left_ankle", "right_hip_pitch", "right_knee", "right_ankle"):
    if nm in JN:
        c = jcol(nm)
        print(f"  {nm:<15} mean={c.mean():+.3f}  min={c.min():+.3f}  max={c.max():+.3f}  swing={c.max()-c.min():.3f}")

# foot peak height (FK the foot bodies)
foot = {}
for fn in ("foot_left", "foot_right", "left_foot", "right_foot"):
    bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, fn)
    if bid >= 0:
        foot[fn] = bid
peak = {}
zmin_base = []
for t in range(start, n):
    fr = trace[t]
    mj_data.qpos[:7] = fr[0:7]
    mj_data.qpos[7:21] = fr[7:21]
    mujoco.mj_forward(mj, mj_data)
    for fn, bid in foot.items():
        z = mj_data.body(bid).xpos[2]
        peak[fn] = max(peak.get(fn, 0.0), z)
    zmin_base.append(mj_data.body(trunk_body).xpos[2])
for fn, z in peak.items():
    print(f"  foot {fn:<12} peak height {z:.3f} m   (trunk z mean {np.mean(zmin_base):.3f})")

