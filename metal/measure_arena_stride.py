"""Stride analysis: world-frame foot positions + trunk posture from an arena trace.
Answers: how long are the strides, what's the cadence, and how far forward is
the torso pitched? (Leaders bend ~90 deg and stride wide; ours stay upright.)

trace (T, 21): x, y, z, qw, qx, qy, qz, j0..j13  (14 joints; ankle=joint 4/13)
"""
import argparse
import numpy as np
import mujoco
import arena.manifest, arena.policy, arena.runner, arena.course
from arena.constants import JOINT_NAMES, CTRL_DT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--vx", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label", default="x")
    a = ap.parse_args()
    m = arena.manifest.load_event("sprint-2m")
    pol = arena.policy.load_policy(a.onnx)
    out = arena.runner.run_once(m, pol, seed=a.seed, command=m.command_at(a.vx))
    tr = out.trace
    skip = min(25, max(0, tr.shape[0] - 5))
    tr = tr[skip:]
    n = tr.shape[0]
    if n < 10:
        print(f"{a.label}: too short ({n})"); return

    mj = arena.course.compile_course(m)
    d = mujoco.MjData(mj)
    lf_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rf_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    trunk_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    qj = {nm: i for i, nm in enumerate(JOINT_NAMES)}
    lf_x, rf_x, lf_y, rf_y, lf_z, rf_z, pitch, roll = [], [], [], [], [], [], [], []
    for t in range(n):
        d.qpos[:3] = tr[t, :3]; d.qpos[3:7] = tr[t, 3:7]
        d.qpos[7:21] = tr[t, 7:21]
        mujoco.mj_forward(mj, d)
        lp = d.site(lf_id).xpos
        rp = d.site(rf_id).xpos
        lf_x.append(lp[0]); rf_x.append(rp[0]); lf_y.append(lp[1]); rf_y.append(rp[1])
        lf_z.append(lp[2]); rf_z.append(rp[2])
        # trunk posture: same world-frame-forward-axis convention as
        # measure_arena_pose.py (body +y forward; pitch positive = nose DOWN).
        R = d.body(trunk_id).xmat.reshape(3, 3)
        vf = R @ np.array([0.0, 1.0, 0.0])
        vr = R @ np.array([0.0, 0.0, 1.0])
        pitch.append(np.degrees(np.arctan2(-vf[2], vf[1])))  # nose down > 0
        roll.append(np.degrees(np.arctan2(vr[2], np.hypot(vr[0], vr[1]))))  # +z up-comp of world-up axis
    lf_x, rf_x = np.array(lf_x), np.array(rf_x)
    lf_z, rf_z = np.array(lf_z), np.array(rf_z)
    pitch = np.array(pitch)
    t_s = n * CTRL_DT
    # stride length: one step = foot leaves then returns; approximate via x-oscillation of each foot
    def osc(x):
        return float(np.percentile(x, 90) - np.percentile(x, 10))
    lf_span, rf_span = osc(lf_x), osc(rf_x)
    lf_air = (lf_z > np.min(lf_z) + 0.015).mean()
    rf_air = (rf_z > np.min(rf_z) + 0.015).mean()
    dx_run = float(tr[-1, 0] - tr[0, 0])
    # cadence: count left-foot plantings (air -> ground transitions)
    air_l = lf_z > np.min(lf_z) + 0.015
    steps_l = int(np.sum((~air_l) & np.roll(air_l, 1) & air_l[0]))
    cadence = (steps_l * 2) / max(t_s, 1e-6)  # both feet
    print(f"{a.label} vx={a.vx} {out.status} t={t_s:.2f}s  dx={dx_run:+.2f}m true={dx_run/max(t_s,1e-6):+.2f} m/s")
    print(f"  left foot:  x-span={lf_span:.3f} m  air%={lf_air*100:.0f}  right foot: x-span={rf_span:.3f} m air%={rf_air*100:.0f}")
    roll = np.array(roll)
    print(f"  cadence~{cadence:.1f} steps/s")
    print(f"  trunk pitch (nose down +): mean={pitch.mean():+.1f} p10={np.percentile(pitch,10):+.1f} p50={np.median(pitch):+.1f} p90={np.percentile(pitch,90):+.1f} deg")
    print(f"  trunk roll  (left lean +): mean={roll.mean():+.1f} p10={np.percentile(roll,10):+.1f} p50={np.median(roll):+.1f} p90={np.percentile(roll,90):+.1f} deg")
    print(f"  foot z range: left [{lf_z.min():.3f},{lf_z.max():.3f}]  right [{rf_z.min():.3f},{rf_z.max():.3f}]  (ground={min(lf_z.min(), rf_z.min()):.3f})")

if __name__ == "__main__":
    main()
