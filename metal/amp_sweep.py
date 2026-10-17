"""Hypothesis A test: command -> real forward speed amplification, in the ARENA.
Run with the LOCKED venv (.venv):  .venv/bin/python metal/amp_sweep.py --onnx <p> [--onnx2 <p>]
"""
import argparse, os, numpy as np, sys
ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--onnx2", default=None, help="second policy to compare")
ap.add_argument("--label", default="policy")
ap.add_argument("--label2", default="policy2")
args = ap.parse_args()
import arena.manifest, arena.policy, arena.runner, arena.constants
m = arena.manifest.load_event("sprint-2m")

def sweep(onnx, label):
    pol = arena.policy.load_policy(onnx)
    print(f"\n=== {label}  ({os.path.basename(onnx)}) ===")
    print(f"  cmd_vx  status   real_fwd_speed  amp_ratio  (mid-run m/s from trace)")
    rows=[]
    for vx in [0.25, 0.5, 0.8, 1.0, 1.15, 1.2, 1.6, 2.0, 2.5]:
        out = arena.runner.run_once(m, pol, 1, None, np.array(m.command_at(vx)))
        t = np.asarray(out.trace, np.float32)
        pos = t[:, 0:3]           # root xyz (world)
        dpos = np.linalg.norm(np.diff(pos, axis=0), axis=1) * (1.0/arena.constants.CTRL_DT)
        # steady window: skip settle/launch (30f) and last 20
        win = dpos[30:-20] if len(dpos) > 60 else dpos
        real = float(win.mean())
        amp = real / vx if vx > 0 else float("nan")
        rows.append((vx, out.status, real, amp))
        print(f"  {vx:5.2f}  {out.status:8s}  {real:5.2f} m/s        {amp:5.2f}x")
    return rows

sweep(args.onnx, args.label)
if args.onnx2:
    sweep(args.onnx2, args.label2)
