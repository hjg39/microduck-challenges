"""Detect reward-hack hopping: measures trunk-height oscillation (the vertical
component air_time pays for) against forward progress. A real stride keeps the
COM within a few cm; a hop oscillates it a lot while gaining little x.

trace (T, 21): x, y, z, qw, qx, qy, qz, j0..j13   (z = trunk/root height)
"""
import argparse
import numpy as np
import arena.manifest, arena.policy, arena.runner
from arena.constants import CTRL_DT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--vx", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label", default="hop")
    a = ap.parse_args()

    m = arena.manifest.load_event("sprint-2m")
    pol = arena.policy.load_policy(a.onnx)
    out = arena.runner.run_once(m, pol, seed=a.seed, command=m.command_at(a.vx))
    tr = out.trace
    n0 = tr.shape[0]
    skip = min(25, max(0, n0 - 5))
    x = tr[skip:, 0]
    z = tr[skip:, 2]
    n = len(z)
    if n < 10:
        print(f"{a.label} vx={a.vx} {out.status}: trace too short ({n})"); return
    t_s = n * CTRL_DT
    dx = float(x[-1] - x[0])
    amp = float(np.percentile(z, 95) - np.percentile(z, 5))
    # count oscillation peaks (local maxima above the mid-band)
    zc = z - z.mean()
    mid = np.median(np.abs(zc)) + 0.005
    up = zc > mid
    peaks = int(np.sum(up & ~np.roll(up, 1) & ~up[0]))
    true = dx / t_s if t_s > 0 else 0
    print(f"{a.label} vx={a.vx} {out.status}  t={t_s:.1f}s")
    print(f"  trunk z: mean={z.mean():.3f}  amp(p5..p95)={amp:.3f} m  osc-freq={peaks/max(t_s,1e-6):.1f} Hz")
    print(f"  forward dx={dx:+.2f} m  TRUE={true:+.2f} m/s")

if __name__ == "__main__":
    main()
