"""Device bridge for running the MicroDuck stack on the Apple GPU (Metal).

mjlab's train.py picks a single device string and uses it for BOTH torch tensors
and the Warp physics. On a Mac there is no CUDA, so it hardcodes "cpu" — which
runs the physics on the CPU. To use the Apple GPU we:

  * keep torch tensors on "cpu" (unified memory: a CPU tensor and a Metal array
    share the same physical memory, so the zero-copy Warp<->torch interop still
    works, and no CUDA stream is needed), and
  * map the Warp physics device "cpu" -> "metal:0".

Import this module (or run `metal/patch.py` first) before building any env.
It is a no-op when the Metal backend is not active.
"""

import os

import torch
import warp as wp

# ---------------------------------------------------------------------------
# Graft normalizer freeze (walk25, 2026-10-19).
#
# The grafted LEADER actor (es2.onnx -> torch) was trained on the arena's
# command/velocity distribution. Its frozen EmpiricalNormalization stats
# (vx mean 0.668, ly/az pinned, the ~2.2 m/s bound amplitude) are the
# "arena contract": the leader's bound weights expect *that* input scale.
#
# rsl_rl's EmpiricalNormalization.update() freezes only when
#   `self.until is not None and self.count >= self.until`
# but `until` is a PLAIN constructor attribute (default None), NOT a buffer,
# so a checkpoint load (load_state_dict) restores _mean/_std/_var/count but
# leaves `until` at None -> update() re-estimates EVERY training step on our
# (walk-distribution) obs, denaturing the bound weights over the run.
# `count=10**12` alone is a no-op (the guard needs `until` non-None).
#
# Fix: pin `until=0` at CONSTRUCTION on the actor normalizer. The runner
# builds MLPModel fresh (then load_state_dict restores the graft's buffers),
# so the constructor is the one place `until` can be set; until=0 freezes
# permanently (count >= 0 always true). Gated on ACTOR_NORM_FREEZE so it
# never touches the critic (which must re-learn values for the bound policy,
# especially since walk25 also arena-matches the env and changes the obs
# distribution) or any non-graft run.
if os.environ.get("ACTOR_NORM_FREEZE", "0") == "1":
    from rsl_rl.models.mlp_model import MLPModel as _MLPModel
    _orig_mlp_init = _MLPModel.__init__

    def _frozen_mlp_init(self, *args, **kwargs):
        _orig_mlp_init(self, *args, **kwargs)
        # __init__ takes obs_set ("actor"/"critic") but does NOT store it;
        # store it here so the freeze can be targeted at the actor only.
        self.obs_set = kwargs.get("obs_set")
        if self.obs_set is None and len(args) >= 3:
            self.obs_set = args[2]
        # Only the ACTOR is the grafted leader; the critic must keep learning.
        if self.obs_set == "actor":
            if self.obs_normalization:
                self.obs_normalizer.until = 0  # freeze the normalizer permanently

    _MLPModel.__init__ = _frozen_mlp_init
    print("[metal.bridge] ACTOR_NORM_FREEZE=1 -> actor normalizer frozen (until=0)")

# The PPO update is CPU-bound (the GPU idles during it). torch defaults to a
# small thread count; on the M5 Max (18 cores) 18 threads shaves ~6% off the
# update (measured 867ms -> 816ms at 2048 envs). Override with METAL_TORCH_THREADS.
torch.set_num_threads(int(os.environ.get("METAL_TORCH_THREADS", "18")))

try:
    import warp_metal
    _ACTIVE = warp_metal.status()[0]
except Exception:  # warp-metal not installed
    _ACTIVE = False

if _ACTIVE and "metal:0" in wp.get_devices():
    _orig_get_device = wp.get_device

    def _get_device(s=None):
        if s is None:
            return _orig_get_device()
        if str(s) == "cpu":
            return _orig_get_device("metal:0")
        return _orig_get_device(s)

    wp.get_device = _get_device
    print("[metal.bridge] physics routed to metal:0 (torch tensors stay on cpu)")
else:
    print("[metal.bridge] Metal not active; running on CPU")
