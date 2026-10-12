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
