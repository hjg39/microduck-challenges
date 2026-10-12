"""Apply the two site-packages patches needed to run the stack on Metal.

Run with the metal venv's python:  .venv-metal/bin/python metal/patch.py

1. mjlab/sim/sim_data.py — TorchArray._is_cuda must be based on the *torch*
   tensor's device, not the Warp device. A Metal (metal:0) Warp array is
   zero-copy with a CPU torch tensor, so it needs no CUDA stream even though
   the Warp device is not 'cpu'. (Otherwise _setup_stream calls
   torch.cuda.current_stream and crashes on a non-CUDA torch build.)

2. mujoco_warp/_src/sensor.py — _frame_axis references `xmat` in its UNKNOWN
   branch without assigning it. warp 1.17's stricter codegen rejects the
   undefined symbol (warp 1.12 tolerated it). Initialise xmat first.

Both patches are idempotent.
"""

import importlib.util
import os
import sys


def _patch_file(path: str, old: str, new: str, marker: str) -> None:
    if not os.path.exists(path):
        print(f"  skip (missing): {path}")
        return
    with open(path) as f:
        s = f.read()
    if marker in s:
        print(f"  already patched: {os.path.basename(path)}")
        return
    if old not in s:
        print(f"  WARN: pattern not found in {path}; skipped")
        return
    with open(path, "w") as f:
        f.write(s.replace(old, new))
    print(f"  patched: {path}")


def main() -> None:
    # mjlab/__init__.py lives in <site-packages>/mjlab/ -> two dirnames up.
    sp = os.path.dirname(os.path.dirname(os.path.abspath(__import__("mjlab").__file__)))
    print(f"site-packages: {sp}")

    print("1. mjlab sim_data.py (_is_cuda from torch device)")
    _patch_file(
        os.path.join(sp, "mjlab", "sim", "sim_data.py"),
        "    self._is_cuda = not self._wp_array.device.is_cpu  # type: ignore",
        "    self._is_cuda = self._tensor.device.type == \"cuda\"  # type: ignore",
        "self._tensor.device.type == \"cuda\"",
    )

    print("2. mujoco_warp sensor.py (initialise xmat)")
    _patch_file(
        os.path.join(sp, "mujoco_warp", "_src", "sensor.py"),
        "  frame_axis: int,\n) -> wp.vec3:\n  if objtype == ObjType.BODY:",
        "  frame_axis: int,\n) -> wp.vec3:\n"
        "  xmat = wp.mat33()  # initialise so the UNKNOWN branch has a defined symbol\n"
        "  if objtype == ObjType.BODY:",
        "initialise so the UNKNOWN branch",
    )

    print("done.")


if __name__ == "__main__":
    main()
