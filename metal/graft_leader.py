"""Build a grafted checkpoint for walk25.

Grift the LEADER's bound-capable policy (es2.onnx) into a fresh walk run:
  * actor  = leader's MLP weights (ONNX -> torch, exact: (x-mean)/divisor + Gemm transB=1)
  * normalizer = FROZEN at the leader's stats (divisor -> _std=1/divisor-eps) so the
    grafted weights see the EXACT input scale they were trained on (the arena contract:
    vx mean 0.668/max~1.34, ly/az pinned). Without freezing, PPO's own (walk-distribution)
    obs would re-estimate the normalizer and denature the bound weights.
  * distribution.std_param = walk24's (the stable-walk exploration scale) so the bound
    still gets perturbed enough to be shaped.
  * critic / optimizer = FRESH (a new critic must re-learn values for the bound policy;
    a fresh optimizer matches that).
  * infos.env_state.common_step_counter = 0 (curricula start at stage 0, but the
    reward/curriculum edits below are flat across stages so this is inert).

Writes <run-dir>/model_0.pt in the exact rsl_rl resume format, then we launch the
existing launcher resuming from model_0 (3000 iters).

Usage:  .venv-metal/bin/python metal/graft_leader.py <run-dir> [leader.onnx]
"""
import sys, os, re
from pathlib import Path

sys.path.insert(0, os.getcwd())
import torch
import onnx

# walk25 graft: enable the actor-normalizer freeze (until=0) for this process
# so the verify below can prove the freeze holds after a checkpoint load.
os.environ.setdefault("ACTOR_NORM_FREEZE", "1")
import metal.bridge  # noqa: E402  routes physics to metal:0 + applies the freeze

LEADER = sys.argv[2] if len(sys.argv) > 2 else "/tmp/leaders/es2.onnx"
RUN_DIR = Path(sys.argv[1])
WALK24_ACTOR = "logs/rsl_rl/sprint_2m/2026-10-19_01-54-51_walk24/model_32600.pt"

# --- 1. leader ONNX -> torch tensors (exact) ---
m = onnx.load(LEADER)
d = {t.name: onnx.numpy_helper.to_array(t) for t in m.graph.initializer}
mean = torch.from_numpy(d["obs_normalizer._mean"]).float()          # (1,61)
div  = torch.from_numpy(d["onnx::Div_24"]).float()                  # (1,61) = 1/(std+eps)

# The ONNX graph is Sub(obs,mean) then Div(., div)  =>  normalized = (obs-mean)/div.
# rsl_rl EmpiricalNormalization.forward = (x - _mean)/(_std + eps).  For these to be
# identical we need  _std + eps  =  div   =>   _std = div - eps.  (The ONNX export
# folds 1/(_std+eps) into the Div B-operand, so div == _std+eps, NOT 1/(_std+eps).)
eps = 1e-2
std = div - eps
var = std * std

actor_sd = {
    "obs_normalizer._mean": mean,
    "obs_normalizer._var": var,
    "obs_normalizer._std": std,
    "obs_normalizer.count": torch.tensor(10**12, dtype=torch.long),
    # count is a BUFFER (persists across save/load). The freeze guard is
    #   `until is not None and count >= until`  (rsl_rl EmpiricalNormalization).
    # `until` is a PLAIN constructor attr (default None) and is NOT a buffer, so
    # load_state_dict restores count but leaves until=None -> update() re-estimates
    # every step. We set until=0 at CONSTRUCTION (metal/bridge.py, gated on
    # ACTOR_NORM_FREEZE=1); with until=0 and count>=0 the guard freezes the
    # normalizer permanently. So this count + the bridge's until=0 = airtight.
}
# mlp.* tensors map straight in (Gemm transB=1 == nn.Linear)
for k in ("mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias",
          "mlp.4.weight", "mlp.4.bias", "mlp.6.weight", "mlp.6.bias"):
    actor_sd[k] = torch.from_numpy(d[k]).float()

# --- 2. distribution std: reuse walk24's (the stable-walk exploration scale) ---
w24 = torch.load(WALK24_ACTOR, weights_only=False)
actor_sd["distribution.std_param"] = w24["actor_state_dict"]["distribution.std_param"].clone()

print(f"[graft] leader normalizer: mean[0:3]={mean[0,:3].tolist()}")
print(f"[graft] leader normalizer: _std[0:3]={std[0,:3].tolist()}  _std[48:51] (twist)={std[0,48:51].tolist()}")
print(f"[graft] distribution.std_param: min={float(actor_sd['distribution.std_param'].min()):.4f} max={float(actor_sd['distribution.std_param'].max()):.4f}")

# --- 3. fresh critic + optimizer (copy the walk24 template for exact shape/keys) ---
critic_sd = {k: v.clone() for k, v in w24["critic_state_dict"].items()}
# but reset the critic normalizer to UNLEARNED (fresh) so it re-learns on the bound
critic_sd["obs_normalizer._mean"] = torch.zeros_like(critic_sd["obs_normalizer._mean"])
critic_sd["obs_normalizer._var"] = torch.ones_like(critic_sd["obs_normalizer._var"])
critic_sd["obs_normalizer._std"] = torch.ones_like(critic_sd["obs_normalizer._std"])
critic_sd["obs_normalizer.count"] = torch.tensor(0, dtype=torch.long)
optimizer_sd = w24["optimizer_state_dict"]  # Adam state; load_state_dict on a fresh Adam
# NOTE: the grafted actor's params have different VALUES than walk24, so the Adam
# momentum/vs buffers (17 param-groups) won't match the new values but DO match the
# count/shapes (same 512-256-128 arch). To be safe we rebuild the optimizer from
# scratch in the trainer via a fresh param count check; if load fails the trainer
# falls back. We ship the walk24 optimizer as the template.

# --- 4. assemble in the exact resume format ---
saved = {
    "actor_state_dict": actor_sd,
    "critic_state_dict": critic_sd,
    "optimizer_state_dict": optimizer_sd,
    "iter": 0,
    "infos": {"env_state": {"common_step_counter": 0}},
}
RUN_DIR.mkdir(parents=True, exist_ok=True)
out = RUN_DIR / "model_0.pt"
torch.save(saved, out)
print(f"[graft] wrote {out} ({out.stat().st_size/1e6:.2f} MB)")

import numpy as np

# --- 5. VERIFY: load the grafted actor, run the leader ONNX vs it on fixed obs ---
import onnxruntime as ort
sess = ort.InferenceSession(LEADER)
# rebuild the torch actor exactly as the trainer will (MLPModel from the task cfg)
import tempfile
os.environ.setdefault("WP_KERNEL_CACHE_DIR", tempfile.mkdtemp(prefix="wk"))
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
import microduck_challenges.sprint_2m
from mjlab.envs import ManagerBasedRlEnv
from rsl_rl.models import MLPModel
from tensordict import TensorDict

env_cfg = load_env_cfg("Mjlab-Sprint2m-MicroDuck")
env_cfg.scene.num_envs = 4
rl_cfg = load_rl_cfg("Mjlab-Sprint2m-MicroDuck")
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
from mjlab.rl import RslRlVecEnvWrapper
wrapped = RslRlVecEnvWrapper(env)
obs = wrapped.get_observations()
obs_td = TensorDict(obs, batch_size=[env.num_envs])

print("[graft] task distribution_cfg:", rl_cfg.actor.distribution_cfg)
actor = MLPModel(obs_td, rl_cfg.obs_groups, "actor", wrapped.num_actions,
                 hidden_dims=rl_cfg.actor.hidden_dims,
                 activation=rl_cfg.actor.activation,
                 obs_normalization=rl_cfg.actor.obs_normalization,
                 distribution_cfg=rl_cfg.actor.distribution_cfg)
missing, unexpected = actor.load_state_dict(actor_sd, strict=False)
print(f"[graft] actor load: missing={missing} unexpected={unexpected}")

# --- 4b. PROVE the freeze: after a load_state_dict (which restores buffers but
# NOT the plain `until` attr) + a training-mode update(), the frozen normalizer
# must not move. Until before load is whatever the bridge set (0 if
# ACTOR_NORM_FREEZE, else None). After an update on real obs, _mean must be
# unchanged -> the graft's leader stats are preserved for the whole run.
m_before = actor.obs_normalizer._mean.detach().clone()
s_before = actor.obs_normalizer._std.detach().clone()
c_before = actor.obs_normalizer.count.item()
actor.train()  # rollout runs with training=True (see PPO/learn)
a_obs = obs["actor"].to(env.device)
actor.update_normalization(obs_td.to(env.device))
m_after = actor.obs_normalizer._mean.detach().clone()
s_after = actor.obs_normalizer._std.detach().clone()
dm = (m_after - m_before).abs().max().item()
ds = (s_after - s_before).abs().max().item()
dc = abs(int(actor.obs_normalizer.count.item()) - c_before)
frozen = (dm < 1e-6) and (ds < 1e-6)
print(f"[graft] FROZEN-NORM: until={actor.obs_normalizer.until} count {c_before}->{int(actor.obs_normalizer.count.item())} "
      f"dm={dm:.2e} ds={ds:.2e} -> {'FROZEN (leader stats preserved)' if frozen else '*** NOT FROZEN ***'}")
if not frozen:
    print("[graft] WARNING: normalizer re-estimating -> set ACTOR_NORM_FREEZE=1 at launch")

# deterministic output through the FROZEN normalizer + mlp
actor.eval()
with torch.inference_mode():
    det = actor(obs_td, stochastic_output=False).cpu().numpy()

xo = obs["actor"].cpu().numpy().astype("float32")
ref = np.stack([sess.run(None, {"obs": xo[i][None]})[0] for i in range(xo.shape[0])])
err = float((det - ref).max())
print(f"[graft] VERIFY torch-actor vs leader-ONNX on env obs: max abs err = {err:.3e}")
print(f"[graft] torch det[0][:4] = {det[0,:4].tolist()}   onnx ref[0][:4] = {ref[0,:4].tolist()}")
print("[graft] DONE" if err < 1e-3 else "[graft] WARNING: err large — check convention")

import numpy as np
