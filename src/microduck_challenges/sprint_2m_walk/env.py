"""The sprint's WALKING variant: same Arena event, reward re-shaped to translate.

The "first" run (sprint_2m) plateaued at ~0 m/s forward: the policy learned to
stand upright and shuffle its feet (satisfying upright/pose/heading/head_pose)
while the forward-velocity reward — a tight Gaussian — was dead (error ~1.4 m/s
vs std 0.316 -> reward ~0, no gradient). Standing scored higher, so it stood
still, and the entropy collapsed (19.8 -> -9.4) so it couldn't escape.

This variant keeps the base env byte-for-byte (imported, not copied) and layers
on the fixes:

  1. forward_speed_reward (NEW)  : tanh(clamp(vx,0)/0.3) — monotonic, command-
                                   independent forward gradient. Pulls the duck
                                   out of the standing basin even when the
                                   tracking Gaussian is dead.
  2. track_linear_velocity std   : 0.316 -> 0.5 (curriculum, loose->tight) so the
                                   Gaussian has a live gradient at the current error.
  3. foot_swing_height           : flipped -0.25 -> +0.5, target 0.02 -> 0.05 m.
                                   Reward feet that LIFT, not feet that drag.
  4. leg_symmetry_reward (NEW)   : -mean|q_left + q_right| over the leg joints —
                                   reward the centred mirror pose (Harry's idea).
  5. gait_symmetry_penalty (NEW) : penalise cumulative L/R swing imbalance —
                                   reward a balanced stride (complements #4).
  6. velocity-command-range      : forward-only curriculum 0.1 -> 1.2 m/s, so the
                                   early target is easy and the gait bootstraps.
  7. standing_envs               : capped at 0.10 (was 0.25) and delayed, so the
                                   standing fraction stops reinforcing standing.
  8. entropy_coef                : 0.01 -> 0.03 in the RL config (more early
                                   exploration; the "first" run collapsed to 0.13).

The actor cannot see foot height/air time (its obs are joint_pos/vel, actions,
base_ang_vel, projected_gravity, commands), so the foot-lift and symmetry terms
shape the policy indirectly through joint angles. If the next run still
under-lifts, the stronger lever is adding foot_height to the ACTOR obs (it is
already in the critic).
"""

from copy import deepcopy

from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab_microduck import challenge
from mjlab_microduck.tasks import mdp as microduck_mdp

from ..sprint_2m.env import NUM_STEPS_PER_ENV, RL_CFG, make_env_cfg as _base_make_env_cfg


def make_env_cfg(play: bool = False):
    """The walking variant: base sprint env + the reward/curriculum fixes above."""
    cfg = _base_make_env_cfg(play=play)

    # --- 1. forward_speed_reward: monotonic forward gradient (the key new term) ---
    # tanh(clamp(vx, 0)/vel_ref): rewards ANY forward motion, never penalises
    # standing (vx=0 -> 0), saturates at vel_ref. Command-independent, so it gives
    # a live gradient even when the tracking Gaussian is dead.
    cfg.rewards["forward_speed"] = RewardTermCfg(
        func=microduck_mdp.forward_speed_reward,
        weight=1.0,
        params={"vel_ref": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )

    # --- 3. foot_swing_height: flip to a positive reward, raise the target ---
    # The base is a penalty (drives feet to 0.02 m -> peak_height 0). Flipped, it
    # rewards feet that lift to ~0.05 m. Gated on command > 0.01 so it doesn't
    # fight the standing envs.
    cfg.rewards["foot_swing_height"].weight = 0.5
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.05
    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01

    # --- 4. leg_symmetry_reward: reward the centred mirror pose ---
    # -mean|q_left + q_right| over the leg joints; positive weight penalises
    # asymmetry. Joint-position based, so the actor (which sees joint_pos) can
    # act on it directly.
    cfg.rewards["leg_symmetry"] = RewardTermCfg(
        func=microduck_mdp.leg_symmetry_reward,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # --- 5. gait_symmetry_penalty: reward a balanced L/R stride ---
    # Penalises cumulative swing imbalance |L-R|/(L+R) over the episode. Complements
    # leg_symmetry (pose-level) with gait-level balance.
    cfg.rewards["gait_symmetry"] = RewardTermCfg(
        func=microduck_mdp.gait_symmetry_penalty,
        weight=-0.2,
        params={"sensor_name": "feet_ground_contact"},
    )

    # --- 2. track_linear_velocity std: widen (curriculum, loose -> tight) ---
    # A wider Gaussian has a live gradient at the current error, so the tracking
    # reward stops being dead. Tightens as the gait forms.
    cfg.curriculum["velocity_tracking_std"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_tracking_std_curriculum,
        params={
            "reward_name": "track_linear_velocity",
            "std_stages": [
                {"step": 0, "std": 0.5},
                {"step": 500 * NUM_STEPS_PER_ENV, "std": 0.35},
                {"step": 1000 * NUM_STEPS_PER_ENV, "std": 0.3},
            ],
        },
    )

    # --- 6. velocity-command-range curriculum (forward-only) ---
    # The base holds the command at 0.2-1.2 m/s from step 0 — a hard target for a
    # policy that can't walk. Ramp it: easy first, then up to the Arena's range.
    # forward_only=True sets lin_vel_x = (0, v) (the Arena never commands backward).
    cfg.curriculum["velocity_command_range"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_command_ranges_curriculum,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_range": 0.1, "ang_vel_range": 0.0},
                {"step": 500 * NUM_STEPS_PER_ENV, "lin_vel_range": 0.4, "ang_vel_range": 0.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "lin_vel_range": 0.8, "ang_vel_range": 0.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "lin_vel_range": 1.2, "ang_vel_range": 0.0},
            ],
            "update_lin_vel_y": False,
            "update_ang_vel_z": False,
            "forward_only": True,
        },
    )

    # --- 7. standing_envs: cap at 0.10 and delay the ramp ---
    # The base ramps to 0.25 by iter 2000, reinforcing standing right when we want
    # walking. Cap it at 0.10 and only start rising after a gait exists (~iter 500).
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0, "rel_standing_envs": 0.02},
                {"step": 500 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.05},
                {"step": 1000 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.10},
                {"step": 2000 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.10},
            ],
        },
    )

    return cfg


# --- 8. RL config: more early exploration ---
# The "first" run's entropy collapsed (19.8 -> -9.4, mean_std 1.0 -> 0.13), so the
# policy converged to one behaviour and couldn't escape the standing basin. A higher
# entropy floor keeps it exploring early. (An entropy-coef *curriculum* can't reach
# the PPO algorithm from an env curriculum — the runner owns the algorithm — so this
# is a static bump instead.)
RL_CFG = deepcopy(RL_CFG)
RL_CFG.algorithm.entropy_coef = 0.03
RL_CFG.experiment_name = "sprint_2m_walk"  # logs/rsl_rl/sprint_2m_walk/
RL_CFG.run_name = "sprint_2m_walk"
