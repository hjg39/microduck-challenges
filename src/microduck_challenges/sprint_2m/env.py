"""The 2 m Sprint's training environment: Microduck walking on flat ground.

A complete copy of microduck_rl's velocity task (microduck_velocity_env_cfg.py),
made the sprint's own: flat ground only, and the three rewards a novice tunes
read from challenge.toml ([params]). Everything here is yours to change; the
robot (MICRODUCK_WALK_ROBOT_CFG) and the reward functions (microduck_mdp)
come from mjlab_microduck and are not edited here.

The main locomotion task: velocity-command tracking + head-pose commands.
The reward/regularization recipe is locomotion-focused (lean tracking +
gait/feet terms, curriculum-ramped action-rate smoothing), with:

  - foot_slip kept at -0.1 (deliberately weak — stronger was too restrictive
    for this robot's pivot-heavy turning)
  - fixed command ranges taken straight from the Arena event's sweep
    (arena/events/sprint-2m.toml) — straight ahead only, never sideways or
    turning (see the command section below)
  - head_pose_tracking as a primary objective, plus an EMA-based head_pose_bias
    penalty that prices only the escapable DC head droop (see below); the head
    command itself is held at its small initial range (no widening curriculum)
  - body_pose tracking infra kept intact but DISABLED (weight 0) so the obs
    slot stays alive for envs that use it
  - heading_hold: the Arena commands yaw 0 and never tells the policy its
    heading, so walking straight is priced on the yaw angle; the critic
    alone sees that heading error
"""

import math
from copy import deepcopy

NUM_STEPS_PER_ENV = 24

# Symmetry
ENABLE_SYMMETRY = False

# Domain randomization toggles
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True  # Randomize CoM of the head assembly bodies
ENABLE_KP_RANDOMIZATION = False # Was True
ENABLE_KD_RANDOMIZATION = False # Was True
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # Can enable once walking is stable
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # Scales BAM's friction budget per-env via FrictionDRBamActuator.friction_scale
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_ARMATURE_RANDOMIZATION = True  # Reflected rotor inertia (microban-style). DOES affect BAM (armature is set, not zeroed).
ENABLE_VELOCITY_PUSHES = True  # Velocity-based pushes for robustness training
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # Simulates mounting errors
ENABLE_ENCODER_BIAS = True  # Per-env joint encoder calibration offset (actor obs sees joint_pos + bias)
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False  # Randomize initial tilt to force reactive behavior

# Head/body pose command tracking (replaces the old neck-offset disturbance scheme).
# Head pose: 4D deltas-from-HOME on neck/head joints; vel env tracks these as a
# primary objective. Body pose: 6D delta in [x, y, z, roll, pitch, yaw]; vel env
# samples small ranges + tiny reward weight so input neurons stay alive but
# tracking isn't the priority (standup env raises the weight).
HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)

# Observation configuration
USE_PROJECTED_GRAVITY = True  # If True, use projected gravity instead of raw accelerometer

# Domain randomization ranges (adjust as needed)
# Conservative ranges proven to be stable - can increase gradually if needed
COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initial, ramped to ±8mm via curriculum
# Head CoM randomization: applied per-episode to every body of the head assembly
# (neck → neck_pitch → yaw_roll_motion → head-roll body). Same non-accumulating
# mechanism as the trunk CoM randomization above. The head-roll body is named
# bottom_head_shell in the walk model and jaw_soft in the 2026-07 roller model,
# hence the alternation. NOTE: bearing_roll is NOT a head body — in both models
# it is the right-hip-yaw link (child of trunk_base); it has always been listed
# here by mistake and is kept only to preserve existing DR behavior.
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initial, ramped via curriculum
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)  # ±5% applied to BOTH mass and inertia together.
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # ±15%
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% (can increase to 0.8-1.2)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% reflected rotor inertia (microban: dr.joint_armature, same range)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # Apply pushes every 3-6 seconds
VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # Velocity change range in m/s. Was ±0.5 — an
# ADDITIVE kick larger than max walk speed (0.4) every 3-6 s trains a permanently
# nervous fall-recovery gait (2026-07 audit). ±0.3 keeps push robustness while
# letting a calmer gait be optimal.
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # up-to-6° random-axis IMU mounting error. NOTE: zero-centered (random axis) — trains tolerance to misalignment *magnitude*, NOT a pitch bias. The real board's systematic ~5° pitch offset is corrected at the source in the runtime (imu-pitch-offset), not here.
ENCODER_BIAS_RANGE = (-0.015, 0.015)  # ±0.86° per-joint encoder offset (constant per env)
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0  # ±10° forward/backward tilt at episode start
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0  # ±5° side-to-side tilt at episode start

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck import challenge
from mjlab_microduck.robot.microduck_constants import (
    HOME_FRAME,
    MICRODUCK_WALK_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def heading_error(env: ManagerBasedRlEnv) -> torch.Tensor:
    """How far the trunk has turned from the heading heading_hold holds (rad, wrapped).

    Critic-only: the Arena never tells the actor its heading. Reads the spawn
    heading heading_hold_reward captures (env._heading_ref); zero on an
    episode's first observation, which comes before that reward captures it.
    """
    w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    reference = torch.where(env.episode_length_buf == 0, yaw, getattr(env, "_heading_ref", yaw))
    error = yaw - reference
    return torch.atan2(torch.sin(error), torch.cos(error)).unsqueeze(-1)


def _finish_line(env: ManagerBasedRlEnv, length_m: float = 2.0) -> torch.Tensor:
    """Per-env finish-line position: spawn position + ``length_m`` along the
    robot's forward direction at spawn.

    The Arena's sprint-2m course is a 2 m lane: the robot spawns at one end
    and the finish is 2 m straight ahead. The command is always straight
    (lin_vel_y = 0, ang_vel_z = 0), so the robot's forward direction at spawn
    IS the lane direction. We anchor the finish to the spawn pose (captured on
    the first step after reset, like ``heading_hold_reward`` anchors the spawn
    heading) so it tracks the per-env spawn randomisation (±2° yaw, ±15 mm CoM).

    Returns a (N, 3) tensor of finish positions in world frame.
    """
    asset = env.scene["robot"]
    pos = asset.data.root_link_pos_w  # (N, 3)
    quat = asset.data.root_link_quat_w  # (N, 4) [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Forward direction in world frame (body +X axis).
    fwd = torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * w - y * z),
            2.0 * (x * y + w * z),
        ],
        dim=-1,
    )
    finish = pos + length_m * fwd
    if not hasattr(env, "_finish_line") or env._finish_line.shape[0] != env.num_envs:
        env._finish_line = finish.clone()
    just_reset = env.episode_length_buf <= 1
    env._finish_line = torch.where(just_reset.unsqueeze(-1), finish, env._finish_line)
    return env._finish_line


def straight_line_reward(env: ManagerBasedRlEnv, length_m: float = 2.0) -> torch.Tensor:
    """Potential-based reward for closing the distance to the finish line.

    The 2026-10-13 run's policy wandered off a straight line to the finish —
    the Arena's score is a straight-line sprint, so any lateral drift or
    backtracking costs time. This rewards the *change* in distance-to-finish
    (potential-based, so it is bias-free and does not inflate the value
    function): positive when the duck gets closer, negative when it moves
    away. The finish is anchored to the spawn pose (see ``_finish_line``).

    Calibrated (gain 2.0) so a 0.5 m/s forward sprint earns ~0.008/step —
    modest but a live, direction-specific gradient that the existing
    ``forward_speed`` term (which rewards speed regardless of direction)
    cannot provide.
    """
    finish = _finish_line(env, length_m)
    root = env.scene["robot"].data.root_link_pos_w
    dist = torch.linalg.vector_norm(root - finish, dim=-1)
    prev = getattr(env, "_finish_dist_prev", None)
    if prev is None:
        env._finish_dist_prev = dist
        return torch.zeros(env.num_envs, device=env.device)
    reward = (prev - dist) * 2.0  # closer -> positive, farther -> negative
    env._finish_dist_prev = dist
    return reward


def lateral_drift_penalty(env: ManagerBasedRlEnv, length_m: float = 2.0) -> torch.Tensor:
    """Penalise sideways (lateral) drift relative to the finish line.

    Complements ``straight_line_reward``: that term rewards *net* progress
    toward the finish, but a policy can earn it by oscillating. This term
    penalises the lateral *velocity* component (perpendicular to the
    finish-line direction), so the duck is pushed to commit to a straight
    line rather than weaving.
    """
    finish = _finish_line(env, length_m)
    root = env.scene["robot"].data.root_link_pos_w
    vel = env.scene["robot"].data.root_link_lin_vel_w
    to_finish = finish - root
    dist = torch.linalg.vector_norm(to_finish, dim=-1, keepdim=True).clamp_min(1e-6)
    # Unit vector toward finish; lateral = velocity minus its forward component.
    fwd = to_finish / dist
    forward_speed = (vel * fwd).sum(dim=-1, keepdim=True)
    lateral = vel - forward_speed * fwd
    return torch.linalg.vector_norm(lateral, dim=-1)


def head_acceleration_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalise jerky head/neck motion (squared change in joint velocity).

    The 2026-10-13 run's policy shook its head a lot. The existing
    ``neck_joint_vel_l2`` term penalises head *velocity* (how fast the head
    is moving); this adds a penalty on the *change* in head velocity per
    control step — a discrete jerk / angular-acceleration signal — so the
    duck is pushed to move its head smoothly rather than in sharp jerks.

    We use the change in joint *velocity* (Δqvel per step) rather than the
    raw MuJoCo ``qacc``: qacc is a noisy second derivative (L2 mean ~90
    rad/s² under random actions) that would dominate the reward at any
    sensible weight, whereas Δqvel is on the same scale as the existing
    ``neck_joint_vel`` term (L2 mean ~1.5, p95 ~2.7) and calibrates cleanly.

    Uses the same neck+head joint set (servo-view indices 5–8: neck_pitch,
    head_pitch, head_yaw, head_roll) as the existing neck terms.
    """
    asset = env.scene["robot"]
    # Index through the servo view (same convention as the existing neck_*
    # terms in mdp.py): the canonical 14-servo layout puts neck+head at
    # indices 5-8. On the walk robot (no passive joints) this is the identity,
    # but indexing through _servo_joint_ids keeps it correct if passive joints
    # are ever added.
    ids = microduck_mdp._servo_joint_ids(env, asset)
    joint_vel = asset.data.joint_vel[:, ids[5:9]]
    prev = getattr(env, "_head_vel_prev", None)
    if prev is None or prev.shape[0] != env.num_envs:
        env._head_vel_prev = joint_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)
    dvel = joint_vel - prev
    env._head_vel_prev = joint_vel.clone()
    return torch.sum(torch.square(dvel), dim=1)


def head_height_below(
    env: ManagerBasedRlEnv,
    min_height: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("neck_pitch",)),
) -> torch.Tensor:
    """Terminate when the HEAD centre drops below ``min_height`` (world z).

    The complement to the trunk-tilt ``fell_over`` termination. The 2026-10-13
    run's final verification fell over by BUCKING: the duck kept its trunk
    upright (so ``bad_orientation``'s 70° tilt gate never tripped) and its head
    well above the ground, repeatedly flexing to stay off the floor — a fall the
    env never saw. The head is the highest, most fall-sensitive part of the
    robot (home z ≈ 0.20 m, vs trunk 0.12 m): when it buckles to the ground the
    head centre is the first thing to drop. Terminating on head height catches
    the buckling fall that the trunk-tilt gate misses.

    ``min_height`` 0.10 m is well below the home head height (0.20 m) and below
    the trunk's own fell_over gate, so a normal gait (head bobbing a few mm)
    never trips it — only a genuine collapse does.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # Index of the head body in the entity's full body list (body_link_pos_w is
    # indexed by that order). find_bodies resolves the name robustly.
    ids, _ = asset.find_bodies("neck_pitch")
    z = torch.nan_to_num(
        asset.data.body_link_pos_w[:, ids[0], 2] - env.scene.terrain.env_origins[:, 2],
        nan=0.0,
    )
    return z < min_height


def head_low_penalty(
    env: ManagerBasedRlEnv,
    length_m: float = 2.0,
    near_height: float = 0.16,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("neck_pitch",)),
) -> torch.Tensor:
    """Penalise the head dropping toward the ground, EXCEPT at the finish line.

    Harry's call (2026-10-14): the duck is *allowed* to dip its head over the
    finish line (a sprint finish), so the penalty applies everywhere else in
    the lane. The walk3 run's policy dipped the head low through the middle of
    the lane (the ``head_height`` termination fired on 50-75% of episodes by
    the end of the run) — that is what this prices.

    Two gates, multiplied:
      1. height ramp: 0 at ``near_height`` (home head z ≈ 0.20 m, so normal
         gait bobbing never enters it) rising to 1 at the ``head_height``
         termination threshold (0.10 m);
      2. position gate: 1 everywhere EXCEPT within the last 0.5 m of the
         anchored finish line, where it is 0 (the dip is free there).

    Reuses the same head body (``neck_pitch`` centre) and finish-line anchor
    as the existing terms, so it stays consistent with the termination.
    """
    asset: Entity = env.scene[asset_cfg.name]
    ids, _ = asset.find_bodies("neck_pitch")
    z = torch.nan_to_num(
        asset.data.body_link_pos_w[:, ids[0], 2] - env.scene.terrain.env_origins[:, 2],
        nan=0.0,
    )
    # 0 at near_height, 1 at the termination threshold (0.10 m), clamped.
    ramp = ((near_height - z) / (near_height - 0.10)).clamp(0.0, 1.0)
    # Everywhere EXCEPT the last 0.5 m of the lane (the finish-line dip is free).
    finish = _finish_line(env, length_m)
    root = env.scene["robot"].data.root_link_pos_w
    dist = torch.linalg.vector_norm(root - finish, dim=-1)
    return ramp * (dist >= 0.5).float()


def neck_pitch_tracking(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise the neck pitch deviating from its HOME angle.

    The 2026-10-13 walk3 pose measurement (metal/measure_pose.py on
    model_5998.pt, vx=0.8) showed the neck sitting 5.7° BELOW home while the
    head sat 2.3° above — the policy extended the head to counterbalance a
    neck that the existing terms never anchored. The base env's
    ``head_pose_tracking`` term tracks the 4D head-pose *command* (sampled
    ±0.05 rad around home), which lets the neck drift as long as the head
    compensates. This term pins the neck itself to home, so the head can
    relax back to home too.

    Uses the same servo-view indexing as ``head_acceleration_penalty``.
    """
    asset = env.scene[asset_cfg.name]
    ids = microduck_mdp._servo_joint_ids(env, asset)
    q = asset.data.joint_pos[:, ids[5]]  # neck_pitch
    home = HOME_FRAME.joint_pos[r".*neck_pitch.*"]
    return torch.square(q - home)


def _make_base_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """The sprint's base training environment (pre reward re-shaping).

    This is the original recipe, kept intact so the reward re-shaping in
    ``make_env_cfg`` below reads as a clean diff on top of it.
    """

    # The knobs a novice tunes, from challenge.toml beside this file.
    knobs = challenge.params(__file__)

    std_standing = {
        # Lower body — tighter to keep the robot in home pose when standing
        r".*hip_yaw.*": 0.1,
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — hold the 5°-inward stance (sole sits flat), stop leg splay
        r".*hip_pitch.*": 0.15,
        r".*knee.*": 0.15,
        r".*ankle.*": 0.1,
    }

    std_walking = {
        # Lower body
        # 2026-10-14 (walk3 pose measurement): the policy cants both hips to a
        # fixed ±25.3° (0.443 rad, near-zero std) — at std 0.3 the Gaussian is
        # exp(-0.443²/0.09) ≈ 0.15, nearly dead, so the canted stance was never
        # priced. Tighten to 0.15 (exp(-0.443²/0.0225) ≈ 0.013 → live gradient)
        # so the pose reward pulls the stance back to the centred mirror pose.
        r".*hip_yaw.*": 0.15,
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — hold the 5°-inward stance, stop the leg splay to vertical
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25, # was 0.15
    }

    site_names = ["left_foot", "right_foot"]

    # Contact sensor for feet - LEFT, RIGHT order
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",  # LEFT foot first, RIGHT foot second
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # mjlab 1.3.0: foot_height obs + foot_clearance/foot_swing_height rewards are
    # now driven by a per-foot terrain-height ray sensor (was site_pos based).
    # Mirrors microban's foot_height_scan.
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    # Base configuration
    cfg = make_velocity_env_cfg()

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === REWARDS ===
    # Pose reward configuration
    cfg.rewards["pose"].params["std_standing"] = std_standing  # tight when command=0
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    # Pose reward operates on LEG joints only. Head/neck are command-driven
    # (head_pose_tracking) — if they were in this reward too, it would pull
    # them to HOME while head_pose_tracking pulls them to the command, and the
    # policy converges to "ignore the command" because pose reward dominates
    # once head_pose_tracking's gradient dies at large commands.
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0

    # Body-specific reward configurations
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    # upright: deliberately strong (2.0 / std²=0.05, was 1.0 / std²=0.1).
    # 2026-07 pitch-vs-speed eval: the policy walks with a +2-4° steady forward
    # lean (p90 ~6-8°) and ~2/3 of push-induced falls at speed are FORWARD. At
    # weight 1.0 / std²=0.1 a 4° lean cost ~0.05/step — effectively free. At
    # 2.0 / std²=0.05 it costs ~0.19/step: enough gradient to hold the trunk
    # level in steady gait while transient lean (push recovery, accel) stays
    # affordable.
    cfg.rewards["upright"].weight = knobs["upright_reward"]
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # Foot-specific configurations. In mjlab 1.3.0 foot_swing_height is fully
    # sensor-driven (no asset_cfg); only foot_clearance/foot_slip still carry an
    # asset_cfg whose site_names select the feet.
    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # Body-specific configurations
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    # foot_slip deliberately weak (-0.1, not -1.0): -1.0 was too restrictive
    # for this robot's pivot-heavy turning.
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01

    cfg.rewards.pop("soft_landing", None)

    # Self-collision penalty: discourages legs from crashing into the trunk
    # battery holder (the self_collision_only-classed geoms on leg, leg_2,
    # battery_holder). With proper joint-range limits the policy can't actually
    # reach the body, but a positive signal here keeps it well clear.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )


    # air_time window. NOTE: standing still at zero command is taught by the
    # standing_envs curriculum, not by an explicit stillness/no-stepping term.
    # 2026-10-13 re-balance (Harry: "big movements with pauses should also be
    # ok"): the old window [0.125, 0.300] s only rewarded CONSTANT small steps —
    # a big stride (0.4 s air) + a pause scored ZERO, so the policy converged to
    # a low hover/shuffle (air_time_mean 7 s but peak foot height ~0: feet barely
    # left the ground, never in the window). Widen to [0.10, 0.50] s so a big
    # step AND a pause both earn reward; the gait can be either cadence.
    cfg.rewards["air_time"].weight = knobs["step_reward"]
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.10
    cfg.rewards["air_time"].params["threshold_max"] = 0.50

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # Velocity tracking rewards
    cfg.rewards["track_linear_velocity"].weight = knobs["forward_reward"]
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    # Walk straight. The Arena commands yaw 0 and never tells the policy its
    # heading, and nothing above prices a slow drift: at this std a 0.07 rad/s
    # turn keeps 99% of track_angular_velocity. Unpriced, the first reference
    # turned right 3–8°/s from 0.35 m/s on the Arena and walked into the
    # lane's right wall. heading_hold prices the yaw ANGLE against the spawn
    # heading (the roller's term and std): the policy cannot see its heading,
    # but it can learn to null its gait's mean yaw rate. The critic sees the
    # heading error (heading_error, below) so it can value what this term pays.
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=1.0,
        params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Action smoothness: stage-0 value; the action_rate_weight curriculum below
    # ramps it -0.1 → -1.0 by iter 1500.
    cfg.rewards["action_rate_l2"].weight = -0.1

    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02  # Increased from 0.01 to penalize dragging

    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02  # Increased from 0.01 to force foot lifting

    # NOTE: no neck-only action-rate term — the shared action_rate_l2 sums over
    # ALL action dims (neck included), and head_pose_tracking below gives the
    # 4 neck/head DOFs a position objective, so the neck is fully shaped.

    # Events
    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # Grippier footpad — narrowed from (0.3, 1.2)
    # Terminate environments that have gone numerically unstable (NaN physics).
    # MuJoCo can produce NaN joint positions on extreme contact impulses.
    # Terminating immediately resets to a valid state before NaN propagates
    # into the observation buffer and corrupts network weights.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )

    # Head-height fall detection (2026-10-13). The base ``fell_over`` gate is a
    # 70° trunk-tilt check (bad_orientation, limit_angle 1.22 rad). The run's
    # final verification fell over by BUCKING — trunk stayed upright, head kept
    # off the floor — so that gate never tripped and the episode ran on. The
    # head centre (home z ≈ 0.20 m) is the first thing to drop when the duck
    # buckles, so terminate when it falls below 0.10 m. This is the complement
    # to the tilt gate: it catches the buckling fall the tilt gate misses.
    cfg.terminations["head_height"] = TerminationTermCfg(
        func=head_height_below,
        time_out=False,
        params={"min_height": 0.10},
    )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    # Velocity-based pushes for robustness training
    if ENABLE_VELOCITY_PUSHES:
        # In play mode, use shorter interval for better visibility
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S

        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # Domain randomization — re-sampled per episode at reset. In mjlab 1.3.0 the
    # stock dr.* ops with operation="add"/"scale" read from the compile-time
    # default field each reset (Operation.uses_defaults=True), so they are
    # NON-accumulating natively — this upstream behavior replaces microduck's old
    # custom restore-then-add functions that worked around the accumulation footgun.
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        # Randomize the CoM of the head assembly bodies (per-body fresh offset each reset).
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        # Randomize motor PD gains
        # Uses custom function that handles DelayedActuator
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        # Physics-consistent mass + inertia randomization via mjlab's pseudo_inertia:
        # alpha scales BOTH mass and inertia by e^(2*alpha) with the CoM unchanged
        # (so it does NOT conflict with randomize_com). alpha_range is derived from
        # the ±5% mass scale range: e^(2*alpha) ∈ [0.95, 1.05].
        # Replaces the old custom randomize_mass_and_inertia, which was a silent
        # no-op under mjlab 1.3.0 (direct per-env body_mass/body_inertia writes are
        # not expanded and collapse to a single shared value). Startup mode = fixed
        # per env for the whole run (standard for mass DR; no accumulation).
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        # Joint-friction DR under BAM: scales BAM's velocity-independent friction
        # budget (Coulomb + Stribeck + load) per-env via the FrictionDRBamActuator
        # friction_scale hook. MuJoCo's dof_frictionloss is zeroed under BAM, so the
        # stock dr.dof_frictionloss is a no-op — this is the BAM-native path.
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        # Randomize joint damping (lubrication, temperature effects).
        # Custom non-accumulating scaler. NOTE: no-op under BAM (dof_damping
        # zeroed in edit_spec); only affects the XML position actuator.
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "field": "dof_damping",  # required by domain_randomization=True
                "scale_range": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        # Randomize reflected rotor inertia (armature), microban-exact
        # (dr.joint_armature, scale, ±10%). Non-accumulating (uses_defaults). DOES
        # affect the BAM actuator — BAM sets dof_armature (~0.0018), it isn't zeroed.
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # IMU orientation randomization (mounting error) is applied at the OBSERVATION
    # level below (per-env constant rotation of projected_gravity + base_ang_vel).
    # The old event-based randomize_imu_orientation wrote site_quat, which under
    # mjlab 1.3.0 is neither per-env expanded nor read by these obs — a no-op.

    # Base orientation randomization (forces reactive behavior)
    if ENABLE_BASE_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_base_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_base_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_pitch_deg": BASE_ORIENTATION_MAX_PITCH_DEG,
                "max_roll_deg": BASE_ORIENTATION_MAX_ROLL_DEG,
            },
        )

    # Observations
    del cfg.observations["actor"].terms["base_lin_vel"]
    # mjlab 1.3.0 adds a height_scan term (terrain ray scan) to both groups by
    # default. The microduck has no such body-mounted terrain sensor for the
    # policy, so drop it from both (mirrors microban).
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Add base_lin_vel to critic only (privileged information)
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )
    # Privileged too: the heading error heading_hold prices.
    cfg.observations["critic"].terms["heading_error"] = ObservationTermCfg(func=heading_error)

    # Determine gravity/accelerometer term name based on flag
    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    # Replace projected_gravity with raw_accelerometer if flag is False
    if not USE_PROJECTED_GRAVITY:
        # Remove projected_gravity and add raw_accelerometer
        del cfg.observations["actor"].terms["projected_gravity"]
        cfg.observations["actor"].terms["raw_accelerometer"] = ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        )

    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # The critic's sensor-derived terms are the one obs path `nan_state` cannot
    # protect (it checks joint + root state; these read raycast/contact sensor
    # data, which MuJoCo can return non-finite for while the state is still
    # clean). A single NaN here kills the whole run via rsl_rl's check_nan —
    # that is the 2026-08-21 Velocity2-Rough-Backlash crash. Critic-only, so
    # sanitizing costs the policy nothing.
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    # Observation noise configuration (edit these values as needed)
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03) # was 0.2
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)  # was 0.15
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)  # was 0.05
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)  # was 2.0

    # IMU mounting-misalignment DR (per-env constant rotation of the IMU-derived
    # observations). Applied to the ACTOR only (the policy sees a slightly rotated
    # IMU frame, like a real mounting error); the critic keeps the true values.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        if USE_PROJECTED_GRAVITY:
            g = cfg.observations["actor"].terms[gravity_term_name]
            g.func = microduck_mdp.projected_gravity_imu_misaligned
            g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel: the Dynamixel firmware computes
    # present_velocity via a moving-average over the previous position-sample
    # window, so the value the policy actually reads is ~1 control period old.
    # Matches reality and stops the policy relying on instantaneous qdot feedback.
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Exclude passive_* joints (jaw linkage) from joint_pos/vel obs so the
    # observation dim matches the action dim (14) instead of the raw articulation (16).
    # Deepcopy each joint_pos/joint_vel term first — actor and critic share the
    # same term objects/params dicts from the base template, so mutating one would
    # leak into the other (e.g. the encoder-bias `biased` flag below).
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR: the base template samples a per-env constant joint-encoder
    # offset (startup event "encoder_bias"), but joint_pos_rel ignores it unless
    # biased=True. Feed the biased joint pos to the ACTOR only (what the real
    # encoders report); the critic keeps the true joint pos (privileged).
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Commands — deepcopy to avoid shared-state corruption from other env cfgs
    # (make_velocity_env_cfg() returns objects with shared mutable references;
    # standup/ground_pick envs mutate commands["twist"] in place, zeroing ranges)
    command: UniformVelocityCommandCfg = deepcopy(cfg.commands["twist"])
    cfg.commands["twist"] = command
    command.rel_standing_envs = 0.02  # small but non-zero from the start, ramped up by curriculum
    command.rel_heading_envs = 0.0
    # The ranges are the Arena event's own (arena/events/sprint-2m.toml):
    # straight ahead, the event's sweep of commanded speeds, never sideways or
    # turning. History: the source task trained ±0.4 m/s, and the Arena's
    # reference walker then peaked at exactly 0.40 m/s — the top of that range.
    command.ranges.lin_vel_x = (0.20, 1.20)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # Head pose command (4D deltas from HOME, in joint order:
    #   neck_pitch, head_pitch, head_yaw, head_roll). Tracked as a primary
    # reward — see "head_pose_tracking" added below. Ranges are small non-zero
    # (not zero) so the input neurons stay alive, and held there for the whole
    # run — the Arena commands no head motion, so there is no widening
    # curriculum here (the head stays still).
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll (tighter — much smaller mechanical range)
        ),
    )
    # Body pose command (6D delta from nominal standing: [x, y, z, roll, pitch, yaw]).
    # Vel env carries this slot for runtime obs-shape parity; tracked at a tiny
    # weight to keep the input neurons alive but not steer the policy. The
    # standup env raises the weight + widens the ranges.
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.005, 0.005),  # x (m)
            (-0.005, 0.005),  # y (m)
            (-0.005, 0.005),  # z (m)
            (-0.05, 0.05),    # roll (rad)
            (-0.05, 0.05),    # pitch (rad)
            (-0.05, 0.05),    # yaw (rad)
        ),
    )

    # Append head + body command obs terms to both policy and critic groups.
    # Order matters for the runtime obs layout: [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

    # === Pose tracking rewards ===
    # head_pose: primary objective in vel env — the whole point of the rewrite.
    # std=0.5 with per-joint Gaussian (see head_pose_tracking in mdp.py). Final
    # reward is the mean over 4 joints, so partial tracking is partial reward
    # (no all-or-nothing).
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )
    # body_pose: infra kept intact but DISABLED (weight 0) — the obs slot and
    # command stay alive for envs that raise the weight (standup).
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15),
        },
    )

    # Head droop fix (2026-08-20). The head walks pitched ~15° down (measured:
    # run ww1g2198 head_pose_tracking 1.544/2.0 → 14.6° mean joint error).
    # DO NOT fix this by tightening head_pose_tracking's std: run 5yay13u4 tried
    # fine_std=0.1 and the policy stopped walking entirely by iter 300 (air_time
    # 1.01 → 0.02, peak foot height 15 mm → 2 mm, entropy collapsed 10.9 → 1.9).
    # An instantaneous tight tolerance taxes walking 0.77/step — 76% of the whole
    # air_time reward — and is UNESCAPABLE, since a 280 g head (38% of robot
    # mass) must oscillate while stepping. Standing still scored higher, so it
    # stood still.
    # The DC bias, unlike the oscillation, IS escapable (bias the neck command up
    # to cancel gravity sag), so price only that: L1 on a 1 s EMA of the error.
    # At the optimum this costs a walking policy nothing.
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # ramped by the head_pose_bias_weight curriculum below
        params={"command_name": "head_pose", "tau_s": 1.0},
    )

    # Terrain: the sprint's lane is flat.
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # action_rate weight ramp: gentle smoothing while the gait bootstraps, then
    # tighten. 2026-10-13 re-balance: the old ramp hit -1.0 by iter 1500, making
    # the leg motion needed to WALK the single largest term in the budget
    # (-0.003/step, ~10x the forward terms) — so the duck shuffles in place
    # (cheap) instead of striding (expensive). Cap at -0.5 and slow the ramp:
    # keep the anti-jerk smoothing, but stop pricing a big stride 10x a small one.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.3},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -0.4},
                {"step": 2000 * NUM_STEPS_PER_ENV, "weight": -0.5},
            ],
        },
    )

    # Gradually increase standing env fraction after walking is established
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0,           "rel_standing_envs": 0.02},
                {"step": 500 * 24,    "rel_standing_envs": 0.05},
                {"step": 750 * 24,    "rel_standing_envs": 0.1},
                {"step": 1000 * 24,   "rel_standing_envs": 0.15},
                {"step": 1500 * 24,   "rel_standing_envs": 0.2},
                {"step": 2000 * 24,   "rel_standing_envs": 0.25},
            ],
        },
    )

    # NOTE: no velocity-command-range curriculum — ranges are fixed (see the
    # command section above). Same for head_pose: no range curriculum — the
    # head stays at its small initial range for the whole run (see the
    # head_pose command section above).

    # Body pose command range curriculum: stay small in vel env. Standup env
    # overrides this curriculum with wide ranges + heavy reward weight.
    cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {"step": 0, "ranges": (
                    (-0.005, 0.005),  # x (m)
                    (-0.005, 0.005),  # y (m)
                    (-0.005, 0.005),  # z (m)
                    (-0.05, 0.05),    # roll
                    (-0.05, 0.05),    # pitch
                    (-0.05, 0.05),    # yaw
                )},
            ],
        },
    )

    # CoM randomization range curriculum - start small, ramp up
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    # Capped at ±15 mm (2026-07 audit): the previous ramp to ±30 mm
                    # exceeded the foot support polygon (heel is only 20 mm behind
                    # the ankle) — the randomized CoM could sit entirely outside
                    # support, forcing a wide/fast hyper-reactive gait and making
                    # BACKWARD balance untrainable. Regression timeline matched the
                    # ramp increases: 0.015 → 0.02 → 0.03 as policies got worse.
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                    {"step": 1500 * 24,  "range": 0.015},
                ],
            },
        )

    # Head CoM randomization range curriculum - start small, ramp up
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    # Capped at ±10 mm (2026-07 audit — same over-conservatism
                    # concern as trunk CoM; head is a large lever arm).
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                ],
            },
        )

    # Disable default curriculum
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # head_pose_bias ramp: OFF until iter 600, then 1.0 → 3.0 by iter 1500.
    # Held at 0 early because a posture-precision term is a distraction before
    # a gait exists. At weight 3.0 a 15° residual bias costs 0.79/step and a
    # 2° bias costs 0.10/step.
    cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 600 * NUM_STEPS_PER_ENV, "weight": 1.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 2.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 3.0},
            ],
        },
    )

    return cfg


def make_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The sprint's training environment: base recipe + the reward re-shaping.

    The "first" run (this env, pre re-shaping) plateaued at ~0 m/s forward: the
    policy learned to stand upright and shuffle its feet (satisfying
    upright/pose/heading/head_pose) while the forward-velocity reward — a tight
    Gaussian — was dead (error ~1.4 m/s vs std 0.316 -> reward ~0, no gradient).
    Standing scored higher, so it stood still, and the entropy collapsed
    (19.8 -> -9.4) so it couldn't escape.

    This keeps the base env (``_make_base_env_cfg``) and layers on the fixes:

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
    cfg = _make_base_env_cfg(play=play)

    # --- 1. forward_speed_reward: monotonic forward gradient (the key new term) ---
    # tanh(clamp(vx, 0)/vel_ref): rewards ANY forward motion, never penalises
    # standing (vx=0 -> 0). Command-independent, so it gives a live gradient even
    # when the tracking Gaussian is dead.
    # 2026-10-13 re-balance (the "not moving" local minimum, again): weight 1.0
    # was outvoted ~10:1 by the standing terms (upright+head+heading+track_ang),
    # and vel_ref 0.3 saturated below the Arena's 0.2-1.2 m/s command range.
    # weight 3.0 + vel_ref 0.5 gives a strong, NON-saturating forward gradient
    # across the whole command range (tanh(1.2/0.5)=0.98) so it bootstraps the
    # duck out of the standing basin and keeps pushing it faster.
    cfg.rewards["forward_speed"] = RewardTermCfg(
        func=microduck_mdp.forward_speed_reward,
        weight=3.0,
        params={"vel_ref": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )

    # --- Straight-line-toward-finish reward (2026-10-13) ---
    # The 2026-10-13 run's policy wandered off a straight line to the finish.
    # forward_speed rewards speed regardless of direction; this term rewards
    # *progress toward the finish* (potential-based: change in distance-to-
    # finish, anchored to the spawn pose). Positive when the duck closes the
    # gap, negative when it drifts or backtracks. Gain 2.0 → a 0.5 m/s forward
    # sprint earns ~0.008/step: modest, so it refines direction without
    # outvoting the forward/standing terms.
    cfg.rewards["straight_line"] = RewardTermCfg(
        func=straight_line_reward,
        weight=1.0,
        params={"length_m": 2.0},
    )

    # --- Lateral-drift penalty (2026-10-13) ---
    # Complements straight_line: penalises the sideways velocity component
    # (perpendicular to the finish-line direction) so the duck commits to a
    # straight line rather than weaving. A 0.1 m/s lateral drift costs
    # ~0.005/step at weight -0.5.
    cfg.rewards["lateral_drift"] = RewardTermCfg(
        func=lateral_drift_penalty,
        weight=-0.5,
        params={"length_m": 2.0},
    )

    # --- Head-acceleration penalty (2026-10-13) ---
    # The run's policy shook its head a lot. The existing neck terms penalise
    # head *velocity* (neck_joint_vel_l2) and *action* rate; this adds a
    # penalty on head *angular acceleration* (rad/s², L2) so the duck moves
    # its head smoothly rather than in sharp jerks. Weight -0.1: a 1 rad/s²
    # jerk across the 4 neck/head joints costs ~0.0004/step — a gentle nudge
    # that won't fight the head_pose_tracking objective.
    cfg.rewards["head_acceleration"] = RewardTermCfg(
        func=head_acceleration_penalty,
        weight=-0.1,
    )

    # --- Head-low penalty (2026-10-14, walk3) ---
    # The walk3 run's policy dipped the head low through the MIDDLE of the
    # lane — the head_height termination fired on 50-75% of episodes by the
    # end of the run (0.0 at resume). Harry's call: the duck is ALLOWED to
    # dip over the finish line (a sprint finish), so the penalty applies
    # everywhere EXCEPT the last 0.5 m of the lane. Height ramp: 0 at 0.16 m
    # (home head z ≈ 0.20 m, so normal gait bobbing never enters it) to 1 at
    # the 0.10 m termination threshold. Weight -1.0: a head at 0.13 m
    # mid-lane costs ~0.5/step — a strong signal that keeps the head up
    # through the sprint without fighting the finish-line dip.
    cfg.rewards["head_low"] = RewardTermCfg(
        func=head_low_penalty,
        weight=-1.0,
        params={"length_m": 2.0, "near_height": 0.16},
    )

    # --- Neck-pitch tracking (2026-10-14, walk3) ---
    # The pose measurement (metal/measure_pose.py, model_5998.pt, vx=0.8)
    # showed the neck 5.7° BELOW home while the head sat 2.3° above — the
    # policy extended the head to counterbalance a neck the existing terms
    # never anchored (head_pose_tracking tracks the 4D *command*, which lets
    # the neck drift as long as the head compensates). Pin the neck to home:
    # squared deviation, weight -0.5. At the measured 0.099 rad deviation
    # that costs ~0.005/step — a steady pull back to home without fighting
    # the head_pose_tracking objective (which still owns the head itself).
    cfg.rewards["neck_pitch_tracking"] = RewardTermCfg(
        func=neck_pitch_tracking,
        weight=-0.5,
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
    # 2026-10-13 re-balance: the old ramp tightened 0.5 -> 0.3 by iter 1000, but
    # the duck was still standing (error ~1.3 m/s) — at std 0.3 the Gaussian is
    # exp(-1.3^2/0.09) ~ 0, i.e. DEAD, so the forward tracking reward collapsed
    # 1.26 -> 0.56 over the run and standing won. Hold std at 0.5 (a live
    # gradient at error 1.3: exp(-1.3^2/0.25) ~ 0.27) until a gait actually
    # exists, then tighten gently. forward_speed (above) carries the bootstrap;
    # this term only refines once the duck is moving.
    cfg.curriculum["velocity_tracking_std"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_tracking_std_curriculum,
        params={
            "reward_name": "track_linear_velocity",
            "std_stages": [
                {"step": 0, "std": 0.5},
                {"step": 1000 * NUM_STEPS_PER_ENV, "std": 0.4},
                {"step": 2000 * NUM_STEPS_PER_ENV, "std": 0.3},
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


RL_CFG = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.01 -> 0.03 (reward re-shaping fix #8): the "first" run's entropy
        # collapsed (19.8 -> -9.4, mean_std 1.0 -> 0.13), so the policy converged
        # to one behaviour and couldn't escape the standing basin. A higher
        # entropy floor keeps it exploring early. (An entropy-coef *curriculum*
        # can't reach the PPO algorithm from an env curriculum — the runner owns
        # the algorithm — so this is a static bump instead.)
        entropy_coef=0.03,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="sprint_2m",  # logs/rsl_rl/sprint_2m/
    run_name="sprint_2m",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=50_000,
)
