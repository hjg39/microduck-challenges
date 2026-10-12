"""Registers the walking variant's task with mjlab, and its contract (challenge.toml) with publish."""

from mjlab.tasks.registry import register_mjlab_task

from mjlab_microduck import challenge
from mjlab_microduck.tasks import MicroduckOnPolicyRunner

from .env import RL_CFG, make_env_cfg

register_mjlab_task(
    task_id=challenge.register(__file__).task,
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=RL_CFG,
    runner_cls=MicroduckOnPolicyRunner,
)
