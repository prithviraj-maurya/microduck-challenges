"""The sprint's own reward terms (hypothesis 4, see hypothesis/hyp_4_run_recipe.md).

The ideas and their constants come from heyskylark's public experiment log
(github.com/heyskylark/microduck-challenges, sprint_2m/EXPERIMENTS.md), which
measured each one on the Arena evaluator. The terms are written here against
the same definitions.

Two helpers are shared by every term:

- lane speed: the trunk's world velocity along the heading it spawned with.
  That is the progress the race clock measures. A duck walking crabwise moves
  fast along its own nose and slowly down the lane; this counts the lane.
- speed gate: 0 below `lo` m/s down the lane, 1 above `hi`, smooth between.
  A balance term is relaxed by the gate, so it only loosens once the duck is
  actually running, never while it stands or walks slowly.
"""

from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab_microduck.tasks import mdp as microduck_mdp

_ROBOT = SceneEntityCfg("robot")


def _yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _lane_heading(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Each env's spawn heading (rad), captured on the first step after a reset."""
    yaw = _yaw(env.scene[_ROBOT.name].data.root_link_quat_w)
    if not hasattr(env, "_lane_heading") or env._lane_heading.shape[0] != env.num_envs:
        env._lane_heading = yaw.clone()
    env._lane_heading = torch.where(env.episode_length_buf <= 1, yaw, env._lane_heading)
    return env._lane_heading


def _lane_velocity(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Trunk velocity (along, across) the spawn heading, m/s."""
    heading = _lane_heading(env)
    vel = env.scene[_ROBOT.name].data.root_link_lin_vel_w[:, :2]
    c, s = torch.cos(heading), torch.sin(heading)
    return vel[:, 0] * c + vel[:, 1] * s, -vel[:, 0] * s + vel[:, 1] * c


def _walking(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
    """True for envs commanded to move (a standing env has a ~zero command)."""
    command = env.command_manager.get_command(command_name)
    return torch.norm(command[:, :2], dim=1) > threshold


def speed_gate(env: ManagerBasedRlEnv, lo: float, hi: float) -> torch.Tensor:
    """0 below `lo` m/s down the lane, 1 above `hi`, smoothstep between."""
    along, _ = _lane_velocity(env)
    t = torch.clamp((along - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lane_progress(
    env: ManagerBasedRlEnv,
    v_ref: float,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """clip(lane speed, 0, v_ref) / v_ref: speed paid for itself, whatever the command.

    The tracking bell curve pays ~0 to a duck far below its command, so it feels
    no pull to speed up. This term is a straight line: every extra cm/s pays.
    Logs Metrics/lane_speed_mean, the plain speed in m/s: the curve to watch.
    """
    along, _ = _lane_velocity(env)
    walking = _walking(env, command_name, command_threshold)
    if walking.any():
        env.extras["log"]["Metrics/lane_speed_mean"] = along[walking].mean()
    return torch.clamp(along, 0.0, v_ref) / v_ref * walking.float()


def lane_drift(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """exp(-v_across² / std²): prices sliding across the lane, by yawing or crabbing.

    Training has no walls; the Arena's are 0.4 m either side. Fast gaits that
    drift reach them before the line. Logs Metrics/lane_drift_abs_mean (m/s).
    """
    _, across = _lane_velocity(env)
    walking = _walking(env, command_name, command_threshold)
    if walking.any():
        env.extras["log"]["Metrics/lane_drift_abs_mean"] = across[walking].abs().mean()
    return torch.exp(-torch.square(across) / std**2) * walking.float()


def track_linear_velocity_at_least(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Velocity tracking that charges only for being SLOWER than the forward command.

    exp(-(max(cmd_x - v_x, 0)² + (cmd_y - v_y)² + v_z²) / std²). The stock bell
    curve also charges for going faster than the command, which caps the duck at
    the top command (1.2 m/s). A standing env still tracks v_x = 0 both ways.
    """
    command = env.command_manager.get_command(command_name)
    v = env.scene[_ROBOT.name].data.root_link_lin_vel_b
    short = command[:, 0] - v[:, 0]
    err_x = torch.where(_walking(env, command_name, command_threshold), torch.clamp(short, min=0.0), short)
    err = torch.square(err_x) + torch.square(command[:, 1] - v[:, 1]) + torch.square(v[:, 2])
    return torch.exp(-err / std**2)


def upright_lean(
    env: ManagerBasedRlEnv,
    std: float,
    lean_std: float,
    speed_lo: float,
    speed_hi: float,
    lean_target_deg: float = 0.0,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """`upright` that lets, then asks, the trunk lean forward as the duck speeds up.

    g is gravity seen from the trunk (nose-down gives g_x > 0). Roll (g_y) always
    keeps the tight `std`. Forward, the target moves from 0 to `lean_target_deg`
    with the speed gate: short of the target costs at `std`, past it at a
    tolerance that widens from `std` to `lean_std` with the gate.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    g = quat_apply_inverse(quat, asset.data.gravity_vec_w.expand(quat.shape[0], 3))
    gate = speed_gate(env, speed_lo, speed_hi)
    e_x = g[:, 0] - gate * math.sin(math.radians(lean_target_deg))
    past, short = torch.clamp(e_x, min=0.0), torch.clamp(e_x, max=0.0)
    var_past = std**2 + gate * (lean_std**2 - std**2)
    return torch.exp(-(torch.square(g[:, 1]) + torch.square(short)) / std**2 - torch.square(past) / var_past)


def head_pose_tracking_at_speed(
    env: ManagerBasedRlEnv,
    std: float,
    fast_std: float,
    fast_offset: tuple[float, float, float, float],
    speed_lo: float,
    speed_hi: float,
    command_name: str = "head_pose",
) -> torch.Tensor:
    """head_pose_tracking whose target moves by `fast_offset` as the duck speeds up.

    The head is 38% of the duck's mass. Standing, it tracks the command as before.
    Running, it is pulled (tolerance `fast_std`) to the command plus `fast_offset`
    (neck_pitch, head_pitch, head_yaw, head_roll, rad): carried forward, as a
    counterweight. The two are blended by the speed gate.
    """
    gate = speed_gate(env, speed_lo, speed_hi)
    slow = microduck_mdp.head_pose_tracking(env, command_name=command_name, std=std)
    # Measured as microduck_mdp.head_pose_tracking measures it (its call above cached the ids).
    asset = env.scene[_ROBOT.name]
    ids = env._head_pose_neck_ids
    q = asset.data.joint_pos
    measured = q[:, ids] + q[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    target = env.command_manager.get_command(command_name) + torch.tensor(fast_offset, device=env.device)
    err = measured - asset.data.default_joint_pos[:, ids] - target
    fast = torch.exp(-torch.square(err / fast_std)).mean(dim=-1)
    return (1.0 - gate) * slow + gate * fast


def head_pose_bias_at_rest(
    env: ManagerBasedRlEnv,
    speed_lo: float,
    speed_hi: float,
    command_name: str = "head_pose",
    tau_s: float = 1.0,
) -> torch.Tensor:
    """head_pose_bias_penalty faded out with speed, so a running duck may hold its head off the command."""
    bias = microduck_mdp.head_pose_bias_penalty(env, command_name=command_name, tau_s=tau_s)
    return bias * (1.0 - speed_gate(env, speed_lo, speed_hi))
