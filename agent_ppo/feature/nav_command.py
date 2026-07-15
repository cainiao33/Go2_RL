# -*- coding: UTF-8 -*-
"""Lightweight goal/navigation command injection for Track fine-tuning.

This module intentionally keeps the teammate policy ABI unchanged:
- policy obs remains 301 in TrackConfig;
- actor/critic MLP shape remains compatible with the Standard checkpoint;
- navigation is expressed by overwriting the existing velocity command slice
  obs[:, 6:9] and the command_manager base_velocity tensor.

Why this path: a fully trained Standard locomotion policy already knows how to
walk when given (vx, vy, wz).  For a last-minute Track run, it is safer to turn
navigation into a command-generation layer than to replace the whole model with
an untrained 512-dim architecture.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch

from agent_ppo.feature.feature_layout import NavFeatureLayout
from agent_ppo.feature.nav_signal import NavMeterConfig
from agent_ppo.feature.track_tensor_bridge import TrackTensorBridge


_POLICY_CMD_SLICE = slice(6, 9)  # default policy obs: velocity_commands(vx, vy, wz)
_CRITIC_CMD_SLICE = slice(9, 12)  # default critic obs: velocity_commands(vx, vy, wz)


class _Once:
    warned_no_goal = False
    warned_no_robot = False
    warned_no_nav_sensor = False
    warned_cmd_write = False
    warned_nav_error = False


class _NavCommandRequiredSignalError(RuntimeError):
    pass


def _is_observation_shape_probe() -> bool:
    """Return True when ObservationManager is only probing an obs term shape.

    During ObservationManager.__init__ / _prepare_terms, Track task tensors such
    as env.goal_positions may not be created yet. Since command injection keeps
    the obs shape unchanged, this probing call must not trigger fail-fast.
    """
    try:
        import inspect

        frame = inspect.currentframe()
        depth = 0
        while frame is not None and depth < 48:
            name = frame.f_code.co_name
            filename = frame.f_code.co_filename.replace("\\", "/")
            if name == "_prepare_terms" and "observation_manager.py" in filename:
                return True
            frame = frame.f_back
            depth += 1
    except Exception:
        return False

    return False


def apply_track_nav_command_to_policy_obs(obs: torch.Tensor, env: Any, stage: Any = None, logger: Any = None) -> torch.Tensor:
    """Overwrite policy velocity command with a goal-directed command in Track mode.

    Returns a tensor with the same shape as ``obs``.  If required Track signals
    are not available, it returns ``obs`` unchanged instead of crashing Standard.
    """
    if obs is None or not torch.is_tensor(obs) or obs.shape[-1] < 9:
        return obs
    if stage is not None and not bool(getattr(stage, "enable_track_nav_command", False)):
        return obs

    cmd = compute_track_nav_command(env, obs, stage=stage, logger=logger)
    if cmd is None:
        return obs

    out = obs.clone()
    out[..., _POLICY_CMD_SLICE] = cmd.to(device=out.device, dtype=out.dtype)
    _try_write_env_command(env, cmd, logger=logger)
    return out


def apply_track_nav_command_to_critic_obs(obs: torch.Tensor, env: Any, stage: Any = None, logger: Any = None) -> torch.Tensor:
    """Overwrite critic velocity command with the same goal-directed command."""
    if obs is None or not torch.is_tensor(obs) or obs.shape[-1] < 12:
        return obs
    if stage is not None and not bool(getattr(stage, "enable_track_nav_command", False)):
        return obs

    cmd = compute_track_nav_command(env, obs, stage=stage, logger=logger)
    if cmd is None:
        return obs

    out = obs.clone()
    out[..., _CRITIC_CMD_SLICE] = cmd.to(device=out.device, dtype=out.dtype)
    _try_write_env_command(env, cmd, logger=logger)
    return out


def _read_bridge_goal_robot(
    env: Any,
    reference: torch.Tensor,
    num_envs: int,
    stage: Any = None,
):
    goal_pos, goal_mask, goal_source = TrackTensorBridge._extract_goal_pos(
        env,
        infos=None,
        reference=reference,
        num_envs=num_envs,
    )

    robot_pos, robot_mask, robot_source = TrackTensorBridge._extract_robot_pos(
        env,
        infos=None,
        reference=reference,
        num_envs=num_envs,
    )

    robot_yaw, robot_yaw_mask, robot_yaw_source, robot_yaw_detail, quat_norm_mean, yaw_attr = (
        TrackTensorBridge._extract_robot_yaw(
            env,
            infos=None,
            reference=reference,
            num_envs=num_envs,
        )
    )

    goal_yaw, goal_yaw_mask, goal_yaw_source = TrackTensorBridge._extract_goal_yaw(
        env,
        infos=None,
        reference=reference,
        num_envs=num_envs,
    )

    valid = goal_mask & robot_mask & robot_yaw_mask

    return {
        "goal_pos": goal_pos,
        "goal_mask": goal_mask,
        "goal_source": goal_source,
        "robot_pos": robot_pos,
        "robot_mask": robot_mask,
        "robot_source": robot_source,
        "robot_yaw": robot_yaw,
        "robot_yaw_mask": robot_yaw_mask,
        "robot_yaw_source": robot_yaw_source,
        "robot_yaw_detail": robot_yaw_detail,
        "goal_yaw": goal_yaw,
        "goal_yaw_mask": goal_yaw_mask,
        "goal_yaw_source": goal_yaw_source,
        "valid": valid,
    }


def compute_track_nav_command(env: Any, reference: torch.Tensor, stage: Any = None, logger: Any = None) -> Optional[torch.Tensor]:
    """Compute [vx, vy, wz] from goal pose and optional nav_scanner clearance."""
    try:
        if env is None or reference is None or not torch.is_tensor(reference):
            return None
        num_envs = int(reference.shape[0]) if reference.ndim >= 2 else 1
        device = reference.device
        dtype = reference.dtype
        shape_probe = _is_observation_shape_probe()

        bridge = _read_bridge_goal_robot(env, reference, num_envs, stage=stage)

        valid = bridge["valid"]
        valid_ratio = valid.float().mean()
        valid_ratio_value = float(valid_ratio.detach().item())

        if (
            not shape_probe
            and bool(getattr(stage, "nav_cmd_fail_fast", False))
            and valid_ratio_value < 0.9
        ):
            raise _NavCommandRequiredSignalError(
                f"[NavCommand] partial invalid goal/robot/yaw data; "
                f"valid_ratio={valid_ratio_value:.4f}; "
                "expected all envs to have valid goal_pos / robot_pos / robot_yaw"
            )

        if not bool(valid.any().item()):
            if shape_probe:
                return None

            if bool(getattr(stage, "nav_cmd_fail_fast", False)):
                raise _NavCommandRequiredSignalError(
                    "[NavCommand] TrackTensorBridge could not read valid "
                    "goal_positions / robot.data.root_pos_w / robot_yaw"
                )

            if logger is not None and not _Once.warned_no_goal:
                logger.warning(
                    f"[NavCommand] TrackTensorBridge goal/robot invalid; "
                    f"valid_ratio={float(valid_ratio.detach().item()):.4f}; "
                    "keeping original velocity command"
                )
                _Once.warned_no_goal = True

            return None

        goal_xy = bridge["goal_pos"][:, :2]
        root_xy = bridge["robot_pos"][:, :2]
        yaw = bridge["robot_yaw"]

        delta_w = goal_xy - root_xy

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        dx_b = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
        dy_b = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]
        dist = torch.linalg.norm(delta_w, dim=1).clamp_min(1.0e-6)
        yaw_err = torch.atan2(dy_b, dx_b)

        # Conservative default: preserve the Standard locomotion gait and only
        # steer the already learned velocity-following policy.
        target_speed = float(getattr(stage, "nav_cmd_target_speed", 0.80))
        min_speed = float(getattr(stage, "nav_cmd_min_speed", 0.18))
        max_speed = float(getattr(stage, "nav_cmd_max_speed", 1.10))
        max_lateral = float(getattr(stage, "nav_cmd_max_lateral", 0.25))
        yaw_gain = float(getattr(stage, "nav_cmd_yaw_gain", 1.35))
        max_yaw = float(getattr(stage, "nav_cmd_max_yaw", 0.95))
        slow_radius = max(float(getattr(stage, "nav_cmd_goal_slow_radius", 0.80)), 1.0e-6)

        align = torch.cos(yaw_err).clamp(0.0, 1.0)
        near_scale = (dist / slow_radius).clamp(0.15, 1.0)
        vx = (target_speed * (0.35 + 0.65 * align) * near_scale).clamp(min_speed, max_speed)
        vy = (0.20 * torch.tanh(dy_b)).clamp(-max_lateral, max_lateral)
        wz = yaw_gain * yaw_err

        nav_clearance = _read_nav_clearance(env, reference, num_envs, stage, logger=logger)
        if nav_clearance is not None and bool(getattr(stage, "nav_cmd_enable_scanner_avoidance", True)):
            front_m, left_m, right_m = nav_clearance
            front_block_m = float(getattr(stage, "nav_cmd_front_block_m", 0.55))
            deadend_m = float(getattr(stage, "nav_cmd_deadend_m", 0.75))
            avoid_yaw = float(getattr(stage, "nav_cmd_avoid_yaw", 0.75))
            blocked = front_m < front_block_m
            deadend = torch.maximum(left_m, right_m) < deadend_m
            # Positive wz means turn left in the command convention used by Unitree velocity tasks.
            turn_sign = torch.where(left_m >= right_m, torch.ones_like(wz), -torch.ones_like(wz))
            avoid = torch.where(blocked | deadend, turn_sign * avoid_yaw, torch.zeros_like(wz))
            wz = wz + avoid
            vx = torch.where(blocked | deadend, torch.minimum(vx, vx.new_full(vx.shape, 0.28)), vx)
            vy = torch.where(blocked | deadend, torch.zeros_like(vy), vy)

        # Near goal, slow down and optionally align to goal_yaw if present.
        goal_yaw = bridge["goal_yaw"]
        goal_yaw_mask = bridge["goal_yaw_mask"]

        if bool(goal_yaw_mask.any().item()):
            yaw_to_goal = torch.atan2(torch.sin(goal_yaw - yaw), torch.cos(goal_yaw - yaw))
            near = dist < slow_radius
            near_and_valid = near & goal_yaw_mask
            wz = torch.where(
                near_and_valid,
                0.5 * wz + 0.5 * yaw_gain * yaw_to_goal,
                wz,
            )

        cmd = torch.stack(
            [
                torch.nan_to_num(vx, nan=0.0, posinf=max_speed, neginf=0.0),
                torch.nan_to_num(vy, nan=0.0, posinf=max_lateral, neginf=-max_lateral),
                torch.nan_to_num(wz, nan=0.0, posinf=max_yaw, neginf=-max_yaw).clamp(-max_yaw, max_yaw),
            ],
            dim=1,
        ).to(device=device, dtype=dtype)
        return cmd
    except Exception as exc:  # keep training alive unless Track fail-fast is enabled
        if isinstance(exc, _NavCommandRequiredSignalError):
            raise
        if logger is not None and not _Once.warned_nav_error:
            logger.warning(f"[NavCommand] failed to compute command, fallback to original command: {exc}")
            _Once.warned_nav_error = True
        return None


def _try_write_env_command(env: Any, cmd: torch.Tensor, logger: Any = None) -> None:
    """Write command_manager['base_velocity'] so rewards match the injected obs command."""
    try:
        wrote = False
        seen_cm = set()

        for candidate in TrackTensorBridge._env_candidates(env):
            cm = getattr(candidate, "command_manager", None)
            if cm is None or not hasattr(cm, "get_command"):
                continue

            cm_id = id(cm)
            if cm_id in seen_cm:
                continue
            seen_cm.add(cm_id)

            base_cmd = cm.get_command("base_velocity")
            if base_cmd is None or not torch.is_tensor(base_cmd) or base_cmd.shape[-1] < 3:
                continue

            with torch.no_grad():
                base_cmd[..., :3].copy_(cmd.to(device=base_cmd.device, dtype=base_cmd.dtype))

            wrote = True

        if not wrote and logger is not None and not _Once.warned_cmd_write:
            logger.warning("[NavCommand] command_manager base_velocity not found; rewards may use original command")
            _Once.warned_cmd_write = True

    except Exception as exc:
        if logger is not None and not _Once.warned_cmd_write:
            logger.warning(f"[NavCommand] could not write command_manager base_velocity: {exc}")
            _Once.warned_cmd_write = True


def _env_candidates(env: Any):
    if env is None:
        return []
    candidates = []
    seen = set()
    pending = [env]
    attr_names = ("unwrapped", "env", "_env", "base_env", "isaac_env")

    while pending and len(candidates) < 8:
        current = pending.pop(0)
        if current is None:
            continue
        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        candidates.append(current)

        for attr in attr_names:
            try:
                child = getattr(current, attr, None)
            except Exception:
                child = None
            if child is not None and id(child) not in seen:
                pending.append(child)

    return candidates


def _as_env_xy_or_none(value: Any, reference: torch.Tensor, num_envs: int) -> Optional[torch.Tensor]:
    try:
        tensor = _to_tensor(value, reference)
    except Exception:
        return None

    if tensor.ndim == 1 and tensor.numel() >= 2:
        selected = tensor[:2].view(1, 2).expand(num_envs, 2)
    elif tensor.ndim >= 2 and tensor.shape[0] == num_envs and tensor.shape[-1] >= 2:
        selected = tensor.reshape(num_envs, -1)[:, :2]
    elif tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[-1] >= 2:
        selected = tensor.reshape(1, -1)[:, :2].expand(num_envs, 2)
    else:
        return None

    if not torch.isfinite(selected).all(dim=1).any().item():
        return None

    return torch.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)


def _as_env_scalar_or_none(value: Any, reference: torch.Tensor, num_envs: int) -> Optional[torch.Tensor]:
    try:
        tensor = _to_tensor(value, reference)
    except Exception:
        return None

    if tensor.ndim == 0:
        selected = tensor.view(1).expand(num_envs)
    elif tensor.numel() == 1:
        selected = tensor.flatten().expand(num_envs)
    elif tensor.numel() == num_envs:
        selected = tensor.reshape(num_envs)
    elif tensor.ndim >= 2 and tensor.shape[0] == num_envs:
        selected = tensor.reshape(num_envs, -1)[:, 0]
    else:
        return None

    if not torch.isfinite(selected).any().item():
        return None

    return torch.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)


def _get_goal_positions(env: Any, reference: torch.Tensor, num_envs: int) -> Optional[torch.Tensor]:
    for candidate in _env_candidates(env):
        goal = getattr(candidate, "goal_positions", None)
        if goal is None:
            continue
        tensor = _as_env_xy_or_none(goal, reference, num_envs)
        if tensor is not None:
            return tensor
    return None


def _get_goal_yaw(env: Any, reference: torch.Tensor, num_envs: int) -> Optional[torch.Tensor]:
    for candidate in _env_candidates(env):
        yaw = getattr(candidate, "goal_yaw", None)
        if yaw is None:
            continue
        tensor = _as_env_scalar_or_none(yaw, reference, num_envs)
        if tensor is not None:
            return tensor
    return None


def _get_robot_asset(env: Any) -> Optional[Any]:
    for candidate in _env_candidates(env):
        objects = [
            getattr(candidate, "robot", None),
            getattr(candidate, "asset", None),
        ]

        scene = getattr(candidate, "scene", None)
        if scene is not None:
            for name in ("robot", "unitree_go2", "go2", "Go2"):
                try:
                    objects.append(scene[name])
                except Exception:
                    pass

            articulations = getattr(scene, "articulations", None)
            if isinstance(articulations, dict):
                for name in ("robot", "unitree_go2", "go2", "Go2"):
                    objects.append(articulations.get(name))
                try:
                    if articulations:
                        objects.append(next(iter(articulations.values())))
                except Exception:
                    pass

        for obj in objects:
            data = getattr(obj, "data", None)
            if data is not None and getattr(data, "root_pos_w", None) is not None:
                return obj

    return None


def _get_nav_sensor(env: Any) -> Optional[Any]:
    for candidate in _env_candidates(env):
        scene = getattr(candidate, "scene", None)
        if scene is None:
            continue

        sensors = getattr(scene, "sensors", None)

        if isinstance(sensors, dict):
            sensor = sensors.get("nav_scanner")
            if sensor is not None:
                return sensor

        try:
            if sensors is not None:
                sensor = sensors["nav_scanner"]
                if sensor is not None:
                    return sensor
        except Exception:
            pass

        try:
            sensor = getattr(sensors, "nav_scanner", None) if sensors is not None else None
            if sensor is not None:
                return sensor
        except Exception:
            pass

        try:
            sensor = getattr(scene, "nav_scanner", None)
            if sensor is not None:
                return sensor
        except Exception:
            pass

        try:
            raw_sensors = getattr(scene, "_sensors", None)
            if isinstance(raw_sensors, dict):
                sensor = raw_sensors.get("nav_scanner")
                if sensor is not None:
                    return sensor
        except Exception:
            pass

    return None


def _warn_no_nav_sensor(logger: Any) -> None:
    if logger is not None and not _Once.warned_no_nav_sensor:
        logger.warning("[NavCommand] nav_scanner not available; obstacle avoidance is disabled")
        _Once.warned_no_nav_sensor = True


def _get_robot_yaw(robot: Any, reference: torch.Tensor, num_envs: int) -> torch.Tensor:
    data = getattr(robot, "data", None)
    for name in ("heading_w", "yaw", "root_yaw_w"):
        value = getattr(data, name, None)
        if value is not None:
            yaw = _to_tensor(value, reference).reshape(-1)
            if yaw.numel() == 1 and num_envs != 1:
                yaw = yaw.expand(num_envs)
            if yaw.numel() == num_envs:
                return yaw

    quat = getattr(data, "root_quat_w", None)
    if quat is not None:
        q = _to_tensor(quat, reference)
        if q.ndim == 1:
            q = q.view(1, -1).expand(num_envs, -1)
        if q.ndim == 2 and q.shape[0] == num_envs and q.shape[1] >= 4:
            # Isaac Lab root_quat_w is wxyz.
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return reference.new_zeros(num_envs)


def _read_nav_clearance(
    env: Any, reference: torch.Tensor, num_envs: int, stage: Any = None, logger: Any = None
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return coarse front/left/right clearance in meters through TrackTensorBridge."""
    scanner_required = (
        bool(getattr(stage, "nav_cmd_fail_fast", False))
        and bool(getattr(stage, "nav_cmd_enable_scanner_avoidance", True))
        and not _is_observation_shape_probe()
    )

    contract = TrackTensorBridge._extract_nav_contract(
        env=env,
        infos=None,
        reference=reference,
        num_envs=num_envs,
        stage_config=stage,
    )

    valid = contract.valid_mask
    env_valid_ratio = valid.float().mean()
    env_valid_ratio_value = float(env_valid_ratio.detach().item())

    if scanner_required and env_valid_ratio_value < 0.9:
        reason = getattr(contract, "failure_reason", "unknown")
        stats = getattr(contract, "stats", {})
        valid_ray_ratio = stats.get("p5a2_nav_valid_ray_ratio", 0.0)
        signal_ratio = stats.get("p5a2_nav_signal_ratio", 0.0)

        if isinstance(valid_ray_ratio, torch.Tensor):
            valid_ray_ratio = float(valid_ray_ratio.detach().mean().item())
        else:
            valid_ray_ratio = float(valid_ray_ratio)

        if isinstance(signal_ratio, torch.Tensor):
            signal_ratio = float(signal_ratio.detach().mean().item())
        else:
            signal_ratio = float(signal_ratio)

        raise _NavCommandRequiredSignalError(
            f"[NavCommand] partial invalid nav_scanner data; "
            f"env_valid_ratio={env_valid_ratio_value:.4f}; "
            f"reason={reason}; "
            f"valid_ray_ratio={valid_ray_ratio:.4f}; "
            f"signal_ratio={signal_ratio:.4f}"
        )

    if not bool(valid.any().item()):
        if scanner_required:
            reason = getattr(contract, "failure_reason", "unknown")
            stats = getattr(contract, "stats", {})
            valid_ratio = stats.get("p5a2_nav_valid_ray_ratio", 0.0)
            signal_ratio = stats.get("p5a2_nav_signal_ratio", 0.0)

            if isinstance(valid_ratio, torch.Tensor):
                valid_ratio = float(valid_ratio.detach().mean().item())
            else:
                valid_ratio = float(valid_ratio)

            if isinstance(signal_ratio, torch.Tensor):
                signal_ratio = float(signal_ratio.detach().mean().item())
            else:
                signal_ratio = float(signal_ratio)

            raise _NavCommandRequiredSignalError(
                f"[NavCommand] TrackTensorBridge nav_scanner invalid; "
                f"reason={reason}; valid_ray_ratio={valid_ratio:.4f}; "
                f"signal_ratio={signal_ratio:.4f}"
            )

        _warn_no_nav_sensor(logger)
        return None

    nav = contract.nav_features

    nav_cfg = NavMeterConfig.from_stage(stage)
    max_m = max(float(nav_cfg.max_clearance_m), 1.0e-6)

    front_m = nav[:, NavFeatureLayout.FRONT_CLEARANCE_NORM] * max_m
    left_m = nav[:, NavFeatureLayout.LEFT_CLEARANCE_NORM] * max_m
    right_m = nav[:, NavFeatureLayout.RIGHT_CLEARANCE_NORM] * max_m

    full = reference.new_full((num_envs,), max_m)
    front_m = torch.where(valid, front_m, full)
    left_m = torch.where(valid, left_m, full)
    right_m = torch.where(valid, right_m, full)

    return front_m, left_m, right_m


def _normalize_hits(hits: torch.Tensor, num_envs: int) -> Optional[torch.Tensor]:
    if hits.ndim == 2 and hits.shape[-1] >= 3:
        return hits[:, :3].unsqueeze(0).expand(num_envs, -1, -1)
    if hits.ndim == 3 and hits.shape[0] == num_envs and hits.shape[-1] >= 3:
        return hits[:, :, :3]
    if hits.ndim == 3 and hits.shape[0] == 1 and hits.shape[-1] >= 3:
        return hits[:, :, :3].expand(num_envs, -1, -1)
    return None


def _normalize_origin(origin: torch.Tensor, num_envs: int, hits: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if origin is None or hits is None:
        return None
    num_rays = int(hits.shape[1])
    if origin.ndim == 1 and origin.numel() >= 3:
        return origin[:3].view(1, 1, 3).expand(num_envs, num_rays, 3)
    if origin.ndim == 2 and origin.shape[0] == num_envs and origin.shape[-1] >= 3:
        return origin[:, :3].unsqueeze(1).expand(num_envs, num_rays, 3)
    if origin.ndim == 2 and origin.shape[0] == 1 and origin.shape[-1] >= 3:
        return origin[:, :3].view(1, 1, 3).expand(num_envs, num_rays, 3)
    if origin.ndim == 3 and origin.shape[0] == num_envs and origin.shape[1] == 1 and origin.shape[-1] >= 3:
        return origin[:, :, :3].expand(num_envs, num_rays, 3)
    if origin.ndim == 3 and origin.shape[0] == 1 and origin.shape[1] == 1 and origin.shape[-1] >= 3:
        return origin[:, :, :3].expand(num_envs, num_rays, 3)
    if origin.ndim == 3 and origin.shape[0] == num_envs and origin.shape[1] == num_rays and origin.shape[-1] >= 3:
        return origin[:, :, :3]
    if origin.ndim == 3 and origin.shape[0] == 1 and origin.shape[1] == num_rays and origin.shape[-1] >= 3:
        return origin[:, :, :3].expand(num_envs, num_rays, 3)
    return None


def _reduce_to_16(distance: torch.Tensor, valid: torch.Tensor, max_m: float) -> torch.Tensor:
    num_envs, num_rays = distance.shape
    sector_count = 16
    pad = (sector_count - (num_rays % sector_count)) % sector_count
    if pad > 0:
        distance = torch.cat([distance, distance.new_full((num_envs, pad), max_m)], dim=1)
        valid = torch.cat([valid, torch.zeros((num_envs, pad), dtype=torch.bool, device=valid.device)], dim=1)
    rays_per_sector = max(distance.shape[1] // sector_count, 1)
    distance = distance[:, : sector_count * rays_per_sector].reshape(num_envs, sector_count, rays_per_sector)
    valid = valid[:, : sector_count * rays_per_sector].reshape(num_envs, sector_count, rays_per_sector)
    count = valid.float().sum(dim=2).clamp_min(1.0)
    sectors = (distance * valid.float()).sum(dim=2) / count
    sectors = torch.where(valid.any(dim=2), sectors, distance.new_full((num_envs, sector_count), max_m))
    return torch.nan_to_num(sectors, nan=max_m, posinf=max_m, neginf=0.0).clamp(0.0, max_m)


def _to_tensor(value: Any, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
