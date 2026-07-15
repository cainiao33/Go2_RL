# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Track raw-data tensor bridge for P5A-0 source probing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from agent_ppo.feature.feature_layout import BasePolicyLayout, GoalFeatureLayout, NavFeatureLayout
from agent_ppo.feature.nav_signal import (
    NavMeterConfig,
    compute_nav_flags_from_meter,
    sanitize_distance_m,
)


@dataclass
class TrackBridgePayload:
    goal_features: torch.Tensor
    nav_features: torch.Tensor
    bridge_stats: dict[str, Any]
    source_flags: dict[str, Any]


@dataclass
class NavContractResult:
    nav_features: torch.Tensor
    valid_mask: torch.Tensor
    tensor_mask: torch.Tensor
    object_mask: torch.Tensor
    source_id: float
    path_id: float
    stats: dict[str, Any]
    attrs: tuple[str, ...]
    failure_reason: str


class TrackTensorBridge:
    """Probe Track-only raw sources and immediately convert them to tensors."""

    SOURCE_NONE = 0.0
    SOURCE_ENV = 1.0
    SOURCE_INFOS = 2.0
    SOURCE_ROBOT_DATA = 3.0
    SOURCE_NAV_SENSOR = 4.0
    SOURCE_OBJECT_NO_TENSOR = 5.0
    SOURCE_TERMINAL_GOAL_DIST = 6.0
    SOURCE_POLICY_HEIGHT_SCAN = 7.0
    SOURCE_OBS_SLICE_UNKNOWN = 8.0
    SOURCE_ID_SCALE = SOURCE_OBS_SLICE_UNKNOWN

    DETAIL_NONE = 0.0
    DETAIL_INFOS_ROBOT_YAW = 1.0
    DETAIL_INFOS_HEADING = 2.0
    DETAIL_INFOS_ROOT_QUAT_WXYZ = 3.0
    DETAIL_ROBOT_DATA_HEADING_W = 4.0
    DETAIL_ROBOT_DATA_YAW = 5.0
    DETAIL_ROBOT_DATA_ROOT_YAW_W = 6.0
    DETAIL_ROBOT_DATA_ROOT_QUAT_WXYZ = 7.0
    DETAIL_ID_SCALE = 11.0

    SOURCE_NAMES = {
        SOURCE_NONE: "none",
        SOURCE_ENV: "env",
        SOURCE_INFOS: "infos",
        SOURCE_ROBOT_DATA: "robot.data",
        SOURCE_NAV_SENSOR: "env.scene.sensors.nav_scanner",
        SOURCE_OBJECT_NO_TENSOR: "object_no_tensor",
        SOURCE_TERMINAL_GOAL_DIST: "terminal_goal_dist",
        SOURCE_POLICY_HEIGHT_SCAN: "policy_height_scan",
        SOURCE_OBS_SLICE_UNKNOWN: "obs_slice_unknown_source",
    }

    _GOAL_POS_INFO_KEYS = ("goal_positions", "goal_pos", "target_pos")
    _GOAL_YAW_INFO_KEYS = ("goal_yaw",)
    _ROBOT_POS_INFO_KEYS = ("root_pos", "base_pos", "robot_pos")
    _ROBOT_DATA_YAW_ATTRS = ("heading_w", "yaw", "root_yaw_w")
    _ROBOT_DATA_QUAT_ATTRS = ("root_quat_w",)
    _OUTCOME_INFO_KEYS = ("success", "completed", "reach_goal", "goal_reached")
    _TERMINAL_GOAL_DIST_INFO_KEYS = ("terminal_goal_dist", "final_goal_dist", "terminal_distance_to_goal")
    _NAV_TENSOR_ATTRS = ("ray_hits_w", "pos_w")

    NAV_PATH_NONE = 0.0
    NAV_PATH_RAY_HITS_POS_W = 2.0

    _logged_source_signatures: set[tuple[str, str, str, str, str, str, str, float]] = set()
    _logged_nav_attrs_warning = False
    _logged_robot_missing_warning = False
    _robot_yaw_was_available = False
    _logged_robot_yaw_drop_scan = False

    @staticmethod
    def build(
        reference_tensor,
        env=None,
        infos=None,
        stage_config=None,
        logger=None,
        source_context="workflow",
    ):
        # P5A DATAFLOW GUARD:
        # This bridge is authoritative only in contexts that can access raw env/robot/sensor objects.
        # In ObservationProcess it can build real goal/nav telemetry.
        # In aisrv workflow it may return an empty payload; callers must check usable signals before trusting it.
        reference = TrackTensorBridge._to_reference_tensor(reference_tensor)
        if reference.ndim == 1:
            reference = reference.view(1, -1)
        num_envs = int(reference.shape[0])
        env_candidate_count = len(TrackTensorBridge._env_candidates(env))

        goal_features = reference.new_zeros((num_envs, GoalFeatureLayout.DIM))
        nav_features = reference.new_zeros((num_envs, NavFeatureLayout.DIM))

        goal_pos, goal_pos_mask, goal_source = TrackTensorBridge._extract_goal_pos(
            env,
            infos,
            reference,
            num_envs,
        )
        robot_pos, robot_pos_mask, robot_source = TrackTensorBridge._extract_robot_pos(
            env,
            infos,
            reference,
            num_envs,
        )
        goal_yaw, goal_yaw_mask, goal_yaw_source = TrackTensorBridge._extract_goal_yaw(
            env,
            infos,
            reference,
            num_envs,
        )
        (
            robot_yaw,
            robot_yaw_mask,
            robot_yaw_source,
            robot_yaw_detail_id,
            robot_yaw_quat_norm_mean,
            robot_yaw_selected_attr,
        ) = TrackTensorBridge._extract_robot_yaw(
            env,
            infos,
            reference,
            num_envs,
        )
        policy_height_mask, policy_height_stats = TrackTensorBridge._probe_policy_height_scan(
            reference,
            num_envs,
        )

        nav_contract = TrackTensorBridge._extract_nav_contract(
            env=env,
            infos=infos,
            reference=reference,
            num_envs=num_envs,
            stage_config=stage_config,
        )
        nav_features = nav_contract.nav_features
        nav_mask = nav_contract.valid_mask
        nav_source = nav_contract.source_id
        nav_attrs = nav_contract.attrs
        outcome_value, outcome_mask, outcome_source = TrackTensorBridge._extract_outcome(
            infos,
            reference,
            num_envs,
            stage_config,
        )
        yaw_source_missing_flag = 0.0
        if bool(goal_yaw_mask.any().item()) and goal_yaw_source <= TrackTensorBridge.SOURCE_NONE:
            goal_yaw_source = TrackTensorBridge.SOURCE_OBS_SLICE_UNKNOWN
            yaw_source_missing_flag = 1.0
        if bool(robot_yaw_mask.any().item()) and (
            robot_yaw_source <= TrackTensorBridge.SOURCE_NONE
            or robot_yaw_source == TrackTensorBridge.SOURCE_OBS_SLICE_UNKNOWN
        ):
            robot_yaw_source = TrackTensorBridge.SOURCE_OBS_SLICE_UNKNOWN
            yaw_source_missing_flag = 1.0

        goal_dist_mask = goal_pos_mask & robot_pos_mask
        goal_distance = torch.linalg.norm(goal_pos - robot_pos, dim=1)
        goal_distance = torch.nan_to_num(goal_distance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        goal_distance = torch.where(goal_dist_mask, goal_distance, torch.zeros_like(goal_distance))
        distance_scale = float(getattr(stage_config, "goal_feature_distance_scale", 20.0))
        distance_scale = max(distance_scale, 1.0e-6)
        delta_world = goal_pos - robot_pos
        dx_world_norm = torch.clamp(delta_world[:, 0] / distance_scale, -1.0, 1.0)
        dy_world_norm = torch.clamp(delta_world[:, 1] / distance_scale, -1.0, 1.0)
        dist_norm = torch.clamp(goal_distance / distance_scale, 0.0, 1.0)
        cos_yaw = torch.cos(robot_yaw)
        sin_yaw = torch.sin(robot_yaw)
        dx_body = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
        dy_body = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]
        dx_body_norm = torch.clamp(dx_body / distance_scale, -1.0, 1.0)
        dy_body_norm = torch.clamp(dy_body / distance_scale, -1.0, 1.0)
        yaw_error = goal_yaw - robot_yaw
        yaw_error_mask = goal_yaw_mask & robot_yaw_mask

        goal_features[:, GoalFeatureLayout.GOAL_POS_VALID] = goal_pos_mask.float()
        goal_features[:, GoalFeatureLayout.ROBOT_POS_VALID] = robot_pos_mask.float()
        goal_features[:, GoalFeatureLayout.GOAL_DIST_VALID] = goal_dist_mask.float()
        goal_features[:, GoalFeatureLayout.GOAL_YAW_VALID] = goal_yaw_mask.float()
        goal_features[:, GoalFeatureLayout.ROBOT_YAW_VALID] = robot_yaw_mask.float()
        goal_features[:, GoalFeatureLayout.DX_WORLD_NORM] = torch.where(goal_dist_mask, dx_world_norm, torch.zeros_like(dx_world_norm))
        goal_features[:, GoalFeatureLayout.DY_WORLD_NORM] = torch.where(goal_dist_mask, dy_world_norm, torch.zeros_like(dy_world_norm))
        goal_features[:, GoalFeatureLayout.DIST_NORM] = torch.where(goal_dist_mask, dist_norm, torch.zeros_like(dist_norm))
        goal_features[:, GoalFeatureLayout.DX_BODY_NORM] = torch.where(robot_yaw_mask & goal_dist_mask, dx_body_norm, torch.zeros_like(dx_body_norm))
        goal_features[:, GoalFeatureLayout.DY_BODY_NORM] = torch.where(robot_yaw_mask & goal_dist_mask, dy_body_norm, torch.zeros_like(dy_body_norm))
        goal_features[:, GoalFeatureLayout.HEADING_SIN] = torch.where(robot_yaw_mask, torch.sin(robot_yaw), torch.zeros_like(robot_yaw))
        goal_features[:, GoalFeatureLayout.HEADING_COS] = torch.where(robot_yaw_mask, torch.cos(robot_yaw), torch.zeros_like(robot_yaw))
        goal_features[:, GoalFeatureLayout.GOAL_YAW_SIN] = torch.where(goal_yaw_mask, torch.sin(goal_yaw), torch.zeros_like(goal_yaw))
        goal_features[:, GoalFeatureLayout.GOAL_YAW_COS] = torch.where(goal_yaw_mask, torch.cos(goal_yaw), torch.zeros_like(goal_yaw))
        goal_features[:, GoalFeatureLayout.YAW_ERROR_SIN] = torch.where(yaw_error_mask, torch.sin(yaw_error), torch.zeros_like(yaw_error))
        goal_features[:, GoalFeatureLayout.YAW_ERROR_COS] = torch.where(yaw_error_mask, torch.cos(yaw_error), torch.zeros_like(yaw_error))
        source_names = {
            "goal_source": TrackTensorBridge._source_name(goal_source),
            "robot_source": TrackTensorBridge._source_name(robot_source),
            "goal_yaw_source": TrackTensorBridge._source_name(goal_yaw_source),
            "robot_yaw_source": TrackTensorBridge._source_name(robot_yaw_source),
            "nav_source": TrackTensorBridge._source_name(nav_source),
            "outcome_source": TrackTensorBridge._source_name(outcome_source),
            "source_context": source_context,
            "robot_yaw_selected_attr": robot_yaw_selected_attr,
            "robot_yaw_detail_id": robot_yaw_detail_id,
        }
        source_flags = {
            "goal_pos_valid": goal_pos_mask,
            "robot_pos_valid": robot_pos_mask,
            "goal_dist_valid": goal_dist_mask,
            # P5A DATAFLOW GUARD:
            # source_flags['goal_distance'] is raw physical distance.
            # source_flags['goal_distance_norm'] is separate diagnostic/feature-scale data.
            # Keep them separate to avoid corrupting progress/stuck calculations.
            "goal_distance": goal_distance,
            "goal_distance_norm": torch.where(goal_dist_mask, dist_norm, torch.zeros_like(dist_norm)),
            "goal_yaw_valid": goal_yaw_mask,
            "robot_yaw_valid": robot_yaw_mask,
            "nav_valid": nav_mask,
            "policy_height_available": policy_height_mask,
            "effective_nav_available": nav_mask,
            "nav_path_id": float(nav_contract.path_id),
            "nav_failure_reason": nav_contract.failure_reason,
            "outcome_valid": outcome_mask,
            "outcome_success": outcome_value,
            "goal_source_id": goal_source,
            "robot_source_id": robot_source,
            "goal_yaw_source_id": goal_yaw_source,
            "robot_yaw_source_id": robot_yaw_source,
            "robot_yaw_detail_id": robot_yaw_detail_id,
            "robot_yaw_quat_norm_mean": robot_yaw_quat_norm_mean,
            "robot_yaw_selected_attr": robot_yaw_selected_attr,
            "yaw_source_missing_flag": yaw_source_missing_flag,
            "nav_source_id": nav_source,
            "outcome_source_id": outcome_source,
            "source_names": source_names,
            "source_context": source_context,
            "env_candidate_count": float(env_candidate_count),
        }
        bridge_stats = {
            "p5a_bridge_goal_pos_tensor_ratio": goal_pos_mask.float().mean(),
            "p5a_bridge_robot_pos_tensor_ratio": robot_pos_mask.float().mean(),
            "p5a_bridge_goal_dist_tensor_ratio": goal_dist_mask.float().mean(),
            "p5a_bridge_goal_yaw_tensor_ratio": goal_yaw_mask.float().mean(),
            "p5a_bridge_robot_yaw_tensor_ratio": robot_yaw_mask.float().mean(),
            "p5a_bridge_nav_tensor_ratio": nav_mask.float().mean(),
            "p5a_bridge_outcome_tensor_ratio": outcome_mask.float().mean(),
            "p5a_bridge_goal_source_id": float(goal_source),
            "p5a_bridge_robot_source_id": float(robot_source),
            "p5a_bridge_goal_yaw_source_id": float(goal_yaw_source),
            "p5a_bridge_robot_yaw_source_id": float(robot_yaw_source),
            "p5a_bridge_robot_yaw_detail_id": float(robot_yaw_detail_id),
            "p5a_yaw_source_missing_flag": float(yaw_source_missing_flag),
            "p5a_robot_yaw_quat_norm_mean": float(robot_yaw_quat_norm_mean),
            "p5a_bridge_nav_source_id": float(nav_source),
            "p5a_bridge_outcome_source_id": float(outcome_source),
            "p5a_obs_bridge_candidate_count": float(env_candidate_count),
        }
        bridge_stats.update(
            TrackTensorBridge._p5a1_goal_stats(
                goal_features=goal_features,
                goal_pos_mask=goal_pos_mask,
                robot_pos_mask=robot_pos_mask,
                goal_dist_mask=goal_dist_mask,
                goal_yaw_mask=goal_yaw_mask,
                robot_yaw_mask=robot_yaw_mask,
                goal_distance=goal_distance,
                dist_norm=dist_norm,
                delta_world=delta_world,
                dx_body=dx_body,
                dy_body=dy_body,
                distance_scale=distance_scale,
            )
        )
        bridge_stats.update(policy_height_stats)
        bridge_stats.update(nav_contract.stats)
        bridge_stats.update(
            {
                "p5a2_effective_nav_available_ratio": nav_mask.float().mean(),
                "p5a2_scanner_nav_available_ratio": nav_mask.float().mean(),
                "p5a2_nav_object_exists_ratio": nav_contract.object_mask.float().mean(),
                "p5a2_nav_path_id": float(nav_contract.path_id),
            }
        )

        if str(source_context).startswith("policy_observation_process"):
            bridge_stats.update(
                {
                    "p5a_obs_bridge_goal_pos_ratio": goal_pos_mask.float().mean(),
                    "p5a_obs_bridge_robot_pos_ratio": robot_pos_mask.float().mean(),
                    "p5a_obs_bridge_goal_dist_ratio": goal_dist_mask.float().mean(),
                    "p5a_obs_bridge_goal_yaw_ratio": goal_yaw_mask.float().mean(),
                    "p5a_obs_bridge_robot_yaw_ratio": robot_yaw_mask.float().mean(),
                    "p5a_obs_bridge_nav_ratio": nav_mask.float().mean(),
                }
            )

        TrackTensorBridge._maybe_log_sources(logger, source_names)
        TrackTensorBridge._maybe_log_robot_yaw_drop_scan(
            logger=logger,
            env=env,
            reference=reference,
            num_envs=num_envs,
            robot_yaw_mask=robot_yaw_mask,
            source_context=source_context,
        )
        if nav_source == TrackTensorBridge.SOURCE_OBJECT_NO_TENSOR:
            TrackTensorBridge._maybe_log_nav_attrs_warning(logger, nav_attrs)
        if not robot_pos_mask.any().item():
            TrackTensorBridge._maybe_log_robot_missing_warning(logger)

        goal_features = TrackTensorBridge._sanitize_features(goal_features)
        nav_features = TrackTensorBridge._sanitize_features(nav_features)
        return TrackBridgePayload(
            goal_features=goal_features,
            nav_features=nav_features,
            bridge_stats=bridge_stats,
            source_flags=source_flags,
        )

    @staticmethod
    def _extract_goal_pos(env, infos, reference, num_envs):
        for candidate in TrackTensorBridge._env_candidates(env):
            value = getattr(candidate, "goal_positions", None)
            if value is not None:
                tensor, mask = TrackTensorBridge._as_env_xy(value, reference, num_envs)
                if mask.any().item():
                    return tensor, mask, TrackTensorBridge.SOURCE_ENV

        value = TrackTensorBridge._first_info_value(infos, TrackTensorBridge._GOAL_POS_INFO_KEYS)
        if value is not None:
            tensor, mask = TrackTensorBridge._as_env_xy(value, reference, num_envs)
            if mask.any().item():
                return tensor, mask, TrackTensorBridge.SOURCE_INFOS

        return TrackTensorBridge._zeros_xy(reference, num_envs), TrackTensorBridge._zeros_mask(reference, num_envs), TrackTensorBridge.SOURCE_NONE

    @staticmethod
    def _extract_goal_yaw(env, infos, reference, num_envs):
        for candidate in TrackTensorBridge._env_candidates(env):
            value = getattr(candidate, "goal_yaw", None)
            if value is not None:
                tensor, mask = TrackTensorBridge._as_env_scalar(value, reference, num_envs)
                if mask.any().item():
                    return tensor, mask, TrackTensorBridge.SOURCE_ENV

        value = TrackTensorBridge._first_info_value(infos, TrackTensorBridge._GOAL_YAW_INFO_KEYS)
        if value is not None:
            tensor, mask = TrackTensorBridge._as_env_scalar(value, reference, num_envs)
            if mask.any().item():
                return tensor, mask, TrackTensorBridge.SOURCE_INFOS

        return TrackTensorBridge._zeros_scalar(reference, num_envs), TrackTensorBridge._zeros_mask(reference, num_envs), TrackTensorBridge.SOURCE_NONE

    @staticmethod
    def _extract_robot_pos(env, infos, reference, num_envs):
        value = TrackTensorBridge._first_info_value(infos, TrackTensorBridge._ROBOT_POS_INFO_KEYS)
        if value is not None:
            tensor, mask = TrackTensorBridge._as_env_xy(value, reference, num_envs)
            if mask.any().item():
                return tensor, mask, TrackTensorBridge.SOURCE_INFOS

        root_pos = TrackTensorBridge._get_robot_data_attr(env, "root_pos_w")
        if root_pos is not None:
            tensor, mask = TrackTensorBridge._as_env_xy(root_pos, reference, num_envs)
            if mask.any().item():
                return tensor, mask, TrackTensorBridge.SOURCE_ROBOT_DATA

        return TrackTensorBridge._zeros_xy(reference, num_envs), TrackTensorBridge._zeros_mask(reference, num_envs), TrackTensorBridge.SOURCE_NONE

    @staticmethod
    def _extract_robot_yaw(env, infos, reference, num_envs):
        value, attr = TrackTensorBridge._get_robot_data_attr_with_name(env, TrackTensorBridge._ROBOT_DATA_YAW_ATTRS)
        if value is not None:
            tensor, mask = TrackTensorBridge._as_env_scalar(value, reference, num_envs)
            if mask.any().item():
                return (
                    tensor,
                    mask,
                    TrackTensorBridge.SOURCE_ROBOT_DATA,
                    TrackTensorBridge._robot_data_yaw_detail_id(attr),
                    0.0,
                    f"robot.data.{attr}",
                )

        quat_value, attr = TrackTensorBridge._get_robot_data_attr_with_name(env, TrackTensorBridge._ROBOT_DATA_QUAT_ATTRS)
        if quat_value is not None:
            yaw, mask, quat_norm_mean = TrackTensorBridge._yaw_from_quat(quat_value, reference, num_envs)
            if mask.any().item():
                return (
                    yaw,
                    mask,
                    TrackTensorBridge.SOURCE_ROBOT_DATA,
                    TrackTensorBridge._robot_data_quat_detail_id(attr),
                    quat_norm_mean,
                    f"robot.data.{attr}.wxyz",
                )

        return (
            TrackTensorBridge._zeros_scalar(reference, num_envs),
            TrackTensorBridge._zeros_mask(reference, num_envs),
            TrackTensorBridge.SOURCE_NONE,
            TrackTensorBridge.DETAIL_NONE,
            0.0,
            "none",
        )

    @staticmethod
    def _robot_data_yaw_detail_id(attr):
        if attr == "heading_w":
            return TrackTensorBridge.DETAIL_ROBOT_DATA_HEADING_W
        if attr == "root_yaw_w":
            return TrackTensorBridge.DETAIL_ROBOT_DATA_ROOT_YAW_W
        return TrackTensorBridge.DETAIL_ROBOT_DATA_YAW

    @staticmethod
    def _robot_data_quat_detail_id(attr):
        return TrackTensorBridge.DETAIL_ROBOT_DATA_ROOT_QUAT_WXYZ

    @staticmethod
    def _probe_policy_height_scan(reference, num_envs):
        zero_mask = TrackTensorBridge._zeros_mask(reference, num_envs)
        stats = {
            "p5a2_policy_height_available_ratio": 0.0,
            "p5a2_policy_height_finite_ratio": 0.0,
            "p5a2_policy_height_nonzero_ratio": 0.0,
            "p5a2_policy_height_abs_mean": 0.0,
            "p5a2_policy_height_std_mean": 0.0,
        }

        if reference is None or not isinstance(reference, torch.Tensor):
            return zero_mask, stats

        x = reference
        if x.ndim == 1:
            x = x.view(1, -1)

        if x.shape[-1] != BasePolicyLayout.DIM:
            return zero_mask, stats

        scan = x[..., BasePolicyLayout.HEIGHT_SCAN]
        finite = torch.isfinite(scan)
        finite_ratio_per_env = finite.float().mean(dim=-1)
        available = finite_ratio_per_env >= 0.99

        safe = torch.nan_to_num(scan.detach(), nan=0.0, posinf=0.0, neginf=0.0)

        stats.update(
            {
                "p5a2_policy_height_available_ratio": available.float().mean(),
                "p5a2_policy_height_finite_ratio": finite.float().mean(),
                "p5a2_policy_height_nonzero_ratio": (safe.abs() > 1.0e-6).float().mean(),
                "p5a2_policy_height_abs_mean": safe.abs().mean(),
                "p5a2_policy_height_std_mean": safe.std(dim=-1, unbiased=False).mean(),
            }
        )
        return available, stats

    @staticmethod
    def _extract_nav_contract(env, infos, reference, num_envs, stage_config=None):
        sensor = TrackTensorBridge._get_nav_sensor(env)
        if sensor is not None:
            data = getattr(sensor, "data", None)
            hits = getattr(data, "ray_hits_w", None) if data is not None else None
            pos_w = getattr(data, "pos_w", None) if data is not None else None
            attrs = tuple(
                TrackTensorBridge._public_attrs(sensor)
                + TrackTensorBridge._public_attrs(data)
            )

            if hits is not None and pos_w is not None:
                result = TrackTensorBridge._build_nav_from_hits_origin(
                    hits=hits,
                    origin=pos_w,
                    reference=reference,
                    num_envs=num_envs,
                    stage_config=stage_config,
                    source_id=TrackTensorBridge.SOURCE_NAV_SENSOR,
                    path_id=TrackTensorBridge.NAV_PATH_RAY_HITS_POS_W,
                    object_exists=True,
                )
                result.stats["p5a2_nav_origin_source_id"] = 1.0
                result.attrs = attrs
                return result

            result = TrackTensorBridge._empty_nav_contract(
                reference=reference,
                num_envs=num_envs,
                object_exists=True,
                reason="nav_scanner_missing_ray_hits_w_or_pos_w",
            )
            result.attrs = attrs
            result.stats.update(
                {
                    "p5a2_nav_object_exists_ratio": 1.0,
                    "p5a2_raycaster_nav_valid_ratio": 0.0,
                    "p5a2_nav_tensor_available_ratio": 0.0,
                    "p5a2_nav_path_id": 0.0,
                }
            )
            return result

        return TrackTensorBridge._empty_nav_contract(
            reference=reference,
            num_envs=num_envs,
            object_exists=False,
            reason="no_nav_sensor",
        )

    @staticmethod
    def _build_nav_from_hits_origin(
        hits,
        origin,
        reference,
        num_envs,
        stage_config,
        source_id,
        path_id,
        object_exists,
    ):
        try:
            hits = TrackTensorBridge._to_tensor(hits, reference)
            origin = TrackTensorBridge._to_tensor(origin, reference)
        except Exception:
            return TrackTensorBridge._empty_nav_contract(
                reference,
                num_envs,
                object_exists,
                "tensor_convert_failed",
            )

        hits = TrackTensorBridge._normalize_hits(hits, reference, num_envs)
        origin = TrackTensorBridge._normalize_origin(origin, reference, num_envs, hits)

        if hits is None or origin is None:
            return TrackTensorBridge._empty_nav_contract(
                reference,
                num_envs,
                object_exists,
                "shape_invalid",
            )

        hit_finite = torch.isfinite(hits).all(dim=-1)
        origin_finite = torch.isfinite(origin).all(dim=-1)
        hit_nonzero = hits.abs().sum(dim=-1) > 1.0e-6
        valid_ray = hit_finite & origin_finite & hit_nonzero

        nav_cfg = NavMeterConfig.from_stage(stage_config)
        max_m = max(float(nav_cfg.max_clearance_m), 1.0e-6)

        raw_vec = hits - origin
        # P7-2A precheck: clearance is ground-plane distance. The nav scanner
        # may have a large vertical offset, so XYZ norm would fake max clearance.
        raw_xy = raw_vec[..., :2]
        safe_xy = torch.where(
            valid_ray.unsqueeze(-1),
            raw_xy,
            torch.zeros_like(raw_xy),
        )
        distance_m = torch.linalg.norm(safe_xy, dim=-1)
        distance_m = torch.where(valid_ray, distance_m, torch.zeros_like(distance_m))

        extra_stats = {
            "p5a2_ray_hits_finite_ratio": hit_finite.float().mean(),
            "p5a2_ray_origin_finite_ratio": origin_finite.float().mean(),
            "p5a2_ray_valid_ratio": valid_ray.float().mean(),
            "p5a2_ray_hits_nan_ratio": torch.isnan(hits).any(dim=-1).float().mean(),
            "p5a2_ray_hits_inf_ratio": torch.isinf(hits).any(dim=-1).float().mean(),
            "p5a2_ray_hits_zero_ratio": (~hit_nonzero).float().mean(),
            "p5a2_nav_distance_xy_mean": torch.where(valid_ray, distance_m, distance_m.new_tensor(0.0)).float().mean(),
            "p5a2_nav_distance_xy_max": torch.where(valid_ray, distance_m, distance_m.new_tensor(0.0)).float().max(),
        }

        return TrackTensorBridge._build_nav_from_distance(
            distance=distance_m,
            reference=reference,
            num_envs=num_envs,
            stage_config=stage_config,
            source_id=source_id,
            path_id=path_id,
            object_exists=object_exists,
            explicit_valid=valid_ray,
            extra_stats=extra_stats,
        )

    @staticmethod
    def _build_nav_from_distance(
        distance,
        reference,
        num_envs,
        stage_config,
        source_id,
        path_id,
        object_exists,
        explicit_valid=None,
        extra_stats=None,
    ):
        try:
            dist = TrackTensorBridge._to_tensor(distance, reference)
        except Exception:
            return TrackTensorBridge._empty_nav_contract(
                reference,
                num_envs,
                object_exists,
                "distance_convert_failed",
            )

        if dist.ndim == 1:
            if dist.numel() == num_envs:
                dist = dist.view(num_envs, 1)
            else:
                dist = dist.view(1, -1).expand(num_envs, -1)
        elif dist.ndim >= 2 and dist.shape[0] == num_envs:
            dist = dist.reshape(num_envs, -1)
        elif dist.ndim >= 2 and dist.shape[0] == 1:
            dist = dist.reshape(1, -1).expand(num_envs, -1)
        else:
            return TrackTensorBridge._empty_nav_contract(
                reference,
                num_envs,
                object_exists,
                "distance_shape_invalid",
            )

        nav_cfg = NavMeterConfig.from_stage(stage_config)
        max_m = max(float(nav_cfg.max_clearance_m), 1.0e-6)
        min_valid_ratio = float(getattr(stage_config, "p5a2_nav_min_valid_ray_ratio", 0.05))
        min_signal_ratio = float(nav_cfg.min_signal_ratio)

        finite = torch.isfinite(dist)
        if explicit_valid is None:
            valid = finite
        else:
            valid = explicit_valid.reshape(num_envs, -1) & finite

        sanitized_m, valid, sanitize_stats = sanitize_distance_m(dist, valid, nav_cfg)
        clipped_m = sanitized_m.clamp(0.0, max_m)
        scan = (clipped_m / max_m).clamp(0.0, 1.0)

        valid_ray_ratio = valid.float().mean(dim=1)
        signal_ratio = ((scan.abs() > 1.0e-6) & valid).float().mean(dim=1)
        valid_mask = (valid_ray_ratio >= min_valid_ratio) & (signal_ratio >= min_signal_ratio)
        tensor_mask = valid_mask.clone()

        nav = reference.new_zeros((num_envs, NavFeatureLayout.DIM))
        front_m = reference.new_zeros(num_envs)
        left_m = reference.new_zeros(num_envs)
        right_m = reference.new_zeros(num_envs)
        best_free_m = reference.new_zeros(num_envs)
        side_best_m = reference.new_zeros(num_envs)
        front_norm = reference.new_zeros(num_envs)
        left_norm = reference.new_zeros(num_envs)
        right_norm = reference.new_zeros(num_envs)
        best_free_norm = reference.new_zeros(num_envs)
        deadend_score_norm = reference.new_zeros(num_envs)

        if bool(valid_mask.any().item()):
            sectors_m_clipped = TrackTensorBridge._reduce_to_16(
                clipped_m,
                valid,
                reference,
                max_value=max_m,
            )
            sectors_norm = (sectors_m_clipped / max_m).clamp(0.0, 1.0)

            front_m = sectors_m_clipped[:, 6:10].mean(dim=1)
            left_m = sectors_m_clipped[:, 0:6].mean(dim=1)
            right_m = sectors_m_clipped[:, 10:16].mean(dim=1)
            wall_close_flag, deadend_score_norm, deadend_flag, best_free_m = compute_nav_flags_from_meter(
                front_m,
                left_m,
                right_m,
                nav_cfg,
            )
            side_best_m = torch.maximum(left_m, right_m)

            front_norm = (front_m / max_m).clamp(0.0, 1.0)
            left_norm = (left_m / max_m).clamp(0.0, 1.0)
            right_norm = (right_m / max_m).clamp(0.0, 1.0)
            best_free_norm = (best_free_m / max_m).clamp(0.0, 1.0)
            min_clear = sectors_norm.min(dim=1).values
            mean_clear = sectors_norm.mean(dim=1)
            best_idx = sectors_m_clipped.argmax(dim=1).to(reference.dtype)
            angle = best_idx / 16.0 * 6.283185307179586 - 3.141592653589793

            gate = valid_mask.to(reference.dtype)
            source_norm = nav.new_full(
                (num_envs,),
                float(source_id) / TrackTensorBridge.SOURCE_ID_SCALE,
            )

            nav[:, NavFeatureLayout.NAV_VALID] = gate
            nav[:, NavFeatureLayout.NAV_TENSOR_AVAILABLE] = gate
            nav[:, NavFeatureLayout.NAV_OBJECT_EXISTS] = float(object_exists)
            nav[:, NavFeatureLayout.NAV_SOURCE_ID_NORM] = torch.where(
                valid_mask,
                source_norm,
                torch.zeros_like(source_norm),
            )

            nav[:, NavFeatureLayout.FRONT_CLEARANCE_NORM] = torch.where(valid_mask, front_norm, torch.zeros_like(front_norm))
            nav[:, NavFeatureLayout.LEFT_CLEARANCE_NORM] = torch.where(valid_mask, left_norm, torch.zeros_like(left_norm))
            nav[:, NavFeatureLayout.RIGHT_CLEARANCE_NORM] = torch.where(valid_mask, right_norm, torch.zeros_like(right_norm))
            nav[:, NavFeatureLayout.MIN_CLEARANCE_NORM] = torch.where(valid_mask, min_clear, torch.zeros_like(min_clear))
            nav[:, NavFeatureLayout.MEAN_CLEARANCE_NORM] = torch.where(valid_mask, mean_clear, torch.zeros_like(mean_clear))

            nav[:, NavFeatureLayout.BLOCKED_FRONT_FLAG] = torch.where(
                valid_mask,
                (front_m <= float(nav_cfg.front_blocked_threshold_m)).to(reference.dtype),
                torch.zeros_like(front_norm),
            )
            nav[:, NavFeatureLayout.WALL_CLOSE_FLAG] = torch.where(
                valid_mask,
                wall_close_flag.to(dtype=reference.dtype),
                torch.zeros_like(front_norm),
            )
            nav[:, NavFeatureLayout.DEADEND_SCORE] = torch.where(
                valid_mask,
                deadend_score_norm,
                torch.zeros_like(front_norm),
            )

            nav[:, NavFeatureLayout.BEST_FREE_DIR_SIN] = torch.where(valid_mask, torch.sin(angle), torch.zeros_like(angle))
            nav[:, NavFeatureLayout.BEST_FREE_DIR_COS] = torch.where(valid_mask, torch.cos(angle), torch.zeros_like(angle))
            nav[:, NavFeatureLayout.BEST_FREE_CLEARANCE_NORM] = torch.where(
                valid_mask,
                best_free_norm,
                torch.zeros_like(front_norm),
            )

            for i in range(16):
                nav[:, NavFeatureLayout.SECTOR_CLEARANCE_START + i] = torch.where(
                    valid_mask,
                    sectors_norm[:, i],
                    torch.zeros_like(sectors_norm[:, i]),
                )
                nav[:, NavFeatureLayout.SECTOR_BLOCKED_START + i] = torch.where(
                    valid_mask,
                    (sectors_m_clipped[:, i] <= float(nav_cfg.front_blocked_threshold_m)).to(reference.dtype),
                    torch.zeros_like(sectors_norm[:, i]),
                )

        nav = TrackTensorBridge._sanitize_features(nav)

        stats = {
            "p5a2_raycaster_nav_valid_ratio": valid_mask.float().mean(),
            "p5a2_nav_tensor_available_ratio": tensor_mask.float().mean(),
            "p5a2_nav_object_exists_ratio": torch.full((num_envs,), bool(object_exists), dtype=torch.float32, device=reference.device).mean(),
            "p5a2_nav_valid_ray_ratio": valid_ray_ratio.mean(),
            "p5a2_nav_signal_ratio": signal_ratio.mean(),
            "p5a2_nav_finite_ratio": finite.float().mean(),
            "p5a2_nav_path_id": float(path_id) if bool(valid_mask.any().item()) else 0.0,
            "p5a2_nav_front_mean": nav[:, NavFeatureLayout.FRONT_CLEARANCE_NORM].mean(),
            "p5a2_nav_left_mean": nav[:, NavFeatureLayout.LEFT_CLEARANCE_NORM].mean(),
            "p5a2_nav_right_mean": nav[:, NavFeatureLayout.RIGHT_CLEARANCE_NORM].mean(),
            "p5a2_nav_deadend_mean": nav[:, NavFeatureLayout.DEADEND_SCORE].mean(),
            "p72f_front_clearance_m_mean": front_m.mean(),
            "p72f_left_clearance_m_mean": left_m.mean(),
            "p72f_right_clearance_m_mean": right_m.mean(),
            "p72f_best_free_clearance_m_mean": best_free_m.mean(),
            "p7_2f_bridge_front_m_clipped_mean": front_m.mean(),
            "p7_2f_bridge_left_m_clipped_mean": left_m.mean(),
            "p7_2f_bridge_right_m_clipped_mean": right_m.mean(),
            "p7_2f_bridge_front_norm_mean": front_norm.mean(),
            "p7_2f_bridge_left_norm_mean": left_norm.mean(),
            "p7_2f_bridge_right_norm_mean": right_norm.mean(),
            "p7_2f_bridge_wall_close_flag_ratio": nav[:, NavFeatureLayout.WALL_CLOSE_FLAG].mean(),
            "p7_2f_bridge_deadend_score_mean": nav[:, NavFeatureLayout.DEADEND_SCORE].mean(),
            "p7_2f_bridge_deadend_flag_ratio": (
                nav[:, NavFeatureLayout.DEADEND_SCORE] >= float(nav_cfg.deadend_score_threshold)
            ).to(reference.dtype).mean(),
            "p7_2f_bridge_nan_ratio": sanitize_stats["nan_ratio"],
            "p7_2f_bridge_inf_ratio": sanitize_stats["inf_ratio"],
            "p7_2f_bridge_huge_ratio": sanitize_stats["huge_ratio"],
            "p7_2f_bridge_miss_ratio": sanitize_stats["miss_ratio"],
            "p7_2f_bridge_signal_ratio": sanitize_stats["signal_ratio"],
        }
        if extra_stats:
            stats.update(extra_stats)

        if not bool(valid_mask.any().item()):
            return TrackTensorBridge._empty_nav_contract(
                reference,
                num_envs,
                object_exists,
                "valid_or_signal_ratio_too_low",
                stats_override=stats,
            )

        return NavContractResult(
            nav_features=nav,
            valid_mask=valid_mask,
            tensor_mask=tensor_mask,
            object_mask=torch.full((num_envs,), bool(object_exists), dtype=torch.bool, device=reference.device),
            source_id=source_id,
            path_id=path_id,
            stats=stats,
            attrs=(),
            failure_reason="ok",
        )

    @staticmethod
    def _reduce_to_16(scan, valid, reference, max_value=1.0):
        num_envs, num_rays = scan.shape
        sector_count = 16

        pad = (sector_count - (num_rays % sector_count)) % sector_count
        if pad > 0:
            scan = torch.cat([scan, scan.new_zeros((num_envs, pad))], dim=1)
            valid = torch.cat(
                [valid, torch.zeros((num_envs, pad), dtype=torch.bool, device=valid.device)],
                dim=1,
            )

        rays_per_sector = scan.shape[1] // sector_count
        scan = scan.reshape(num_envs, sector_count, rays_per_sector)
        valid = valid.reshape(num_envs, sector_count, rays_per_sector)

        count = valid.float().sum(dim=2).clamp_min(1.0)
        sectors = (scan * valid.float()).sum(dim=2) / count
        fill_value = torch.full_like(sectors, float(max_value))
        sectors = torch.where(valid.any(dim=2), sectors, fill_value)
        sectors = torch.nan_to_num(
            sectors,
            nan=float(max_value),
            posinf=float(max_value),
            neginf=0.0,
        )
        return torch.clamp(sectors, 0.0, float(max_value))

    @staticmethod
    def _stat_float(stats, key, default=0.0):
        value = stats.get(key, default)
        try:
            if isinstance(value, torch.Tensor):
                return float(torch.nan_to_num(value.detach(), nan=0.0, posinf=0.0, neginf=0.0).mean().item())
            return float(value)
        except Exception:
            return float(default)

    def _normalize_hits(hits, reference, num_envs):
        if hits is None:
            return None
        if hits.ndim == 2 and hits.shape[-1] >= 3:
            return hits[:, :3].unsqueeze(0).expand(num_envs, -1, -1)
        if hits.ndim == 3 and hits.shape[0] == num_envs and hits.shape[-1] >= 3:
            return hits[:, :, :3]
        if hits.ndim == 3 and hits.shape[0] == 1 and hits.shape[-1] >= 3:
            return hits[:, :, :3].expand(num_envs, -1, -1)
        return None

    @staticmethod
    def _normalize_origin(origin, reference, num_envs, hits):
        if origin is None or hits is None:
            return None

        num_rays = hits.shape[1]

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

    @staticmethod
    def _empty_nav_contract(reference, num_envs, object_exists, reason, stats_override=None):
        zero = TrackTensorBridge._zeros_mask(reference, num_envs)
        stats = {
            "p5a2_raycaster_nav_valid_ratio": 0.0,
            "p5a2_nav_tensor_available_ratio": 0.0,
            "p5a2_nav_object_exists_ratio": 1.0 if object_exists else 0.0,
            "p5a2_nav_valid_ray_ratio": 0.0,
            "p5a2_nav_signal_ratio": 0.0,
            "p5a2_nav_finite_ratio": 0.0,
            "p5a2_nav_path_id": 0.0,
            "p5a2_nav_origin_source_id": 0.0,
            "p5a2_ray_hits_finite_ratio": 0.0,
            "p5a2_ray_origin_finite_ratio": 0.0,
            "p5a2_ray_valid_ratio": 0.0,
            "p5a2_ray_hits_nan_ratio": 0.0,
            "p5a2_ray_hits_inf_ratio": 0.0,
            "p5a2_ray_hits_zero_ratio": 0.0,
            "p5a2_nav_distance_xy_mean": 0.0,
            "p5a2_nav_distance_xy_max": 0.0,
            "p5a2_nav_front_mean": 0.0,
            "p5a2_nav_left_mean": 0.0,
            "p5a2_nav_right_mean": 0.0,
            "p5a2_nav_deadend_mean": 0.0,
        }
        if stats_override:
            stats.update(stats_override)
            stats["p5a2_raycaster_nav_valid_ratio"] = 0.0
            stats["p5a2_nav_tensor_available_ratio"] = 0.0
            stats["p5a2_nav_path_id"] = 0.0

        return NavContractResult(
            nav_features=reference.new_zeros((num_envs, NavFeatureLayout.DIM)),
            valid_mask=zero,
            tensor_mask=zero,
            object_mask=torch.full((num_envs,), bool(object_exists), dtype=torch.bool, device=reference.device),
            source_id=TrackTensorBridge.SOURCE_NONE,
            path_id=TrackTensorBridge.NAV_PATH_NONE,
            stats=stats,
            attrs=(),
            failure_reason=reason,
        )

    @staticmethod
    def _extract_nav_signal(env, infos, reference, num_envs):
        sensor = TrackTensorBridge._get_nav_sensor(env)
        if sensor is not None:
            data = getattr(sensor, "data", None)
            hits = getattr(data, "ray_hits_w", None) if data is not None else None
            pos_w = getattr(data, "pos_w", None) if data is not None else None
            if hits is not None and pos_w is not None:
                hits_mask = TrackTensorBridge._as_env_any_mask(hits, reference, num_envs)
                pos_mask = TrackTensorBridge._as_env_any_mask(pos_w, reference, num_envs)
                return hits_mask & pos_mask, TrackTensorBridge.SOURCE_NAV_SENSOR, ()
            attrs = TrackTensorBridge._public_attrs(sensor)
            data_attrs = TrackTensorBridge._public_attrs(data)
            return TrackTensorBridge._zeros_mask(reference, num_envs), TrackTensorBridge.SOURCE_OBJECT_NO_TENSOR, tuple(attrs + data_attrs)

        return TrackTensorBridge._zeros_mask(reference, num_envs), TrackTensorBridge.SOURCE_NONE, ()

    @staticmethod
    def _extract_outcome(infos, reference, num_envs, stage_config):
        value = TrackTensorBridge._first_info_value(infos, TrackTensorBridge._OUTCOME_INFO_KEYS)
        if value is not None:
            tensor, mask = TrackTensorBridge._as_env_scalar(value, reference, num_envs)
            if mask.any().item():
                return tensor > 0.5, mask, TrackTensorBridge.SOURCE_INFOS

        value = TrackTensorBridge._first_info_value(infos, TrackTensorBridge._TERMINAL_GOAL_DIST_INFO_KEYS)
        if value is not None:
            tensor, mask = TrackTensorBridge._as_env_scalar(value, reference, num_envs)
            if mask.any().item():
                threshold = float(getattr(stage_config, "goal_reach_threshold", 0.6))
                return tensor <= threshold, mask, TrackTensorBridge.SOURCE_TERMINAL_GOAL_DIST

        return (
            torch.zeros(num_envs, dtype=torch.bool, device=reference.device),
            TrackTensorBridge._zeros_mask(reference, num_envs),
            TrackTensorBridge.SOURCE_NONE,
        )

    @staticmethod
    def _first_info_value(infos, names):
        value, _ = TrackTensorBridge._first_info_value_with_name(infos, names)
        return value

    @staticmethod
    def _first_info_value_with_name(infos, names):
        if isinstance(infos, dict):
            for name in names:
                if name in infos and infos[name] is not None:
                    return infos[name], name
            extras = infos.get("extras")
            if isinstance(extras, dict):
                for name in names:
                    if name in extras and extras[name] is not None:
                        return extras[name], name
            return None, None

        if isinstance(infos, (list, tuple)):
            for name in names:
                values = []
                has_value = False
                for item in infos:
                    if isinstance(item, dict) and name in item and item[name] is not None:
                        values.append(item[name])
                        has_value = True
                    else:
                        values.append(float("nan"))
                if has_value:
                    return values, name
        return None, None

    @staticmethod
    def _get_nav_sensor(env):
        for candidate in TrackTensorBridge._env_candidates(env):
            scene = getattr(candidate, "scene", None)

            sensors = getattr(scene, "sensors", None) if scene is not None else None

            if isinstance(sensors, dict):
                sensor = sensors.get("nav_scanner")
                if sensor is not None:
                    return sensor

            try:
                sensor = sensors["nav_scanner"] if sensors is not None else None
                if sensor is not None:
                    return sensor
            except Exception:
                pass

            try:
                sensor = getattr(scene, "nav_scanner", None) if scene is not None else None
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
                raw_sensors = getattr(scene, "_sensors", None) if scene is not None else None
                if isinstance(raw_sensors, dict):
                    sensor = raw_sensors.get("nav_scanner")
                    if sensor is not None:
                        return sensor
            except Exception:
                pass

        return None

    @staticmethod
    def _get_robot_data_attr(env, attr):
        value, _ = TrackTensorBridge._get_robot_data_attr_with_name(env, (attr,))
        return value

    @staticmethod
    def _get_robot_data_attr_with_name(env, attrs):
        if env is None:
            return None, None
        if isinstance(attrs, str):
            attrs = (attrs,)

        for candidate in TrackTensorBridge._env_candidates(env):
            objects = [getattr(candidate, "robot", None), getattr(candidate, "asset", None)]
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
                    if articulations:
                        try:
                            objects.append(next(iter(articulations.values())))
                        except Exception:
                            pass

            for obj in objects:
                data = getattr(obj, "data", None)
                for attr in attrs:
                    value = getattr(data, attr, None)
                    if value is not None:
                        return value, attr
        return None, None

    @staticmethod
    def _env_candidates(env):
        if env is None:
            return []

        candidates = []
        seen = set()

        if isinstance(env, (list, tuple, set)):
            pending = list(env)
        else:
            pending = [env]

        attr_names = ("unwrapped", "env", "_env", "base_env", "isaac_env")

        while pending and len(candidates) < 16:
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

    @staticmethod
    def _first_attr_tensor(objects, attr_names, reference):
        for obj in objects:
            if obj is None:
                continue
            for attr in attr_names:
                value = getattr(obj, attr, None)
                if value is None:
                    continue
                try:
                    tensor = TrackTensorBridge._to_tensor(value, reference)
                except Exception:
                    continue
                if tensor.numel() > 0:
                    return tensor
        return None

    @staticmethod
    def _as_env_xy(value, reference, num_envs):
        try:
            tensor = TrackTensorBridge._to_tensor(value, reference)
        except Exception:
            return TrackTensorBridge._zeros_xy(reference, num_envs), TrackTensorBridge._zeros_mask(reference, num_envs)

        if tensor.ndim == 1 and tensor.numel() >= 2:
            selected = tensor[:2].view(1, 2).expand(num_envs, 2)
        elif tensor.ndim >= 2 and tensor.shape[0] == num_envs and tensor.shape[-1] >= 2:
            selected = tensor.reshape(num_envs, -1)[:, :2]
        elif tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[-1] >= 2:
            selected = tensor.reshape(1, -1)[:, :2].expand(num_envs, 2)
        else:
            return TrackTensorBridge._zeros_xy(reference, num_envs), TrackTensorBridge._zeros_mask(reference, num_envs)

        mask = torch.isfinite(selected).all(dim=1)
        selected = torch.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)
        return selected, mask

    @staticmethod
    def _as_env_scalar(value, reference, num_envs):
        try:
            tensor = TrackTensorBridge._to_tensor(value, reference)
        except Exception:
            return TrackTensorBridge._zeros_scalar(reference, num_envs), TrackTensorBridge._zeros_mask(reference, num_envs)

        if tensor.ndim == 0:
            selected = tensor.view(1).expand(num_envs)
        elif tensor.numel() == 1:
            selected = tensor.flatten().expand(num_envs)
        elif tensor.numel() == num_envs:
            selected = tensor.reshape(num_envs)
        elif tensor.ndim >= 2 and tensor.shape[0] == num_envs:
            selected = tensor.reshape(num_envs, -1)[:, 0]
        else:
            return TrackTensorBridge._zeros_scalar(reference, num_envs), TrackTensorBridge._zeros_mask(reference, num_envs)

        mask = torch.isfinite(selected)
        selected = torch.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)
        return selected, mask

    @staticmethod
    def _as_env_any_mask(value, reference, num_envs):
        try:
            tensor = TrackTensorBridge._to_tensor(value, reference)
        except Exception:
            return TrackTensorBridge._zeros_mask(reference, num_envs)

        if tensor.ndim == 0:
            return torch.isfinite(tensor).view(1).expand(num_envs)
        if tensor.shape[0] == num_envs:
            flat = tensor.reshape(num_envs, -1)
            return torch.isfinite(flat).any(dim=1)
        if tensor.numel() == num_envs:
            return torch.isfinite(tensor.flatten())
        if tensor.numel() > 0 and torch.isfinite(tensor).any().item():
            return torch.ones(num_envs, dtype=torch.bool, device=reference.device)
        return TrackTensorBridge._zeros_mask(reference, num_envs)

    @staticmethod
    def _yaw_from_quat(value, reference, num_envs):
        try:
            tensor = TrackTensorBridge._to_tensor(value, reference)
        except Exception:
            return TrackTensorBridge._zeros_scalar(reference, num_envs), TrackTensorBridge._zeros_mask(reference, num_envs), 0.0

        if tensor.ndim == 1 and tensor.numel() >= 4:
            quat = tensor[:4].view(1, 4).expand(num_envs, 4)
        elif tensor.ndim >= 2 and tensor.shape[0] == num_envs and tensor.shape[-1] >= 4:
            quat = tensor.reshape(num_envs, -1)[:, :4]
        elif tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[-1] >= 4:
            quat = tensor.reshape(1, -1)[:, :4].expand(num_envs, 4)
        else:
            return TrackTensorBridge._zeros_scalar(reference, num_envs), TrackTensorBridge._zeros_mask(reference, num_envs), 0.0

        return TrackTensorBridge._yaw_from_quat_tensor(quat, reference, num_envs)

    @staticmethod
    def _yaw_from_quat_tensor(quat, reference, num_envs):
        finite = torch.isfinite(quat).all(dim=1)
        quat = torch.nan_to_num(quat, nan=0.0, posinf=0.0, neginf=0.0)
        norm = torch.linalg.norm(quat, dim=1)
        norm_finite = torch.isfinite(norm)
        norm_gate = (norm >= 0.5) & (norm <= 1.5)
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy.square() + qz.square()),
        )
        yaw_finite = torch.isfinite(yaw)
        mask = finite & norm_finite & norm_gate & yaw_finite
        yaw = torch.nan_to_num(yaw, nan=0.0, posinf=0.0, neginf=0.0)
        norm_mean = 0.0
        valid_norm = norm[norm_finite]
        if valid_norm.numel() > 0:
            norm_mean = float(valid_norm.mean().detach().item())
        return yaw, mask, norm_mean

    @staticmethod
    def _to_reference_tensor(reference_tensor):
        if isinstance(reference_tensor, torch.Tensor):
            return reference_tensor
        return torch.as_tensor(reference_tensor, dtype=torch.float32)

    @staticmethod
    def _to_tensor(value, reference):
        if isinstance(value, torch.Tensor):
            return value.to(device=reference.device, dtype=reference.dtype)
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _zeros_mask(reference, num_envs):
        return torch.zeros(num_envs, dtype=torch.bool, device=reference.device)

    @staticmethod
    def _zeros_scalar(reference, num_envs):
        return reference.new_zeros((num_envs,))

    @staticmethod
    def _zeros_xy(reference, num_envs):
        return reference.new_zeros((num_envs, 2))

    @staticmethod
    def _sanitize_features(features):
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(features, -1.0, 1.0)

    @staticmethod
    def _p5a1_goal_stats(
        goal_features,
        goal_pos_mask,
        robot_pos_mask,
        goal_dist_mask,
        goal_yaw_mask,
        robot_yaw_mask,
        goal_distance,
        dist_norm,
        delta_world,
        dx_body,
        dy_body,
        distance_scale,
    ):
        with torch.no_grad():
            goal = goal_features.detach()
            nan_mask = torch.isnan(goal)
            inf_mask = torch.isinf(goal)
            nonfinite_mask = ~torch.isfinite(goal)
            out_of_range_mask = torch.isfinite(goal) & (goal.abs() > 1.0)
            safe_goal = torch.nan_to_num(goal, nan=0.0, posinf=1.0, neginf=-1.0)

            goal_distance_safe = torch.nan_to_num(
                goal_distance.detach(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)
            dist_norm_safe = torch.nan_to_num(
                dist_norm.detach(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            delta_world_safe = torch.nan_to_num(
                delta_world.detach(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            dx_body_safe = torch.nan_to_num(dx_body.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            dy_body_safe = torch.nan_to_num(dy_body.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            body_mask = goal_dist_mask & robot_yaw_mask
            dist_clipped = goal_distance_safe >= float(distance_scale)

            return {
                "p5a1_goal_slice_abs_mean": safe_goal.abs().mean(),
                "p5a1_goal_valid_flag_mean": goal_pos_mask.float().mean(),
                "p5a1_robot_valid_flag_mean": robot_pos_mask.float().mean(),
                "p5a1_goal_dist_valid_flag_mean": goal_dist_mask.float().mean(),
                "p5a1_goal_yaw_valid_flag_mean": goal_yaw_mask.float().mean(),
                "p5a1_robot_yaw_valid_flag_mean": robot_yaw_mask.float().mean(),
                "p5a1_goal_distance_raw_mean": TrackTensorBridge._masked_mean_tensor(
                    goal_distance_safe,
                    goal_dist_mask,
                ),
                "p5a1_goal_dist_norm_mean": TrackTensorBridge._masked_mean_tensor(
                    dist_norm_safe,
                    goal_dist_mask,
                ),
                "p5a1_goal_dist_clip_ratio": TrackTensorBridge._masked_mean_tensor(
                    dist_clipped.float(),
                    goal_dist_mask,
                ),
                "p5a1_dx_world_abs_mean": TrackTensorBridge._masked_mean_tensor(
                    delta_world_safe[:, 0].abs(),
                    goal_dist_mask,
                ),
                "p5a1_dy_world_abs_mean": TrackTensorBridge._masked_mean_tensor(
                    delta_world_safe[:, 1].abs(),
                    goal_dist_mask,
                ),
                "p5a1_dx_body_abs_mean": TrackTensorBridge._masked_mean_tensor(
                    dx_body_safe.abs(),
                    body_mask,
                ),
                "p5a1_dy_body_abs_mean": TrackTensorBridge._masked_mean_tensor(
                    dy_body_safe.abs(),
                    body_mask,
                ),
                "p5a1_goal_nan_count": nan_mask.float().sum(),
                "p5a1_goal_inf_count": inf_mask.float().sum(),
                "p5a1_goal_sanitized_count": (nonfinite_mask | out_of_range_mask).float().sum(),
            }

    @staticmethod
    def _masked_mean_tensor(values, mask):
        if values is None or mask is None or values.numel() == 0:
            return torch.tensor(0.0)
        mask_f = mask.float()
        return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    @staticmethod
    def _source_name(source_id):
        return TrackTensorBridge.SOURCE_NAMES.get(float(source_id), "unknown")

    @staticmethod
    def _public_attrs(obj):
        if obj is None:
            return []
        try:
            attrs = [name for name in dir(obj) if not name.startswith("_")]
        except Exception:
            return []
        return attrs[:40]

    @staticmethod
    def _maybe_log_sources(logger, source_names):
        if logger is None:
            return
        signature = (
            source_names.get("source_context", "unknown"),
            source_names["goal_source"],
            source_names["robot_source"],
            source_names["goal_yaw_source"],
            source_names["robot_yaw_source"],
            source_names["nav_source"],
            source_names["outcome_source"],
            float(source_names.get("robot_yaw_detail_id", 0.0)),
        )
        if signature in TrackTensorBridge._logged_source_signatures:
            return
        if len(TrackTensorBridge._logged_source_signatures) >= 3:
            return
        TrackTensorBridge._logged_source_signatures.add(signature)
        logger.warning(
            "[P5A-0Bridge] "
            f"source_context={source_names.get('source_context', 'unknown')}, "
            f"goal_source={source_names['goal_source']}, "
            f"robot_source={source_names['robot_source']}, "
            f"goal_yaw_source={source_names['goal_yaw_source']}, "
            f"robot_yaw_source={source_names['robot_yaw_source']}, "
            f"robot_yaw_detail_id={float(source_names.get('robot_yaw_detail_id', 0.0)):.0f}, "
            f"robot_yaw_attr={source_names.get('robot_yaw_selected_attr', 'none')}, "
            f"nav_source={source_names['nav_source']}, "
            f"outcome_source={source_names['outcome_source']}"
        )

    @staticmethod
    def _maybe_log_nav_attrs_warning(logger, attrs):
        if logger is None or TrackTensorBridge._logged_nav_attrs_warning:
            return
        TrackTensorBridge._logged_nav_attrs_warning = True
        logger.warning(
            "[P5A-0BridgeWarning] "
            f"nav_scanner exists but tensor fields not found, attrs={list(attrs)[:40]}"
        )

    @staticmethod
    def _maybe_log_robot_missing_warning(logger):
        if logger is None or TrackTensorBridge._logged_robot_missing_warning:
            return
        TrackTensorBridge._logged_robot_missing_warning = True
        logger.warning("[P5A-0BridgeWarning] robot_pos unavailable; goal_distance disabled.")

    @staticmethod
    def _maybe_log_robot_yaw_drop_scan(logger, env, reference, num_envs, robot_yaw_mask, source_context):
        if logger is None:
            return
        ratio = float(robot_yaw_mask.float().mean().detach().item()) if robot_yaw_mask.numel() > 0 else 0.0
        if ratio >= 0.95:
            TrackTensorBridge._robot_yaw_was_available = True
            return
        if (
            ratio > 0.0
            or not TrackTensorBridge._robot_yaw_was_available
            or TrackTensorBridge._logged_robot_yaw_drop_scan
        ):
            return
        TrackTensorBridge._logged_robot_yaw_drop_scan = True
        logger.warning(
            "[P5A-YawSourceDrop] "
            f"source_context={source_context}, "
            f"robot_yaw_tensor_ratio={ratio:.4f}, "
            f"heading_w={TrackTensorBridge._describe_robot_yaw_attr(env, 'heading_w', reference, num_envs, 'scalar')}, "
            f"yaw={TrackTensorBridge._describe_robot_yaw_attr(env, 'yaw', reference, num_envs, 'scalar')}, "
            f"root_yaw_w={TrackTensorBridge._describe_robot_yaw_attr(env, 'root_yaw_w', reference, num_envs, 'scalar')}, "
            f"root_quat_w={TrackTensorBridge._describe_robot_yaw_attr(env, 'root_quat_w', reference, num_envs, 'quat')}"
        )

    @staticmethod
    def _describe_robot_yaw_attr(env, attr, reference, num_envs, kind):
        value = TrackTensorBridge._get_robot_data_attr(env, attr)
        if value is None:
            return "none"
        shape = getattr(value, "shape", None)
        desc = f"type={type(value).__name__},shape={tuple(shape) if shape is not None else None}"
        if kind == "scalar":
            _, mask = TrackTensorBridge._as_env_scalar(value, reference, num_envs)
            return f"{desc},finite_ratio={TrackTensorBridge._mask_ratio(mask):.4f}"
        if kind == "quat":
            _, mask, norm_mean = TrackTensorBridge._yaw_from_quat(value, reference, num_envs)
            return f"{desc},valid_ratio={TrackTensorBridge._mask_ratio(mask):.4f},norm_mean={norm_mean:.4f}"
        return desc

    @staticmethod
    def _mask_ratio(mask):
        if mask is None or mask.numel() <= 0:
            return 0.0
        return float(mask.float().mean().detach().item())
