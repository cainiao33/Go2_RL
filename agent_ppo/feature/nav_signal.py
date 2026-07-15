# -*- coding: UTF-8 -*-
"""P7-2F canonical navigation signal helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from agent_ppo.feature.feature_layout import NavFeatureLayout


@dataclass(frozen=True)
class NavMeterConfig:
    max_clearance_m: float = 2.5
    wall_close_threshold_m: float = 0.45
    front_blocked_threshold_m: float = 0.45
    deadend_free_threshold_m: float = 0.50
    deadend_score_threshold: float = 0.80
    min_signal_ratio: float = 0.05

    @classmethod
    def from_stage(cls, stage: Any) -> "NavMeterConfig":
        max_m = float(
            getattr(
                stage,
                "p7_2f_nav_max_clearance_m",
                getattr(stage, "p5a2_nav_max_clearance_m", cls.max_clearance_m),
            )
        )
        return cls(
            max_clearance_m=max(max_m, 1.0e-6),
            wall_close_threshold_m=float(
                getattr(
                    stage,
                    "p7_2f_wall_close_threshold_m",
                    getattr(stage, "p5b2_wall_close_m", cls.wall_close_threshold_m),
                )
            ),
            front_blocked_threshold_m=float(
                getattr(
                    stage,
                    "p7_2f_front_blocked_threshold_m",
                    getattr(stage, "p5b2_front_block_m", cls.front_blocked_threshold_m),
                )
            ),
            deadend_free_threshold_m=float(
                getattr(
                    stage,
                    "p7_2f_deadend_free_threshold_m",
                    getattr(stage, "p5b2_deadend_best_free_m", cls.deadend_free_threshold_m),
                )
            ),
            deadend_score_threshold=float(
                getattr(stage, "p7_2f_deadend_score_threshold", cls.deadend_score_threshold)
            ),
            min_signal_ratio=float(
                getattr(
                    stage,
                    "p7_2f_min_signal_ratio",
                    getattr(stage, "p5a2_nav_min_signal_ratio", cls.min_signal_ratio),
                )
            ),
        )


@dataclass
class NavRuntimeSignal:
    nav_valid: torch.Tensor
    nav_tensor_available: torch.Tensor
    front_m_clipped: torch.Tensor
    left_m_clipped: torch.Tensor
    right_m_clipped: torch.Tensor
    best_free_m_clipped: torch.Tensor
    front_norm: torch.Tensor
    left_norm: torch.Tensor
    right_norm: torch.Tensor
    best_free_norm: torch.Tensor
    wall_close_flag: torch.Tensor
    deadend_score: torch.Tensor
    deadend_flag: torch.Tensor
    invalid_nav_nonzero_ratio: torch.Tensor
    wall_flag_mismatch_ratio: torch.Tensor
    deadend_flag_mismatch_ratio: torch.Tensor
    deadend_score_reconstruct_error: torch.Tensor
    norm_meter_reconstruct_error: torch.Tensor


def safe_ratio(num: torch.Tensor, denom: torch.Tensor | float, default: float = 0.0) -> torch.Tensor:
    denom_t = denom if isinstance(denom, torch.Tensor) else num.new_tensor(float(denom))
    return torch.where(denom_t > 0.0, num / denom_t.clamp_min(1.0e-6), num.new_tensor(float(default)))


def sanitize_distance_m(
    distance_m: torch.Tensor,
    valid_mask: torch.Tensor | None,
    cfg: NavMeterConfig,
    huge_multiple: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Sanitize raw meter rays before sector reduction.

    Invalid, missing, huge, NaN, and Inf rays are filled with max clearance and
    removed from the aggregation mask so they cannot become fake 0m obstacles.
    """
    raw = distance_m.to(dtype=torch.float32)
    finite = torch.isfinite(raw)
    nan_mask = torch.isnan(raw)
    inf_mask = torch.isinf(raw)
    negative_mask = finite & (raw < 0.0)
    zero_mask = finite & (raw <= 1.0e-6)
    huge_limit = max(float(cfg.max_clearance_m) * float(huge_multiple), float(cfg.max_clearance_m))
    huge_mask = finite & (raw > huge_limit)

    if valid_mask is None:
        valid = finite & (~negative_mask) & (~zero_mask) & (~huge_mask)
    else:
        valid = valid_mask.to(dtype=torch.bool) & finite & (~negative_mask) & (~zero_mask) & (~huge_mask)

    max_value = raw.new_tensor(float(cfg.max_clearance_m))
    sanitized = torch.where(valid, raw.clamp_min(0.0), max_value)
    sanitized = torch.nan_to_num(
        sanitized,
        nan=float(cfg.max_clearance_m),
        posinf=float(cfg.max_clearance_m),
        neginf=0.0,
    )

    stats = {
        "nan_ratio": nan_mask.float().mean(),
        "inf_ratio": inf_mask.float().mean(),
        "huge_ratio": huge_mask.float().mean(),
        "negative_ratio": negative_mask.float().mean(),
        "zero_ratio": zero_mask.float().mean(),
        "miss_ratio": (~valid).float().mean(),
        "signal_ratio": (valid & (sanitized < float(cfg.max_clearance_m))).float().mean(),
    }
    return sanitized, valid, stats


def reconstruct_meter_from_norm(
    front_norm: torch.Tensor,
    left_norm: torch.Tensor,
    right_norm: torch.Tensor,
    cfg: NavMeterConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_m = float(cfg.max_clearance_m)
    front = torch.nan_to_num(front_norm, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0) * max_m
    left = torch.nan_to_num(left_norm, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0) * max_m
    right = torch.nan_to_num(right_norm, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0) * max_m
    return front, left, right


def compute_nav_flags_from_meter(
    front_m_clipped: torch.Tensor,
    left_m_clipped: torch.Tensor,
    right_m_clipped: torch.Tensor,
    cfg: NavMeterConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_free_m = torch.maximum(front_m_clipped, torch.maximum(left_m_clipped, right_m_clipped))
    wall_close_flag = (front_m_clipped <= float(cfg.wall_close_threshold_m)).to(dtype=torch.float32)
    deadend_score = 1.0 - torch.clamp(max_free_m / float(cfg.max_clearance_m), 0.0, 1.0)
    deadend_flag = (max_free_m <= float(cfg.deadend_free_threshold_m)).to(dtype=torch.float32)
    return wall_close_flag, deadend_score.clamp(0.0, 1.0), deadend_flag, max_free_m


def parse_nav_runtime_from_obs(obs: torch.Tensor, cfg: NavMeterConfig, layout=NavFeatureLayout) -> NavRuntimeSignal:
    nav = obs
    nav_valid = nav[..., layout.NAV_VALID] > 0.5
    nav_tensor = nav[..., layout.NAV_TENSOR_AVAILABLE] > 0.5
    effective = nav_valid & nav_tensor

    front_norm = torch.nan_to_num(nav[..., layout.FRONT_CLEARANCE_NORM], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    left_norm = torch.nan_to_num(nav[..., layout.LEFT_CLEARANCE_NORM], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    right_norm = torch.nan_to_num(nav[..., layout.RIGHT_CLEARANCE_NORM], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    best_free_norm_obs = torch.nan_to_num(
        nav[..., layout.BEST_FREE_CLEARANCE_NORM],
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    deadend_obs = torch.nan_to_num(nav[..., layout.DEADEND_SCORE], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    wall_obs = (torch.nan_to_num(nav[..., layout.WALL_CLOSE_FLAG], nan=0.0, posinf=1.0, neginf=0.0) > 0.5).to(torch.float32)

    front_m, left_m, right_m = reconstruct_meter_from_norm(front_norm, left_norm, right_norm, cfg)
    wall_m, deadend_m, deadend_flag_m, best_free_m = compute_nav_flags_from_meter(front_m, left_m, right_m, cfg)
    best_free_norm = (best_free_m / float(cfg.max_clearance_m)).clamp(0.0, 1.0)

    nonzero = (
        front_norm.abs()
        + left_norm.abs()
        + right_norm.abs()
        + deadend_obs.abs()
        + wall_obs.abs()
        + best_free_norm_obs.abs()
    ) > 1.0e-6
    invalid_nav_nonzero = ((~effective) & nonzero).to(torch.float32).mean()

    valid_denom = effective.to(torch.float32).sum().clamp_min(1.0)
    wall_mismatch = ((wall_obs != (wall_m > 0.5).to(torch.float32)) & effective).to(torch.float32).sum() / valid_denom
    deadend_flag_obs = (deadend_obs >= float(cfg.deadend_score_threshold)).to(torch.float32)
    deadend_mismatch = ((deadend_flag_obs != (deadend_flag_m > 0.5).to(torch.float32)) & effective).to(torch.float32).sum() / valid_denom
    deadend_err = (torch.abs(deadend_obs - deadend_m) * effective.to(torch.float32)).sum() / valid_denom
    recon_err = (
        torch.abs(front_m - front_norm * float(cfg.max_clearance_m))
        + torch.abs(left_m - left_norm * float(cfg.max_clearance_m))
        + torch.abs(right_m - right_norm * float(cfg.max_clearance_m))
    ) / 3.0
    recon_err = (recon_err * effective.to(torch.float32)).sum() / valid_denom

    zero = torch.zeros_like(front_m)
    front_m = torch.where(effective, front_m, zero)
    left_m = torch.where(effective, left_m, zero)
    right_m = torch.where(effective, right_m, zero)
    best_free_m = torch.where(effective, best_free_m, zero)
    front_norm = torch.where(effective, front_norm, zero)
    left_norm = torch.where(effective, left_norm, zero)
    right_norm = torch.where(effective, right_norm, zero)
    best_free_norm = torch.where(effective, best_free_norm, zero)
    wall_m = torch.where(effective, wall_m, zero)
    deadend_m = torch.where(effective, deadend_m, zero)
    deadend_flag_m = torch.where(effective, deadend_flag_m, zero)

    return NavRuntimeSignal(
        nav_valid=nav_valid,
        nav_tensor_available=nav_tensor,
        front_m_clipped=front_m,
        left_m_clipped=left_m,
        right_m_clipped=right_m,
        best_free_m_clipped=best_free_m,
        front_norm=front_norm,
        left_norm=left_norm,
        right_norm=right_norm,
        best_free_norm=best_free_norm,
        wall_close_flag=wall_m,
        deadend_score=deadend_m,
        deadend_flag=deadend_flag_m,
        invalid_nav_nonzero_ratio=invalid_nav_nonzero.detach(),
        wall_flag_mismatch_ratio=wall_mismatch.detach(),
        deadend_flag_mismatch_ratio=deadend_mismatch.detach(),
        deadend_score_reconstruct_error=deadend_err.detach(),
        norm_meter_reconstruct_error=recon_err.detach(),
    )
