#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    #
    # Note: this builder only keeps metrics that are unique to algorithm training
    # (loss-series metrics, episode_reward, track traversal progress).
    # Other reward_* metrics (velocity tracking, posture, gait, navigation rewards, etc.)
    # are rendered by the project-side tools/conf/monitor_default.yaml and
    # tools/conf/monitor_default_track.yaml, and are no longer redefined here,
    # to avoid duplicated panels with the same name in the final merged dashboard.
    #
    # 注意：本 builder 只保留算法训练独有的指标（loss 类、episode_reward、赛道穿越进度）。
    # 其余 reward_* 指标（速度跟踪、姿态、步态、导航奖励等）由项目侧
    # tools/conf/monitor_default.yaml 与 tools/conf/monitor_default_track.yaml 负责展示，
    # 这里不再重复定义，避免最终合并后的监控面板出现同名指标重复绘制。

    Returns:
        dict: monitor configuration dictionary
        返回值：监控配置字典
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("四足机器人导航")
        # ==============================================================
        # Group 1: Algorithm training loss metrics (unique to this builder, not covered by yaml)
        # Group 1: 算法训练损失指标（本 builder 独有，yaml 未覆盖）
        # ==============================================================
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        .end_group()
        # ==============================================================
        # Group 2: Reward metrics (examples, players can add more reward panels as needed)
        # Group 2: Reward 指标（示例，选手可按需补充更多 reward 面板）
        # ==============================================================
        .add_group(group_name="奖励指标", group_name_en="reward")
        .add_panel(name="线速度跟踪奖励", name_en="reward_track_lin_vel_xy", type="line")
            .add_metric(metrics_name="reward_track_lin_vel_xy",
                        expr="avg(reward_track_lin_vel_xy{})")
            .end_panel()
        .add_panel(name="动作平滑惩罚", name_en="reward_action_smoothness", type="line")
            .add_metric(metrics_name="reward_action_smoothness",
                        expr="avg(reward_action_smoothness{})")
            .end_panel()
        .add_panel(name="脚部滞空时间", name_en="reward_feet_air_time", type="line")
            .add_metric(metrics_name="reward_feet_air_time",
                        expr="avg(reward_feet_air_time{})")
            .end_panel()
        .add_panel(name="站立姿态惩罚", name_en="reward_stand_still", type="line")
            .add_metric(metrics_name="reward_stand_still",
                        expr="avg(reward_stand_still{})")
            .end_panel()
        .add_panel(name="接近目标奖励", name_en="reward_approach_goal", type="line")
            .add_metric(metrics_name="reward_approach_goal",
                        expr="avg(reward_approach_goal{})")
            .end_panel()
        .add_panel(name="目标方向奖励", name_en="reward_goal_direction", type="line")
            .add_metric(metrics_name="reward_goal_direction",
                        expr="avg(reward_goal_direction{})")
            .end_panel()
        .add_panel(name="导航时间惩罚", name_en="reward_navigation_time", type="line")
            .add_metric(metrics_name="reward_navigation_time",
                        expr="avg(reward_navigation_time{})")
            .end_panel()
        .end_group()
        .build()
    )
    return config_dict
