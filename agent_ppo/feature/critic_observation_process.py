# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CriticObservationProcess — custom critic observation processor.
CriticObservationProcess — 自定义 critic 观测处理器。

critic obs layout: [critic_proprio(60) | height_scan(256)] → 316 dim
critic 观测布局：[critic_proprio(60) | height_scan(256)] → 316 维

When extending to track terrain, please refer to the extension guide in
policy_observation_process.py; the critic observation must stay in sync
with the policy on the task-information convention.
扩展到 track 地形时，请参考 policy_observation_process.py 的扩展指引；
critic 观测需保持与 policy 同步的任务信息约定。
"""

from tools.base_env.observation_process import ObservationProcess
from agent_ppo.conf.conf import Config
from agent_ppo.feature.nav_command import apply_track_nav_command_to_critic_obs
from agent_ppo.feature.policy_observation_process import _is_reset_bootstrap_env


class CriticObservationProcess(ObservationProcess):
    """Critic observation processor with optional goal obs.

    与 policy 观测保持同步的 critic 观测处理器，可选拼接 goal obs。

    Stage1 (standard): critic_proprio(60) + height_scan(256) = 316 dim
    Stage2 (track):     critic_proprio(60) + height_scan(256) = 316 dim
    """

    target_group = "critic"

    def process(self):
        """Compute critic observation.

        计算 critic 观测。

        Standard: critic_obs = 316 dim
        Track:    critic_obs = 316 dim
        """
        obs = self.default_observation()

        # Track 地形下拼接与 policy 相同的目标点特征
        # Do not append goal observations. Keep the critic command slice and
        # reward command_manager synchronized with the policy-side command.
        env_sources = []
        for name in ("env", "_env", "base_env", "unwrapped", "isaac_env"):
            value = getattr(self, name, None)
            if value is not None:
                env_sources.append(value)

        # ObservationProcess may proxy raw environment attributes itself.
        env_sources.append(self)

        if not _is_reset_bootstrap_env(tuple(env_sources)):
            obs = apply_track_nav_command_to_critic_obs(
                obs,
                tuple(env_sources),
                stage=Config.CURRENT,
                logger=getattr(self, "logger", None),
            )

        return obs
