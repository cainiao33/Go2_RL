# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PolicyObservationProcess — custom policy observation processor.
PolicyObservationProcess — 自定义 policy 观测处理器。

obs layout: [proprio(45) | height_scan(256)] → 301 dim
观测布局：[proprio(45) | height_scan(256)] → 301 维

Extending to track terrain (optional):
    In track terrain the environment additionally provides the following
    read-only attributes (not available in standard terrain):
      - env.goal_positions  (num_envs, 3)  — exit position in world frame
      - env.goal_yaw        (num_envs,)    — exit heading in world frame
    The environment always exposes these scene sensors (available in both
    standard and track terrains, accessed via env.scene.sensors["<name>"]):
      - "height_scanner"  — default forward ground-clearance scan
      - "nav_scanner"     — forward-looking occlusion scan (wider range,
                             suited for obstacle avoidance / turning)
    Players can construct their own obs from these inputs. After appending,
    update the Stage config (observation dim) and model input dim accordingly.

扩展到 track 地形时（可选）：
    track 地形下，环境会额外提供以下只读属性（standard 地形没有）：
      - env.goal_positions  (num_envs, 3)  — 出口在世界坐标系下的 3D 位置
      - env.goal_yaw        (num_envs,)    — 出口在世界坐标系下的朝向
    环境在两种地形下都会通过 env.scene.sensors["<name>"] 提供以下传感器：
      - "height_scanner"  — 默认前方地面高度扫描
      - "nav_scanner"     — 前瞻遮挡扫描（范围更大，适合避障 / 转向判断）
    选手可从这些属性和传感器自行构造 obs。
    拼接后需同步修改 Stage 的观测维度和 model 输入维度。
"""

import torch

from tools.base_env.observation_process import ObservationProcess
from agent_ppo.conf.conf import Config
from agent_ppo.feature.nav_command import apply_track_nav_command_to_policy_obs


def _is_reset_bootstrap_env(env_sources) -> bool:
    """Return True during env.reset() / first observation bootstrap.

    During reset, goal_positions / robot root state may not be ready yet.
    We should not fail-fast in this phase.
    """
    if env_sources is None:
        return False

    try:
        from agent_ppo.feature.track_tensor_bridge import TrackTensorBridge

        candidates = TrackTensorBridge._env_candidates(env_sources)
    except Exception:
        if not isinstance(env_sources, (list, tuple)):
            candidates = (env_sources,)
        else:
            candidates = env_sources

    for env in candidates:
        if env is None:
            continue

        try:
            ep_len = getattr(env, "episode_length_buf", None)
            if ep_len is not None and torch.is_tensor(ep_len):
                if bool((ep_len <= 0).all().item()):
                    return True
        except Exception:
            pass

    return False


class PolicyObservationProcess(ObservationProcess):
    """Policy observation processor with height_scan and optional goal obs.

    带 height_scan 和可选 goal obs 的 policy 观测处理器。

    Stage1 (standard): proprio(45) + height_scan(256) = 301 dim
    Stage2 (track):     proprio(45) + height_scan(256) = 301 dim
    """

    target_group = "policy"

    def process(self):
        """Compute policy observation.

        计算 policy 观测。

        Standard: obs = 301 dim
        Track:    obs = 301 dim
        """
        obs = self.default_observation()

        # Track 地形下拼接目标点特征（goal 在机器人本体坐标系下的相对位置）
        # Track navigation keeps the Standard observation ABI and turns goal/nav
        # inputs into velocity commands. This overwrites
        # obs[:, 6:9] and command_manager["base_velocity"] in TrackConfig only,
        # so rewards and actor input use the same goal-directed command.
        env_sources = []
        for name in ("env", "_env", "base_env", "unwrapped", "isaac_env"):
            value = getattr(self, name, None)
            if value is not None:
                env_sources.append(value)

        # ObservationProcess may proxy raw environment attributes itself.
        env_sources.append(self)

        if not _is_reset_bootstrap_env(tuple(env_sources)):
            obs = apply_track_nav_command_to_policy_obs(
                obs,
                tuple(env_sources),
                stage=Config.CURRENT,
                logger=getattr(self, "logger", None),
            )

        return obs
