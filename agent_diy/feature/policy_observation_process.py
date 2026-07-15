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

from tools.base_env.observation_process import ObservationProcess
from agent_ppo.conf.conf import Config
from agent_ppo.feature.nav_command import apply_track_nav_command_to_policy_obs


class PolicyObservationProcess(ObservationProcess):
    """Policy observation processor with height_scan and optional goal obs.

    带 height_scan 和可选 goal obs 的 policy 观测处理器。

    Stage1 (standard): proprio(45) + height_scan(256) = 301 dim
    Stage2 (track):     proprio(45) + height_scan(256) + goal(4) = 305 dim
    """

    target_group = "policy"

    def process(self):
        """Compute policy observation.

        计算 policy 观测。

        Standard: obs = 301 dim
        Track:    obs = 301 + 4 (goal) = 305 dim
        """
        obs = self.default_observation()

        # Track 地形下拼接目标点特征（goal 在机器人本体坐标系下的相对位置）
        if self._get_num_goal_obs() > 0:
            goal_obs = self.goal_position_in_robot_frame()
            obs = self.concatenate_terms(obs, goal_obs)

        # Last-minute Track migration: keep the trained Standard locomotion policy
        # and turn navigation into a velocity-command generator.  This overwrites
        # obs[:, 6:9] and command_manager["base_velocity"] in TrackConfig only,
        # so rewards and actor input use the same goal-directed command.
        obs = apply_track_nav_command_to_policy_obs(
            obs,
            getattr(self, "env", getattr(self, "_env", None)),
            stage=Config.CURRENT,
            logger=getattr(self, "logger", None),
        )

        return obs
