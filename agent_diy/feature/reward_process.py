# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
RewardProcess — custom reward processor (lite baseline).
RewardProcess — 自定义奖励处理器（lite baseline）。

This file only ships two example rewards:
    1. _reward_reach_goal       — goal-reaching judgment (0.6 m)
    2. _reward_forward_velocity — forward velocity reward (dense, demonstrates reward writing style)
本文件仅预置两个示例 reward：
    1. _reward_reach_goal       — 赛题到达判定（0.6 m）
    2. _reward_forward_velocity — 前向速度奖励（dense，展示 reward 写法）

Other generic locomotion rewards (track_lin_vel_xy / joint_acc / action_rate, etc.)
are inherited from RewardProcessBase (see tools/base_env/base_reward.py).
Players only need to activate them in the TOML; no need to re-implement them here.
其余通用 locomotion reward（track_lin_vel_xy / joint_acc / action_rate 等）
继承自 RewardProcessBase（见 tools/base_env/base_reward.py），
选手在 TOML 中激活即可，无需在此重复实现。

If players need to train a navigation policy, please add more rewards in this file.
选手若需训练导航策略，请在本文件自行添加更多 reward。
"""

import torch

from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):
    def _reward_reach_goal(self, threshold: float = 0.6):
        """Reward for reaching the maze exit (returns 1.0 when distance < 0.6 m).
        到达迷宫出口奖励（distance < 0.6 m 时返回 1.0）。

        Note:
            The threshold must match the threshold of _goal_reached_termination
            in tools/unitree_rl_lab/.../velocity_env_cfg.py (currently 0.6 m),
            otherwise a "termination-reward dead zone" will appear.
            threshold 必须与 tools/unitree_rl_lab/.../velocity_env_cfg.py 中
            _goal_reached_termination 的 threshold 一致（当前 0.6 m），
            否则会产生"终止-奖励死区"。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()

    def _reward_forward_velocity(self):
        """Forward velocity reward: x-direction velocity in the robot body frame (the larger the better).
        前向速度奖励：机器人本体坐标系下 x 方向速度（越大越好）。

        This is an example reward that demonstrates how to read the robot state and
        build a dense signal.
        示例性 reward，展示如何读取机器人状态并构造 dense signal。
        """
        robot = self._get_robot_asset()
        return robot.data.root_lin_vel_b[:, 0]

    # =====================================================================
    # 冠军方案迁移奖励（Phase 1）
    # 以下奖励项借鉴冠军方案的 reward 设计，用于提升 Standard 模式下的
    # 运动稳定性、能效和步态质量。
    # =====================================================================

    def _reward_action_smoothness(self):
        """二阶动作平滑惩罚：a_t - 2*a_{t-1} + a_{t-2}。

        惩罚动作抖动，鼓励平滑连续的关节控制。
        冠军方案中该项权重为 -0.02。

        实现说明：
        - 当前动作 a_t 从 action_manager.action 获取
        - 历史动作通过 env._last_action / env._last_last_action 手动缓存
          （ActionManager 不提供 last_action 属性）
        - 每次 env.reset 后自动清零历史，避免跨 episode 串扰
        """
        current_action = self.env.action_manager.action  # a_t

        # 首次调用：初始化历史缓存
        if not hasattr(self.env, '_last_action') or self.env._last_action is None:
            self.env._last_action = current_action.clone()
            self.env._last_last_action = current_action.clone()
            return torch.zeros(self.env.num_envs, device=self.env.device)

        # 二阶差分：a_t - 2*a_{t-1} + a_{t-2}
        jerk = current_action - 2 * self.env._last_action + self.env._last_last_action
        reward = torch.sum(jerk.pow(2), dim=1)

        # 处理 episode reset：reset 的 env 本步不计算惩罚（历史来自上一个 episode）
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        reward[reset_mask] = 0.0

        # 更新历史缓存：shift last -> last_last, current -> last
        self.env._last_last_action[:] = self.env._last_action[:]
        self.env._last_action[:] = current_action[:]

        return reward

    def _reward_feet_air_time(self, command_name: str = "base_velocity", threshold: float = 0.5):
        """奖励长步幅（移动时脚部滞空时间超过阈值）。

        冠军方案中该项权重为 1.0。
        仅在有速度命令时生效，零命令时不给奖励。

        Args:
            command_name: 命令项名称。
            threshold: 最小滞空时间阈值（秒）。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]

        # 检查是否启用了 track_air_time
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError(
                "Activate ContactSensor's track_air_time! "
                "feet_air_time reward requires track_air_time=True in ContactSensor config."
            )

        # 当前刚接触地面的脚（first_contact）
        first_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] == 0.0
        # 上一次滞空时间
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

        # 奖励 = 滞空时间超过阈值的部分，仅在首次接触时给
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

        # 零命令时不给奖励（静止状态不应鼓励抬脚）
        is_moving = torch.norm(self.env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        return reward * is_moving.float()

    def _reward_stand_still(self, command_name: str = "base_velocity", scale: float = 5.0):
        """零命令时惩罚关节偏离默认姿态。

        当机器人没有收到运动命令时，惩罚关节位置偏离默认值，
        鼓励机器人保持站立姿态。冠军方案中该项权重为 -0.5~1.0。

        Args:
            command_name: 命令项名称。
            scale: 零命令时的惩罚缩放因子。
        """
        robot = self._get_robot_asset()

        # 获取速度命令
        cmd = torch.linalg.norm(self.env.command_manager.get_command(command_name)[:, :2], dim=1)

        # 关节位置偏离默认姿态的程度
        joint_deviation = torch.linalg.norm(
            robot.data.joint_pos - robot.data.default_joint_pos, dim=1
        )

        # 有命令时正常惩罚偏离，无命令时加大惩罚
        is_stationary = cmd < 0.1
        return torch.where(
            is_stationary,
            scale * joint_deviation,  # 零命令：大幅惩罚偏离
            joint_deviation,           # 有命令：正常惩罚偏离
        )

    # =====================================================================
    # 评测对齐奖励
    # 以下奖励项用于对齐评测环境的 reward 配置。
    # =====================================================================

    def _reward_base_linear_velocity(self):
        """惩罚基座 XY 方向线速度（评测环境使用）。

        对齐评测 base_linear_velocity 权重 -2.0。
        """
        robot = self._get_robot_asset()
        return torch.sum(torch.square(robot.data.root_lin_vel_b[:, :2]), dim=1)

    def _reward_base_angular_velocity(self):
        """惩罚基座 roll/pitch 角速度（评测环境使用）。

        对齐评测 base_angular_velocity 权重 -0.05。
        """
        robot = self._get_robot_asset()
        return torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)

    def _reward_joint_vel(self):
        """惩罚关节速度（评测环境使用）。

        对齐评测 joint_vel 权重 -0.001。
        """
        robot = self._get_robot_asset()
        return torch.sum(torch.square(robot.data.joint_vel), dim=1)

    def _reward_energy(self):
        """能耗惩罚：扭矩 × 关节速度（评测环境使用）。

        对齐评测 energy 权重 -2e-05。
        """
        robot = self._get_robot_asset()
        return torch.sum(torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1)

    def _reward_flat_orientation_l2(self):
        """L2 范数姿态惩罚：projected_gravity 的 XY 分量（评测环境使用）。

        对齐评测 flat_orientation_l2 权重 -2.5。
        与 flat_orientation 的区别：L2 范数 vs 求和后再平方。
        """
        robot = self._get_robot_asset()
        return torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)

    def _reward_joint_pos(self):
        """关节位置偏离默认姿态惩罚（评测环境使用）。

        对齐评测 joint_pos 权重 -0.7。
        """
        robot = self._get_robot_asset()
        return torch.sum(torch.square(robot.data.joint_pos - robot.data.default_joint_pos), dim=1)

    def _reward_air_time_variance(self):
        """步态对称性惩罚：脚部滞空/接触时间的方差（评测环境使用）。

        对齐评测 air_time_variance 权重 -1.0。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
        return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
            torch.clip(last_contact_time, max=0.5), dim=1
        )

    def _reward_feet_slide(self):
        """脚部滑动惩罚：接触地面时的脚部速度（评测环境使用）。

        对齐评测 feet_slide 权重 -0.1。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]
        contacts = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        )
        body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
        return reward

    # =====================================================================
    # 冠军方案迁移奖励（V7 新增）
    # 以下奖励项借鉴冠军方案的 reward 设计，用于提升基座高度控制和髋关节稳定性。
    # =====================================================================

    def _reward_base_height(self):
        """惩罚基座高度偏离理想值（冠军方案权重 -1.0）。

        基座高度偏离理想值（0.35m）时给予惩罚，
        鼓励机器人保持稳定的基座高度，避免上下起伏过大。
        """
        robot = self._get_robot_asset()
        base_height = robot.data.root_pos_w[:, 2]
        # 理想高度 0.35m（与 init_state 一致）
        return torch.square(base_height - 0.35)

    def _reward_hip_to_default(self):
        """惩罚髋关节偏离默认角度（冠军方案权重 -0.5）。

        髋关节偏离默认角度时给予惩罚，
        鼓励机器人保持髋关节在合理范围内，减少异常姿态。
        Go2 的髋关节是前 4 个关节（indices 0,1,4,5）。
        """
        robot = self._get_robot_asset()
        # Go2 的髋关节是前 4 个关节（indices 0,1,4,5）
        hip_indices = [0, 1, 4, 5]
        hip_deviation = torch.sum(
            torch.square(robot.data.joint_pos[:, hip_indices] - robot.data.default_joint_pos[:, hip_indices]),
            dim=1
        )
        return hip_deviation

    # =====================================================================
    # 下楼梯专项优化奖励
    # 针对 pyramid_stairs_inv 地形的短板进行优化
    # =====================================================================

    def _reward_stairs_descend_progress(self):
        """下楼梯前进奖励：在高度下降时奖励正向速度。

        下楼梯时机器人高度持续下降是正常行为，当前进速度为正时给予奖励，
        鼓励机器人在下楼梯时保持前进而不是停滞或摔倒。
        """
        robot = self._get_robot_asset()
        forward_vel = robot.data.root_lin_vel_b[:, 0]
        # 使用 projected_gravity 的 z 分量判断是否在斜面上
        # 平地时 gravity_z ≈ -9.81，倾斜时绝对值变小（更接近 0）
        gravity_z = robot.data.projected_gravity_b[:, 2]
        is_on_slope = gravity_z > -9.0  # 倾斜时 gravity_z > -9.0
        # 奖励前进速度，仅在斜面上时生效
        reward = torch.clamp(forward_vel, min=0.0)
        return reward * is_on_slope.float()

    def _reward_adaptive_orientation(self):
        """自适应姿态惩罚：根据坡度动态调整惩罚强度。

        平地时惩罚最重（保持水平），坡度越大惩罚越轻（允许倾斜）。
        替代固定的 flat_orientation_l2，实现地形自适应。

        原理：
        - projected_gravity_b[:, 2] 在平地时 ≈ -9.81，倾斜时绝对值变小
        - 通过 gravity_z 计算坡度因子，映射到 [0, 1]：0=平地，1=45°坡
        - 权重 = 1.0 - 0.8 * slope_factor：平地1.0，45°坡0.2
        """
        robot = self._get_robot_asset()
        gravity_proj = robot.data.projected_gravity_b[:, :2]
        base_penalty = torch.sum(gravity_proj ** 2, dim=-1)

        # 基于 gravity_z 计算坡度因子
        # gravity_z: -9.81(平地) → ~-6.9(45°坡)
        gravity_z = robot.data.projected_gravity_b[:, 2]
        # 归一化到 [0, 1]：0=平地，1=45°坡
        slope_factor = torch.clamp((gravity_z + 9.81) / 2.91, 0.0, 1.0)
        # 权重：平地1.0，45°坡0.2
        adaptive_weight = 1.0 - 0.8 * slope_factor

        return base_penalty * adaptive_weight

    # =====================================================================
    # Track 导航奖励（Phase 2）
    # 以下奖励项用于 Track 模式下的目标导航训练。
    # =====================================================================

    def _reward_approach_goal(self):
        """接近目标点奖励：-(current_dist - previous_dist)。

        距离减少→正奖励，距离增加→负奖励。
        仅在 env.goal_positions 存在时生效（Track 地形）。

        实现说明：
        - 需要 env.goal_positions 由 TerrainExitManager 设置
        - 首次调用时初始化 previous_dist，返回零
        - Episode reset 时清零 delta，避免距离跳变
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]  # (N, 2)
        goal_pos = self.env.goal_positions[:, :2]  # (N, 2)

        current_dist = torch.norm(goal_pos - robot_pos, dim=1)  # (N,)

        # 首次调用：初始化 previous_dist，返回零
        if not hasattr(self.env, "_previous_goal_dist") or self.env._previous_goal_dist is None:
            self.env._previous_goal_dist = current_dist.clone()
            return torch.zeros(self.env.num_envs, device=self.env.device)

        # 距离变化（正=远离，负=接近）
        delta_dist = current_dist - self.env._previous_goal_dist

        # 重置的 env 不计算 delta（距离跳变）
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        delta_dist[reset_mask] = 0.0

        # 更新 previous
        self.env._previous_goal_dist = current_dist.clone()

        # 返回负的距离变化 = 接近→正奖励
        return -delta_dist

    def _reward_goal_direction(self, command_name: str = "base_velocity"):
        """奖励朝目标方向移动的速度分量。

        计算机器人 XY 速度在目标方向上的投影，鼓励机器人朝目标前进。
        仅在有速度命令且 goal_positions 存在时生效。

        Args:
            command_name: 命令项名称。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]  # (N, 2)
        goal_pos = self.env.goal_positions[:, :2]  # (N, 2)

        # 目标方向单位向量
        goal_dir = goal_pos - robot_pos  # (N, 2)
        goal_dir_norm = torch.norm(goal_dir, dim=1, keepdim=True).clamp(min=1e-6)
        goal_unit = goal_dir / goal_dir_norm  # (N, 2)

        # 机器人 XY 速度（世界坐标系）
        base_vel_xy = robot.data.root_lin_vel_w[:, :2]  # (N, 2)

        # 速度在目标方向上的投影（点积）
        vel_toward_goal = torch.sum(base_vel_xy * goal_unit, dim=1)  # (N,)

        # 仅在有前进命令时给奖励
        cmd = self.env.command_manager.get_command(command_name)
        has_fwd_cmd = cmd[:, 0] > 0.05

        return vel_toward_goal * has_fwd_cmd.float()

    def _reward_navigation_time(self):
        """每步固定惩罚，鼓励快速到达目标。

        返回固定值 1.0，由 weight 控制大小（weight 应为负）。
        仅在 goal_positions 存在时生效（Track 地形）。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        return torch.ones(self.env.num_envs, device=self.env.device)
