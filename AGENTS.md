# AGENTS.md — Legged Robot Competition (Tencent AI Arena)

> This file is written for AI coding agents. It assumes the reader knows nothing about the project.
> Language: 中文（项目主要注释与文档语言）

---

## 1. 项目概述

本项目是**腾讯开悟平台（Tencent AI Arena）** 的**四足机器人自主导航运控赛题**代码包。

- **任务目标**：使用强化学习算法训练智能体，控制 Unitree Go2 四足机器人在仿真环境中实现自主导航与运动控制，在未知/半未知场景中自主寻路、跨越地形。
- **环境**：基于 Isaac Lab 的仿真环境，使用 trimesh 地形。
- **机器人**：Unitree Go2，4 条腿 × 3 关节 = 12 个可控关节。
- **算法**：基于 PPO（Proximal Policy Optimization）的 Actor-Critic 算法。

### 两种任务模式

| 模式 | 说明 | 评分重点 |
|------|------|----------|
| **Standard（标准模式）** | 在多种复杂地形（坡面、楼梯、迷宫）上行走 | 前进距离、通过时间、能量效率、姿态稳定性 |
| **Track（赛道模式）** | 在由多种子地形串联构成的赛道上从起点导航至终点 | 完成数量、通过时间、姿态稳定性、能量效率 |

### 项目代号

- `project_code`: `legged_robot_competition_26`
- 版本: `22.0.12-comp-normal-lite.saas.sim`

---

## 2. 技术栈

| 组件 | 技术 |
|------|------|
| 编程语言 | Python 3 |
| 深度学习框架 | PyTorch |
| 仿真环境 | Isaac Lab (NVIDIA Isaac Sim) |
| 强化学习框架 | KaiwuDRL（腾讯自研分布式 RL 框架） |
| 配置格式 | TOML |
| 环境管理 | Conda (`env_isaaclab`) |

---

## 3. 项目结构

```
.
├── train_test.py              # 训练/测试入口脚本
├── kaiwu.json                 # 项目元数据配置
├── conf/                      # 全局框架配置
│   ├── algo_conf_legged_robot_competition_26.toml   # 算法映射配置
│   ├── app_conf_legged_robot_competition_26.toml    # 应用配置
│   └── configure_app.toml     # 训练框架全局配置（样本池、模型保存等）
├── agent_ppo/                 # PPO 算法实现（当前主要使用）
│   ├── agent.py               # 智能体入口（Agent 类）
│   ├── algorithm/
│   │   └── algorithm_ppo.py   # PPO 算法核心
│   ├── model/
│   │   └── actor_critic.py    # Actor-Critic 网络模型
│   ├── feature/                 # 特征处理
│   │   ├── definition.py      # ObsData/ActData/RolloutStorage 定义
│   │   ├── policy_observation_process.py  # Policy 观测处理
│   │   ├── critic_observation_process.py  # Critic 观测处理
│   │   ├── reward_process.py    # 自定义奖励函数
│   │   ├── nav_command.py       # Track 导航命令注入
│   │   ├── track_tensor_bridge.py # Track 张量桥接
│   │   ├── nav_signal.py        # 导航信号辅助
│   │   └── feature_layout.py    # 观测布局常量定义
│   ├── conf/                    # 配置
│   │   ├── conf.py              # StageConfig / 配置加载逻辑
│   │   ├── monitor_builder.py   # 监控面板配置
│   │   ├── train_env_conf_standard_locomotion.toml   # Standard 训练配置
│   │   └── train_env_conf_track_navigation.toml      # Track 训练配置
│   ├── workflow/
│   │   └── train_workflow.py    # 训练工作流
│   └── tool/
│       └── scan.py              # 静态扫描工具（foot keyword 扫描）
├── agent_diy/                   # DIY 算法模板（供选手自行开发）
│   ├── agent.py                 # 智能体入口（与 agent_ppo 结构相同）
│   ├── algorithm/
│   │   └── algorithm.py         # PPO 算法（复制自 agent_ppo）
│   ├── model/
│   │   └── actor_critic.py      # Actor-Critic 网络（复制自 agent_ppo）
│   ├── feature/                   # 特征处理（与 agent_ppo 类似）
│   ├── conf/                    # 配置（与 agent_ppo 类似）
│   └── workflow/
│       └── train_workflow.py    # 训练工作流（复制自 agent_ppo）
├── isaac_env/                   # 环境占位目录（空，环境由框架提供）
├── docs/                        # 项目文档（中文）
│   ├── 开发指南/                # 开发指南
│   ├── 腾讯开悟强化学习框架/      # 框架文档
│   ├── 适配方案/                # 适配方案与审查建议
│   ├── 分布式计算框架.md          # KaiwuDRL 架构说明
│   ├── 强化学习系列系统技术标准.md # 技术标准
│   └── 其他工具/                # 日志与监控等
└── .vscode/
    └── launch.json              # VS Code 调试配置
```

---

## 4. 代码组织与模块划分

### 4.1 智能体（Agent）

`agent_ppo/agent.py` 中的 `Agent` 类继承自 `BaseAgent`，是核心入口：

- `__init__`: 加载配置、初始化模型、优化器、算法
- `predict`: 训练时生成动作（带探索噪声）
- `exploit`: 评估时确定性动作（均值）
- `learn`: 触发 PPO 训练
- `save_model` / `load_model`: 模型保存/加载（支持部分加载用于跨阶段迁移）

### 4.2 模型（Model）

`agent_ppo/model/actor_critic.py`:

- `ActorCritic` 类：独立的 Actor MLP + Critic MLP
- Actor: `num_obs → [512, 256, 128] → num_actions(12)`，ELU 激活
- Critic: `num_critic_obs → [512, 256, 128] → 1`，含 LayerNorm + ELU 激活
- 动作分布：高斯分布（Normal），支持 `scalar` / `log` 两种 std 类型

### 4.3 算法（Algorithm）

`agent_ppo/algorithm/algorithm_ppo.py`:

- `AlgorithmPPO` 类：PPO 核心训练逻辑
- 支持自适应学习率（基于 KL 散度）
- 含 NaN/Inf 防护机制（跳过非法 mini-batch）
- 支持 value loss 归一化（按 returns 方差）

### 4.4 特征处理（Feature）

| 文件 | 职责 |
|------|------|
| `definition.py` | `ObsData`, `ActData`, `RolloutStorage`（PPO 经验回放缓冲区） |
| `policy_observation_process.py` | Policy 观测处理（301 dim = proprio 45 + height_scan 256） |
| `critic_observation_process.py` | Critic 观测处理（316 dim = critic_proprio 60 + height_scan 256） |
| `reward_process.py` | 自定义奖励函数（reach_goal, forward_velocity 等） |
| `nav_command.py` | Track 模式下导航命令注入（改写 velocity command） |
| `track_tensor_bridge.py` | Track 模式下环境张量桥接（goal/robot/nav sensor 数据提取） |
| `feature_layout.py` | 观测张量布局常量定义（slice 索引） |

### 4.5 配置系统（Config）

`agent_ppo/conf/conf.py`:

- `StageConfig` 基类：定义模型架构维度、训练超参数
- `LocomotionConfig`：Standard 阶段配置
- `TrackConfig`：Track 阶段配置（含导航专用参数）
- `Config.CURRENT`：切换当前阶段的开关
- 配置加载：先加载 `tools/conf/base/*.toml` 基础配置，再与用户 TOML 深度合并

### 4.6 工作流（Workflow）

`agent_ppo/workflow/train_workflow.py`:

- `workflow()`: 主训练循环
- 四阶段循环：
  1. **数据收集**：`run_episodes_` 收集轨迹
  2. **策略更新**：`agent.learn()`
  3. **监控指标**：每 60 秒上报
  4. **模型保存**：按 `save_interval` 保存

---

## 5. 关键配置说明

### 5.1 切换算法

编辑 `train_test.py`：

```python
algorithm_name = "ppo"   # 或 "diy"
```

### 5.2 切换训练阶段

编辑 `agent_ppo/conf/conf.py`（或 `agent_diy/conf/conf.py`）：

```python
class Config:
    CURRENT = TrackConfig      # Track 导航训练
    # CURRENT = LocomotionConfig  # Standard 基础运动训练
```

### 5.3 训练配置 TOML

| 文件 | 用途 |
|------|------|
| `agent_ppo/conf/train_env_conf_standard_locomotion.toml` | Standard 模式训练参数 |
| `agent_ppo/conf/train_env_conf_track_navigation.toml` | Track 模式训练参数 |

关键参数：
- `[env] num_envs`: 并行环境数（默认 4096）
- `[terrain] mode`: 地形模式 (`standard` / `track`)
- `[rewards.*] weight`: 各奖励项权重

### 5.4 全局框架配置

`conf/configure_app.toml`:

- `replay_buffer_capacity`: 样本池容量（默认 4096）
- `train_batch_size`: Learner 训练批次大小（默认 2048）
- `dump_model_freq`: 模型保存间隔（默认 200 步）
- `preload_model`: 是否启用预加载模型

---

## 6. 构建与运行

### 6.1 环境要求

- Conda 环境：`env_isaaclab`
- Python 路径：`/opt/conda/envs/env_isaaclab/bin/python`
- 需要 NVIDIA GPU（CUDA）

### 6.2 本地训练/测试

```bash
# 激活环境
conda activate env_isaaclab

# 运行训练测试（使用 train_test.py 中指定的算法）
python train_test.py
```

`train_test.py` 调用 `kaiwudrl.common.utils.train_test_utils.run_train_test()`，支持：
- 本地单机训练
- 跳过 aisrv 存活检查（`skip_aisrv_alive_check=True`）
- 跳过错误扫描（`skip_error_scan=True`）

### 6.3 VS Code 调试

已配置 `.vscode/launch.json`，可直接使用 `TestTrain` 配置调试 `train_test.py`。

---

## 7. 测试策略

本项目**无传统单元测试框架**。测试主要通过以下方式：

1. **训练测试（train_test）**：运行 `train_test.py` 验证训练流程是否通畅
2. **配置校验**：`tools.train_env_conf_validate.check_usr_conf()` 在 Agent 初始化时自动校验 TOML 配置合法性
3. **评估任务**：在腾讯开悟平台创建评估任务，验证模型性能
4. **监控指标**：通过 `monitor_builder.py` 配置的监控面板观察训练指标

---

## 8. 代码风格指南

### 8.1 文件头模板

所有 Python 文件使用统一头：

```python
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""
```

### 8.2 注释规范

- **中英双语注释**：所有重要接口、类、函数均需提供中文和英文注释
- 英文注释在前，中文注释在后（或反之），保持成对出现
- 示例：
  ```python
  def process(self):
      """Compute policy observation.
      计算 policy 观测。
      """
  ```

### 8.3 命名规范

- 类名：PascalCase（如 `ActorCritic`, `RolloutStorage`）
- 函数/变量：snake_case（如 `compute_returns`, `num_envs`）
- 私有方法：下划线前缀（如 `_init_flat`, `_load_conf`）
- 常量：全大写（如 `NAV_VALID = 0`）

### 8.4 类型注解

- 鼓励使用 Python 3 类型注解（`from __future__ import annotations`）
- 张量形状在 docstring 中标注（如 `[B, num_obs]`）

---

## 9. 开发约定

### 9.1 模型架构维度

**模型架构维度是架构常量，非用户可调业务参数**。这些值由 Isaac Lab 任务定义决定，**不应放入用户 TOML 配置**，也不应随意修改：

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_actions` | 12 | Go2 关节动作维度 |
| `num_proprio_obs` | 45 | 本体感知观测维度 |
| `num_scan` | 256 | 16×16 高度扫描维度 |
| `num_critic_observations` | 316 | critic 观测维度（Standard） |

### 9.2 奖励函数开发

- 自定义奖励写在 `reward_process.py` 中
- 通用 locomotion 奖励（如 `track_lin_vel_xy`, `joint_acc` 等）继承自 `RewardProcessBase`，**只需在 TOML 中激活，无需重复实现**
- 奖励函数命名：`def _reward_<name>(self, ...)`
- 返回 `torch.Tensor`，形状为 `(num_envs,)`

### 9.3 观测处理开发

- `PolicyObservationProcess` 和 `CriticObservationProcess` 需保持同步
- Track 模式下可扩展 goal/nav 特征，但需同步修改 `StageConfig` 的观测维度

### 9.4 新增训练阶段

1. 继承 `StageConfig` 创建新子类
2. 在同目录创建 `train_env_conf_<task_type>_<stage.name>.toml`
3. 修改 `Config.CURRENT` 指向新阶段

### 9.5 NaN/Inf 防护

项目中有严格的 NaN/Inf 防护机制：
- PPO 训练时若 loss 非法，跳过该 mini-batch
- backward 后若梯度含 NaN，跳过 `optimizer.step()`
- GAE 计算前清洗输入数据（`torch.nan_to_num`）
- 修改时需保持防护逻辑完整

---

## 10. 安全注意事项

1. **模型加载安全**：`load_model` 支持部分加载（跨阶段迁移），但会检查 shape 不匹配。禁止修改原始模型文件名，否则预加载失败。
2. **配置校验**：`check_usr_conf` 在初始化时校验配置合法性，非法配置会抛出异常并终止训练。
3. **动作裁剪**：环境交互前动作会被裁剪到 `[-6.0, 6.0]` 范围。
4. **Git 忽略**：`.gitignore` 已配置忽略二进制文件、模型文件（`.pkl`, `.pth`）、日志等。

---

## 11. 部署与提交

### 11.1 腾讯开悟平台

- 代码包通过平台上传/提交
- 模型文件（`.pkl`）由框架自动打包到 zip 中
- 评估任务由平台官方实现，调用 `agent.exploit()`

### 11.2 模型保存

- 保存路径：`{path}/model.ckpt-{id}.pkl`
- 文件名必须包含 `model.ckpt-id` 字段
- 平台可能有最小保存间隔限制

---

## 12. 常用文件速查

| 目的 | 文件路径 |
|------|----------|
| 切换算法 | `train_test.py` |
| 切换训练阶段 | `agent_ppo/conf/conf.py` → `Config.CURRENT` |
| Standard 训练配置 | `agent_ppo/conf/train_env_conf_standard_locomotion.toml` |
| Track 训练配置 | `agent_ppo/conf/train_env_conf_track_navigation.toml` |
| 修改奖励 | `agent_ppo/feature/reward_process.py` |
| 修改观测 | `agent_ppo/feature/policy_observation_process.py` / `critic_observation_process.py` |
| 修改模型结构 | `agent_ppo/model/actor_critic.py` |
| 修改训练算法 | `agent_ppo/algorithm/algorithm_ppo.py` |
| 修改监控指标 | `agent_ppo/conf/monitor_builder.py` |
| 全局框架配置 | `conf/configure_app.toml` |

---

## 13. 外部依赖与框架接口

本项目依赖以下外部模块（由腾讯开悟平台/客户端提供）：

- `kaiwudrl.interface.agent.BaseAgent`
- `kaiwudrl.common.utils.train_test_utils.run_train_test`
- `kaiwudrl.common.monitor.monitor_config_builder.MonitorConfigBuilder`
- `common_python.utils.common_func.Frame`, `create_cls`
- `common_python.config.config_control.CONFIG`
- `tools.base_env.observation_process.ObservationProcess`
- `tools.base_env.base_reward.RewardProcessBase`
- `tools.train_env_conf_validate.check_usr_conf`
- `tools.utils.load_reward_keys_from_monitor_config`

**注意**：这些模块在本地开发环境中可能不可用，需在腾讯开悟平台/客户端中运行。

---

*最后更新：基于项目版本 22.0.12-comp-normal-lite.saas.sim*
