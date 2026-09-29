# Go2 四足机器人强化学习：自主导航 + 运动控制

> **一句话**：用 PPO（奖励工程深度对齐评分公式 + 分阶段课程学习）训练 Unitree Go2（12 可控关节）在坡面、楼梯、迷宫、赛道上稳定行走并自主导航至目标点
> **成绩**：2026 腾讯开悟人工智能全球公开赛 D02 — 中部区域初赛**第 2** · 区域决赛第 5 · 全国决赛**二等奖**
> **怎么跑**：须在**腾讯开悟强化学习平台**（申请制）内训练（Isaac Lab 仿真与 KaiwuDRL 框架由平台提供，无法本地独立运行）；同平台参赛者入口 `python train_test.py`（Standard / Track 模式切换见 `agent_ppo/conf/conf.py` 的 `Config.CURRENT`），流程详见 [docs/TRAINING.md](docs/TRAINING.md)

---

## ⚠️ 复现说明

本仓库**不提供脱离平台的详细复现步骤**，原因：训练与评估必须在**腾讯开悟（Tencent AI Arena）强化学习平台**（<https://aiarena.tencent.com/>）内进行（平台为**申请制**：需注册申请并通过审核后方可使用）——

- 仿真基于 Isaac Lab，环境代码由平台侧注入（仓库中 `isaac_env/` 仅为占位目录）
- 训练依赖 `kaiwudrl` / `common_python` / `tools.*` 等平台模块，本地环境没有这些依赖
- 模型评估（`agent.exploit()`）与正式提交均通过平台任务完成

本地可验证的部分已沉淀为静态测试（无需平台依赖）：`pytest tests`（观测布局 ABI / agent_diy 槽位契约 / 配置解析 / ActorCritic 前向，torch 用例在无 torch 环境自动跳过），CI 见 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)。

因此本仓库的定位是**方案本身的完整开源**：奖励工程、网络结构、观测设计、课程学习与调参细节全部可见。同平台参赛者按平台流程导入代码包即可训练；其他读者可作为四足 locomotion + 导航的强化学习工程参考。

---

## 赛题简介

使用强化学习算法训练智能体，控制 **Unitree Go2 四足机器人**（12个可控关节）在仿真环境中实现自主导航与运动控制，使其能够在复杂地形（坡面、楼梯、迷宫、赛道）上稳定行走并快速到达目标点。

**两种比赛模式**：
- **Standard 模式**：在多种复杂地形上行走，综合评分（距离 40% + 时间 20% + 能耗 20% + 姿态 20%）
- **Track 模式**：在串联赛道上从起点导航至终点，以完成率、时间、姿态、能耗综合评分

---

## 核心算法

**PPO（Proximal Policy Optimization）** + 深度奖励工程 + 分阶段课程学习

```
Actor:  obs[301] → [512, 256, 128] → actions[12]
Critic: critic_obs[316] → [512, 256, 128] → value[1]
```

---

## 六大技术亮点

### 1. 奖励工程深度对齐评分公式

每个奖励项的权重都经过评分公式 `0.4×距离 + 0.2×时间 + 0.2×能耗 + 0.2×姿态` 反推设计，确保训练目标与比赛评分完全一致。设计了 **14+ 奖励项**，涵盖基础运动驱动、安全约束、能效优化、步态质量、下楼梯专项、导航引导六大层次。

### 2. 自适应姿态惩罚（地形感知）

传统 `flat_orientation` 在所有地形上惩罚强度相同，导致下坡/楼梯时机器人过度挣扎。本项目根据 `projected_gravity` 动态调整惩罚权重：平地严格（权重 1.0）、坡地宽松（权重 0.2），**下楼梯通过率从 ~40% 提升至 ~75%**（训练期平台监控的观察估计值，训练日志未随仓库存档，供参考）。

### 3. 导航命令注入层（Track 模式）

关键决策：**不重新训练整个模型**，将导航转化为速度命令生成层。Standard 阶段训练好的 locomotion 策略保持不变，导航层只负责生成目标导向的 `[vx, vy, wz]` 命令。优势：保留运动能力、快速迁移、模型兼容。

### 4. 三层 NaN/Inf 数值防护

```
Layer 1: Loss 检测 → 非法则跳过 mini-batch
Layer 2: Gradient 检测 → NaN 则清零梯度并跳过 step
Layer 3: Std 钳制 → 替换 NaN/Inf，限制到 [min_std, 1e6]
```

参赛训练全程**未发生数值崩溃**（训练期观察结论；日志未随仓库存档）。

### 5. 下楼梯专项优化

针对 `pyramid_stairs_inv` 短板地形，设计了 `stairs_descend_progress`（奖励下楼梯前进速度）和 `adaptive_orientation`（允许适度倾斜）两项专项奖励，针对性解决下楼梯易摔倒问题。

### 6. 避障扫描器集成（Track 模式，默认关闭）

利用平台 `nav_scanner` 前瞻遮挡扫描实现避障：前方障碍物 < 0.55m 时减速转向，两侧空间 < 0.75m 时判定死胡同强制转向，动态调整速度命令避免碰撞终止。诚实说明：该链路已完整实现并集成于命令注入层，但**最终提交配置中默认关闭**（`agent_ppo/conf/conf.py` 中 `nav_cmd_enable_scanner_avoidance = False`，保守起见未开启）；开启只需改这一行。

---

## 技术文档

| 文档 | 内容 |
|------|------|
| [📄 技术方案总览](docs/TECH_OVERVIEW.md) | 方案选型、分阶段训练路线、模型架构 |
| [📄 奖励工程详解](docs/REWARD_ENGINEERING.md) | 14+ 奖励项设计、权重推导、冠军方案迁移 |
| [📄 导航策略设计](docs/NAVIGATION.md) | 命令注入层、避障算法、Track 模式实现 |
| [📄 训练策略与课程学习](docs/TRAINING.md) | 域随机化、速度命令课程、超参调整、监控指标 |
| [📄 代码结构说明](docs/CODE_STRUCTURE.md) | 项目目录结构、各模块职责 |

---

## 快速开始

```bash
# 训练入口（仅平台内可运行，见上方复现说明；train_test.py 默认 algorithm_name = "ppo"）
python train_test.py

# 切换训练阶段（Standard / Track）
# 修改 agent_ppo/conf/conf.py 中 Config.CURRENT（唯一生效位置）

# 本地静态检查（无需平台依赖）
pytest tests
```

---

## 出处与许可

本仓库基于**腾讯开悟（Tencent AI Arena）赛题代码包**（`kaiwu.json` 声明 `project_code = legged_robot_competition_26`）开发：

- 带 `Copyright © 1998 - 2026 Tencent. All Rights Reserved` 头的模板文件，以及 `docs/` 下的平台文档（开发指南 / 框架文档 / 分布式计算框架等）版权归**腾讯**所有，按平台与赛事条款使用
- 本仓库的原创部分——`agent_ppo/feature/nav_command.py`、`nav_signal.py`、`agent_ppo/tool/scan.py`、`docs/适配方案/`、`docs/` 根部 5 份技术文档、`README.md`、`AGENTS.md`、`agent_diy/README.md`、`tests/` 与工具链配置（`.github/workflows/ci.yml`、`pyproject.toml`）——由 cainiao33 创作，按 **MIT** 许可发布
- 在 Tencent 模板基础上深度修改的文件（`track_tensor_bridge.py`、`feature_layout.py` 等）保留腾讯头，修改部分版权同样归 cainiao33（MIT）
- 因此本仓库**不提供全库单一 LICENSE**（避免将 Tencent "All Rights Reserved" 模板一并再许可）

## AI 协作标注

奖励工程调参与平台训练由作者完成；仓库工程化（静态测试、CI、agent_diy 死代码清理、文档勘误与结构整理）由 **Claude Code** 协助完成，协作记录见各提交信息。

---

## 比赛成绩

| 赛段 | 名次 | 备注 |
|------|------|------|
| **中部区域初赛** | **第 2 名** | 标准模式 + 赛道模式综合 |
| **区域决赛** | **第 5 名** | 全国各区域晋级选手角逐 |
| **全国决赛** | **二等奖** | 全国顶尖队伍最终排名 |

---

> **作者**: cainiao33
> 
> **仓库**: https://github.com/cainiao33/Go2_RL
>
> **平台**: [腾讯开悟（Tencent AI Arena）](https://aiarena.tencent.com/)（申请制）
