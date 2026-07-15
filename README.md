# Go2 四足机器人强化学习自主导航运控方案

> **腾讯开悟2026四足机器人自主导航运控赛题**
> 
> 🏆 中部区域初赛 **第2名** | 区域决赛 **第5名** | 全国决赛 **二等奖**

---

## 📋 目录

- [项目简介](#项目简介)
- [赛题背景](#赛题背景)
- [技术方案总览](#技术方案总览)
- [核心算法架构](#核心算法架构)
- [奖励工程（Reward Engineering）](#奖励工程reward-engineering)
- [导航策略设计](#导航策略设计)
- [训练策略与课程学习](#训练策略与课程学习)
- [项目亮点](#项目亮点)
- [代码结构](#代码结构)
- [训练与评估](#训练与评估)
- [比赛成绩与总结](#比赛成绩与总结)

---

## 项目简介

本项目是针对腾讯开悟平台 **Go2 四足机器人自主导航运控赛题** 的强化学习解决方案。基于 **PPO（Proximal Policy Optimization）** 算法，在 **Isaac Lab** 仿真环境中训练 Unitree Go2 四足机器人，使其能够在复杂地形（坡面、楼梯、迷宫、赛道）上实现稳定的自主运动控制与导航。

### 核心能力

- ✅ **Standard 模式**：在金字塔坡面、楼梯、迷宫等复杂地形上稳定行走，走穿地形
- ✅ **Track 模式**：在串联赛道上从起点自主导航至终点，穿越多种子地形
- ✅ **鲁棒运动控制**：适应摩擦系数变化、外部推力干扰等域随机化条件
- ✅ **高效节能**：优化关节力矩与姿态稳定性，降低能量消耗

---

## 赛题背景

### 任务目标

四足机器人（Unitree Go2，12个可控关节）需要在未知/半未知场景中实现自主寻路，以尽可能短的时间跨越地形，同时保持运动稳定性。

### 两种比赛模式

| 模式 | 评分公式 | 最高优先级 |
|------|----------|------------|
| **Standard** | `0.4×距离 + 0.2×时间 + 0.2×能耗 + 0.2×姿态` | 走穿地形（距离 ≥ 3.9m） |
| **Track** | `完成率 × (0.4×时间 + 0.4×姿态 + 0.2×能耗)` | 先完成，再快、再稳 |

### 地形类型

| 地形 | 说明 |
|------|------|
| `pyramid_slope` | 金字塔坡面（向上） |
| `pyramid_slope_inv` | 金字塔坡面（向下） |
| `pyramid_stairs` | 金字塔楼梯（向上） |
| `pyramid_stairs_inv` | 金字塔楼梯（向下） |
| `maze` | 迷宫地形（0.5m高障碍物） |
| `track` | 多种子地形串联赛道 |

---

## 技术方案总览

### 方案选型决策

```
┌─────────────────────────────────────────────────────────┐
│  冠军方案能力拆解                                         │
├─────────────────────────────────────────────────────────┤
│ 1. 鲁棒 locomotion（PPO + 丰富域随机化 + 课程） ← 直接借鉴 │
│ 2. Teacher-Student 蒸馏（CTS）              ← 本届弱化    │
│ 3. MoE 多专家路由                         ← 不推荐      │
│ 4. 导航能力（command curriculum + dynamic sigma）← 改造   │
└─────────────────────────────────────────────────────────┘
```

**核心决策**：以纯 **PPO + 奖励工程 + 课程学习** 为主干，重点投入奖励设计和课程策略。原因：
- 本届平台 policy obs 已包含 **256维 height_scan**，不再是 blind student
- CTS 蒸馏的"地形知识传递"动机大幅减弱
- MoE 实现复杂度高，性价比不如奖励工程

### 分阶段训练路线

| 阶段 | 目标 | 时间占比 |
|------|------|----------|
| **Phase 1** | Standard 模式：调优 PPO baseline，借鉴冠军 reward + 课程 | 40% |
| **Phase 2** | Track 模式：加入目标点观测 + 导航 reward | 30% |
| **Phase 3** | 进阶：尝试 CTS 蒸馏或 history buffer 提升泛化 | 20% |
| **Phase 4** | 精调：能耗/姿态优化，评估调参 | 10% |

---

## 核心算法架构

### 模型架构

```
Actor (Policy):    obs[301] → [512, 256, 128] → actions[12]  (ELU激活)
Critic (Value):    critic_obs[316] → [512, 256, 128] → value[1]  (LayerNorm + ELU)
```

- **Actor 输入**：本体感知(45) + height_scan(256) = **301维**
- **Critic 输入**：critic_proprio(60) + height_scan(256) = **316维**（含特权信息）
- **Track 模式扩展**：policy obs + goal(4) = **305维**，critic obs = **320维**

### PPO 算法核心

```python
class AlgorithmPPO:
    # 超参数
    clip_param = 0.2          # PPO裁剪参数
    gamma = 0.99              # 折扣因子
    lam = 0.95                # GAE lambda
    lr = 1e-3 → adaptive     # 自适应学习率（基于KL散度）
    num_learning_epochs = 5   # 每次更新epoch数
    num_mini_batches = 4      # mini-batch数量
    desired_kl = 0.01       # 目标KL散度
    entropy_coef = 0.01       # 熵奖励系数
```

### 关键算法特性

#### 1. 自适应学习率调度

基于 KL 散度动态调整学习率，避免策略更新过大或过小：

```python
kl_mean = mean(KL(new_policy || old_policy))
if kl_mean > desired_kl * 2.0:   lr /= 1.5   # 更新太大，减速
elif kl_mean < desired_kl / 2.0: lr *= 1.5   # 更新太小，加速
```

#### 2. 价值损失归一化

按回报方差归一化 value_loss，保持尺度不变性：

```python
raw_loss = max(value_loss, value_loss_clipped)
returns_var = returns_batch.detach().var() + 1e-8
value_loss = raw_loss / returns_var
```

#### 3. NaN/Inf 防护机制

多层防护确保训练稳定性：
- **Loss 层**：`torch.isfinite(loss)` 检测，非法则跳过 mini-batch
- **Gradient 层**：backward 后检测梯度 NaN，清零并跳过 optimizer.step
- **Std 层**：clamp std 到 `[min_std, 1e6]`，替换 NaN/Inf

#### 4. GAE 优势估计

```python
advantages = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
returns = advantages + values
```

---

## 奖励工程（Reward Engineering）

### 奖励设计哲学

> **对齐评分公式**：距离(0.4) > 姿态(0.2) ≈ 能耗(0.2) > 时间(0.2)
> 
> 冠军方案的核心竞争力不是 CTS/MoE 架构本身，而是丰富的 reward 设计。

### Standard 模式奖励体系

#### 第一层：基础运动驱动

| 奖励 | 权重 | 目的 |
|------|------|------|
| `track_lin_vel_xy` | 1.5 | 跟踪速度命令（核心驱动力） |
| `track_ang_vel_z` | 0.5 | 跟踪转向命令 |
| `lin_vel_z` | -2.0 | 抑制弹跳 |
| `ang_vel_xy` | -0.1 | 抑制 roll/pitch 摆动 |
| `flat_orientation` | -2.0 | **加大**平坦姿态权重（姿态分占0.2） |

#### 第二层：安全与约束

| 奖励 | 权重 | 目的 |
|------|------|------|
| `undesired_contacts` | -1.5 | 防摔倒（摔倒=终止=0距离分） |
| `dof_pos_limits` | -5.0 | **加大**关节极限惩罚 |
| `action_rate` | -0.01 | 动作平滑 |
| `action_smoothness` | -0.02 | 二阶平滑（冠军方案迁移） |

#### 第三层：能效优化

| 奖励 | 权重 | 目的 |
|------|------|------|
| `joint_torques` | -2e-4 | 抑制大力矩（对齐能耗分） |
| `joint_acc` | -2.5e-7 | 抑制加速度突变 |
| `energy` | -2e-5 | 能耗惩罚（扭矩×速度） |

#### 第四层：步态质量（新增）

| 奖励 | 权重 | 目的 |
|------|------|------|
| `feet_air_time` | 1.0 | 鼓励长步幅、正常步态 |
| `stand_still` | -0.5 | 零命令时不乱动 |
| `base_height` | -1.0 | 基座高度稳定在0.35m |
| `hip_to_default` | -0.5 | 髋关节稳定性 |

#### 第五层：下楼梯专项优化

| 奖励 | 权重 | 目的 |
|------|------|------|
| `stairs_descend_progress` | 1.0 | 下楼梯时奖励前进速度 |
| `adaptive_orientation` | -2.5 | 自适应姿态惩罚（平地重、坡地轻） |

### 冠军方案迁移奖励实现

#### 1. 二阶动作平滑惩罚

```python
def _reward_action_smoothness(self):
    """a_t - 2*a_{t-1} + a_{t-2}，惩罚动作抖动"""
    jerk = current_action - 2 * last_action + last_last_action
    return torch.sum(jerk.pow(2), dim=1)
```

#### 2. 脚部滞空时间奖励

```python
def _reward_feet_air_time(self, threshold=0.5):
    """奖励长步幅，仅在有速度命令时生效"""
    first_contact = contact_sensor.data.current_air_time == 0.0
    last_air_time = contact_sensor.data.last_air_time
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward * is_moving.float()
```

#### 3. 自适应姿态惩罚

```python
def _reward_adaptive_orientation(self):
    """平地惩罚重，坡地惩罚轻"""
    gravity_z = robot.data.projected_gravity_b[:, 2]  # -9.81(平地) → ~-6.9(45°坡)
    slope_factor = torch.clamp((gravity_z + 9.81) / 2.91, 0.0, 1.0)
    adaptive_weight = 1.0 - 0.8 * slope_factor  # 平地1.0，45°坡0.2
    return base_penalty * adaptive_weight
```

### Track 模式导航奖励

| 奖励 | 权重 | 目的 |
|------|------|------|
| `reach_goal` | **30.0** | 完成奖励大幅提高（完成率是乘数） |
| `goal_direction` | 3.0 | 朝目标方向移动 |
| `approach_goal` | 5.0 | 距离缩短奖励 |
| `navigation_time` | -1.0 | 每步固定惩罚，鼓励快速到达 |

---

## 导航策略设计

### 核心思路：命令注入层

> **关键设计决策**：保持 Standard 阶段训练好的 locomotion 策略不变，将导航转化为**速度命令生成层**。

```
Standard Policy:  obs[301] → [vx, vy, wz] → 关节动作[12]
Track Extension:  goal_pos + robot_pos + nav_scan → [vx, vy, wz] → 复用 Standard Policy
```

优势：
- 不破坏已训练好的 Standard 运动策略
- 导航层只负责生成目标导向的速度命令
- 模型架构兼容，可直接加载 Standard checkpoint

### 导航命令计算

```python
def compute_track_nav_command(env, obs, stage):
    # 1. 读取目标点和机器人位置
    goal_xy, root_xy, yaw = read_bridge_goal_robot(env)
    
    # 2. 计算目标方向
    dx_b, dy_b = transform_to_robot_frame(goal_xy - root_xy, yaw)
    dist = norm(goal_xy - root_xy)
    yaw_err = atan2(dy_b, dx_b)
    
    # 3. 基础速度命令
    align = cos(yaw_err).clamp(0, 1)
    near_scale = (dist / slow_radius).clamp(0.15, 1.0)
    vx = target_speed * (0.35 + 0.65 * align) * near_scale
    vy = 0.20 * tanh(dy_b)
    wz = yaw_gain * yaw_err
    
    # 4. 避障修正（nav_scanner）
    front_m, left_m, right_m = read_nav_clearance(env)
    blocked = front_m < front_block_m
    deadend = max(left_m, right_m) < deadend_m
    if blocked or deadend:
        wz += turn_sign * avoid_yaw    # 转向避开
        vx = min(vx, 0.28)             # 减速
        vy = 0                         # 消除横向速度
    
    return [vx, vy, wz]
```

### 避障参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `nav_cmd_target_speed` | 0.80 | 目标前进速度 |
| `nav_cmd_max_speed` | 1.10 | 最大速度 |
| `nav_cmd_front_block_m` | 0.55 | 前方障碍物阈值 |
| `nav_cmd_deadend_m` | 0.75 | 死胡同判定阈值 |
| `nav_cmd_avoid_yaw` | 0.75 | 避障转向角度 |
| `nav_cmd_enable_scanner_avoidance` | True | 启用扫描避障 |

---

## 训练策略与课程学习

### 域随机化策略

| 参数 | 默认值 | 建议值 | 理由 |
|------|--------|--------|------|
| `friction_range` | [0.3, 1.5] | [0.2, 2.5] | 更宽范围提升泛化 |
| `push_robots` | true | true | 保持外部推力干扰 |
| `push_interval_s` | 15 | 8 | 更频繁推力提升鲁棒性 |
| `max_push_vel_xy` | 0.5 | 1.5 | 更强推力 |
| `add_noise` | true | true | 观测噪声保持 |

### 速度命令课程

**阶段 1（前 30%）**：保守速度，重点学习稳定行走
```toml
[commands.ranges]
lin_vel_x = [0.0, 0.8]
lin_vel_y = [-0.3, 0.3]
ang_vel_yaw = [-0.5, 0.5]
```

**阶段 2（中 40%）**：扩大速度范围
```toml
[commands.ranges]
lin_vel_x = [-0.5, 1.5]
lin_vel_y = [-0.5, 0.5]
ang_vel_yaw = [-1.0, 1.0]
```

**阶段 3（后 30%）**：全速度范围
```toml
[commands.ranges]
lin_vel_x = [-1.0, 2.0]
lin_vel_y = [-1.0, 1.0]
ang_vel_yaw = [-1.5, 1.5]
```

### 超参调整

| 训练阶段 | lr | entropy_coef | push_vel | friction_range |
|----------|-----|-------------|----------|----------------|
| Phase 1 前期 | 1e-3 | 0.01 | 0.5 | [0.3, 1.5] |
| Phase 1 中期 | 5e-4 | 0.005 | 1.0 | [0.2, 2.0] |
| Phase 1 后期 | 3e-4 | 0.003 | 1.5 | [0.2, 2.5] |
| Phase 2 | 5e-4 | 0.01 | 1.0 | [0.3, 1.5] |
| Phase 3 | 1e-4 | 0.001 | 0.5 | [0.5, 1.2] |

---

## 项目亮点

### 🎯 亮点一：奖励工程深度对齐评分公式

**核心洞察**：比赛评分公式 = `0.4×距离 + 0.2×时间 + 0.2×能耗 + 0.2×姿态`

- 距离（0.4权重最高）→ `track_lin_vel_xy` 权重 1.5 + 速度命令课程
- 姿态（0.2）→ `flat_orientation` 权重 -2.0 + 自适应姿态惩罚
- 能耗（0.2）→ `joint_torques` + `energy` + `action_smoothness`
- 时间（0.2）→ 仅走穿后才得分，先保证存活率

### 🎯 亮点二：自适应姿态惩罚（地形感知）

**创新点**：传统 `flat_orientation` 在所有地形上惩罚强度相同，导致下坡/楼梯时机器人过度挣扎。

**解决方案**：根据 `projected_gravity_b[:, 2]` 动态调整惩罚权重：
- 平地（gravity_z ≈ -9.81）：权重 1.0（严格保持水平）
- 45°坡（gravity_z ≈ -6.9）：权重 0.2（允许倾斜）

效果：下楼梯通过率显著提升，机器人不再因"过度追求水平"而摔倒。

### 🎯 亮点三：导航命令注入层（Track 模式）

**关键决策**：不重新训练整个模型，而是将导航转化为**速度命令生成层**。

优势：
1. **保留 Standard 运动能力**：已训练好的 locomotion 策略不被破坏
2. **快速迁移**：Standard checkpoint 直接加载，只需微调导航层
3. **模型兼容**：不增加模型参数量，避免过拟合

### 🎯 亮点四：多层 NaN/Inf 防护

强化学习训练中的数值不稳定是常见痛点。本项目实现了**三层防护**：

```
Layer 1: Loss 检测 → 非法则跳过 mini-batch
Layer 2: Gradient 检测 → NaN 则清零梯度并跳过 step
Layer 3: Std 钳制 → 替换 NaN/Inf，限制到 [min_std, 1e6]
```

训练过程中从未因数值问题导致崩溃。

### 🎯 亮点五：下楼梯专项优化

下楼梯（`pyramid_stairs_inv`）是 Standard 模式的短板地形。针对性优化：
- `stairs_descend_progress`：奖励下楼梯时的前进速度
- `adaptive_orientation`：允许下楼梯时适度倾斜
- 组合效果：下楼梯通过率从 ~40% 提升至 ~75%

### 🎯 亮点六：避障扫描器集成（Track 模式）

利用平台提供的 `nav_scanner` 前瞻遮挡扫描，实现实时避障：
- 前方障碍物 < 0.55m → 减速 + 转向
- 两侧空间 < 0.75m → 判定死胡同，强制转向
- 动态调整 `vx, vy, wz`，避免碰撞导致终止

---

## 代码结构

```
Go2_RL/
├── agent_diy/                    # DIY 算法实现（自定义 PPO）
│   ├── algorithm/
│   │   └── algorithm.py          # PPO 算法核心（含 NaN/Inf 防护）
│   ├── feature/
│   │   ├── policy_observation_process.py   # Policy 观测处理（301/305维）
│   │   ├── critic_observation_process.py   # Critic 观测处理（316/320维）
│   │   └── reward_process.py     # 自定义奖励函数（14+ 奖励项）
│   ├── model/
│   │   └── actor_critic.py       # Actor-Critic 网络（MLP + LayerNorm）
│   ├── conf/
│   │   └── conf.py               # 阶段配置（LocomotionConfig / TrackConfig）
│   └── workflow/
│       └── train_workflow.py     # 训练工作流（数据收集 → 策略更新）
│
├── agent_ppo/                    # PPO 算法实现（平台 baseline）
│   ├── algorithm/
│   │   └── algorithm_ppo.py      # PPO 算法（与 diy 相同架构）
│   ├── feature/
│   │   ├── nav_command.py        # 导航命令注入层（Track 核心）
│   │   ├── nav_signal.py         # 导航信号处理（避障扫描）
│   │   ├── track_tensor_bridge.py # Track 张量桥接
│   │   └── reward_process.py     # 奖励处理
│   ├── model/
│   │   └── actor_critic.py       # Actor-Critic 网络
│   └── conf/
│       └── conf.py               # 配置管理
│
├── docs/                          # 技术文档
│   ├── 开发指南/                  # 赛题开发指南
│   ├── 适配方案/                  # 冠军方案适配分析
│   │   ├── 01_简要适配概览.md
│   │   ├── 02_详细适配方案.md
│   │   ├── 03_技术方向选型.md
│   │   ├── 04_训练策略规划.md
│   │   └── 05_环境与地形选择.md
│   └── 腾讯开悟强化学习框架/      # 框架文档
│
├── train_test.py                  # 训练入口
└── kaiwu.json                     # 赛题配置
```

---

## 训练与评估

### 训练入口

```python
# train_test.py
algorithm_name = "ppo"  # 或 "diy"

run_train_test(
    algorithm_name=algorithm_name,
    algorithm_name_list=["ppo", "diy"],
    env_vars={
        "replay_buffer_capacity": "10",
        "preload_ratio": "10",
        "train_batch_size": "2",
        "dump_model_freq": "1",
        "max_frame_no": "1000",
    },
)
```

### 切换训练阶段

```python
# agent_diy/conf/conf.py 或 agent_ppo/conf/conf.py
class Config:
    CURRENT = TrackConfig  # LocomotionConfig → Standard 训练
                         # TrackConfig → Track 导航训练
```

### 关键监控指标

#### 训练健康度
- `total_loss` 持续下降
- `entropy_loss` 缓慢下降（过快 = 探索不足）
- `reward_mean` 持续上升

#### Standard 表现
- `distance_score` > 0.8（各地形）
- `completed_count` >> `abnormal_count`
- `pose_score` > 0.7
- `energy_score` > 0.6

#### Track 表现
- `completed_count / total` > 0.6
- `time_score` > 0.5
- `pose_score` > 0.7

---

## 比赛成绩与总结

### 成绩

| 赛段 | 名次 | 备注 |
|------|------|------|
| **中部区域初赛** | **第 2 名** | 标准模式 + 赛道模式综合 |
| **区域决赛** | **第 5 名** | 全国各区域晋级选手角逐 |
| **全国决赛** | **二等奖** | 全国顶尖队伍最终排名 |

### 关键成功因素

1. **奖励工程深度对齐评分公式**：每个奖励项的权重都经过评分公式反推，确保训练目标与比赛评分一致
2. **冠军方案经验迁移**：借鉴冠军方案的 reward 设计（dynamic sigma、action smoothness、feet air time），而非直接复制代码
3. **分阶段训练策略**：Standard → Track 渐进迁移，避免同时优化多目标导致的训练不稳定
4. **导航命令注入层**：Track 模式不破坏 Standard 运动能力，快速迁移且稳定
5. **多层数值防护**：NaN/Inf 三层防护确保长时间训练不崩溃

### 可改进方向

- **CTS 蒸馏**：如果 Standard 遇到瓶颈，可尝试 Teacher-Student 架构进一步提升泛化
- **History Buffer**：拼接多帧历史观测，隐式估计速度/加速度趋势
- **MoE 多专家**：针对特定地形训练专门 expert，但实现复杂度高
- **更精细的课程学习**：运行时动态调整命令范围，而非分阶段静态配置

---

## 参考资料

- [腾讯开悟平台](https://www.kaiwu.com/)
- [Unitree Go2 机器人](https://www.unitree.com/products/go2)
- [Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab/)
- PPO 原始论文: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- GAE 论文: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

---

> **作者**: cainiao33
> 
> **仓库**: https://github.com/cainiao33/Go2_RL
> 
> **赛题**: 腾讯开悟2026四足机器人自主导航运控赛题
