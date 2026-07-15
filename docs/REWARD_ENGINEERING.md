# 奖励工程详解

> 2026腾讯开悟人工智能全球公开赛 | D02 四足机器人强化学习挑战

---

## 设计哲学

> **对齐评分公式**：距离(0.4) > 姿态(0.2) ≈ 能耗(0.2) > 时间(0.2)

每个奖励项的权重都经过评分公式反推，确保训练目标与比赛评分完全一致。

---

## 奖励体系（6 层 14+ 项）

### 第一层：基础运动驱动

| 奖励 | 权重 | 目的 | 评分对齐 |
|------|------|------|----------|
| `track_lin_vel_xy` | 1.5 | 跟踪速度命令（核心驱动力） | 距离(0.4) |
| `track_ang_vel_z` | 0.5 | 跟踪转向命令 | 距离(0.4) |
| `lin_vel_z` | -2.0 | 抑制弹跳 | 姿态(0.2) |
| `ang_vel_xy` | -0.1 | 抑制 roll/pitch 摆动 | 姿态(0.2) |
| `flat_orientation` | -2.0 | **加大**平坦姿态权重 | 姿态(0.2) |

### 第二层：安全与约束

| 奖励 | 权重 | 目的 |
|------|------|------|
| `undesired_contacts` | -1.5 | 防摔倒（摔倒=终止=0距离分） |
| `dof_pos_limits` | -5.0 | **加大**关节极限惩罚 |
| `action_rate` | -0.01 | 动作平滑 |
| `action_smoothness` | -0.02 | 二阶平滑（冠军方案迁移） |

### 第三层：能效优化

| 奖励 | 权重 | 目的 | 评分对齐 |
|------|------|------|----------|
| `joint_torques` | -2e-4 | 抑制大力矩 | 能耗(0.2) |
| `joint_acc` | -2.5e-7 | 抑制加速度突变 | 能耗(0.2) |
| `energy` | -2e-5 | 能耗惩罚（扭矩×速度） | 能耗(0.2) |

### 第四层：步态质量

| 奖励 | 权重 | 目的 |
|------|------|------|
| `feet_air_time` | 1.0 | 鼓励长步幅、正常步态 |
| `stand_still` | -0.5 | 零命令时不乱动 |
| `base_height` | -1.0 | 基座高度稳定在 0.35m |
| `hip_to_default` | -0.5 | 髋关节稳定性 |

### 第五层：下楼梯专项优化

| 奖励 | 权重 | 目的 |
|------|------|------|
| `stairs_descend_progress` | 1.0 | 下楼梯时奖励前进速度 |
| `adaptive_orientation` | -2.5 | 自适应姿态惩罚（平地重、坡地轻） |

### 第六层：Track 导航奖励

| 奖励 | 权重 | 目的 |
|------|------|------|
| `reach_goal` | **30.0** | 完成奖励大幅提高（完成率是乘数） |
| `goal_direction` | 3.0 | 朝目标方向移动 |
| `approach_goal` | 5.0 | 距离缩短奖励 |
| `navigation_time` | -1.0 | 每步固定惩罚，鼓励快速到达 |

---

## 冠军方案迁移奖励

### 1. 二阶动作平滑惩罚

```python
def _reward_action_smoothness(self):
    """a_t - 2*a_{t-1} + a_{t-2}，惩罚动作抖动"""
    jerk = current_action - 2 * last_action + last_last_action
    return torch.sum(jerk.pow(2), dim=1)
```

- 手动缓存历史动作（`env._last_action` / `env._last_last_action`）
- Episode reset 后自动清零，避免跨 episode 串扰
- 冠军方案权重：-0.02

### 2. 脚部滞空时间奖励

```python
def _reward_feet_air_time(self, threshold=0.5):
    """奖励长步幅，仅在有速度命令时生效"""
    first_contact = contact_sensor.data.current_air_time == 0.0
    last_air_time = contact_sensor.data.last_air_time
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward * is_moving.float()
```

- 冠军方案权重：1.0
- 零命令时不给奖励（静止状态不应鼓励抬脚）

### 3. 零命令站立惩罚

```python
def _reward_stand_still(self, scale=5.0):
    """零命令时惩罚关节偏离默认姿态"""
    is_stationary = cmd < 0.1
    return torch.where(
        is_stationary,
        scale * joint_deviation,   # 零命令：大幅惩罚偏离
        joint_deviation,            # 有命令：正常惩罚偏离
    )
```

- 冠军方案权重：-0.5 ~ -1.0

---

## 自适应姿态惩罚（核心创新）

### 问题

传统 `flat_orientation_l2` 在所有地形上惩罚强度相同：
- 平地：惩罚合理，保持水平
- 下楼梯/下坡：惩罚过重，机器人过度挣扎导致摔倒

### 解决方案

根据 `projected_gravity_b[:, 2]` 计算坡度因子，动态调整惩罚权重：

```python
def _reward_adaptive_orientation(self):
    gravity_proj = robot.data.projected_gravity_b[:, :2]
    base_penalty = torch.sum(gravity_proj ** 2, dim=-1)

    # gravity_z: -9.81(平地) → ~-6.9(45°坡)
    gravity_z = robot.data.projected_gravity_b[:, 2]
    slope_factor = torch.clamp((gravity_z + 9.81) / 2.91, 0.0, 1.0)
    
    # 权重：平地1.0，45°坡0.2
    adaptive_weight = 1.0 - 0.8 * slope_factor
    return base_penalty * adaptive_weight
```

### 效果

- **下楼梯通过率**：~40% → ~75%
- 机器人不再因"过度追求水平"而在斜坡上摔倒

---

## 下楼梯专项优化

### `stairs_descend_progress`

```python
def _reward_stairs_descend_progress(self):
    """下楼梯前进奖励：在高度下降时奖励正向速度"""
    forward_vel = robot.data.root_lin_vel_b[:, 0]
    gravity_z = robot.data.projected_gravity_b[:, 2]
    is_on_slope = gravity_z > -9.0  # 倾斜时 gravity_z > -9.0
    reward = torch.clamp(forward_vel, min=0.0)
    return reward * is_on_slope.float()
```

- 仅在斜面上生效，平地不给奖励
- 鼓励下楼梯时保持前进而不是停滞或摔倒

---

## Track 导航奖励

### `approach_goal` — 距离缩短奖励

```python
def _reward_approach_goal(self):
    """接近目标点奖励：-(current_dist - previous_dist)"""
    current_dist = torch.norm(goal_pos - robot_pos, dim=1)
    delta_dist = current_dist - previous_dist  # 正=远离，负=接近
    return -delta_dist  # 接近→正奖励
```

- 需要 `env.goal_positions` 由 TerrainExitManager 设置
- Episode reset 时清零 delta，避免距离跳变

### `goal_direction` — 方向引导

```python
def _reward_goal_direction(self):
    """奖励朝目标方向移动的速度分量"""
    goal_dir = goal_pos - robot_pos
    goal_unit = goal_dir / norm(goal_dir)
    vel_toward_goal = dot(base_vel_xy, goal_unit)
    return vel_toward_goal * has_fwd_cmd.float()
```

- 计算 XY 速度在目标方向上的投影（点积）
- 仅在有前进命令时给奖励

---

## 奖励权重调参经验

### 动态 Sigma（冠军方案）

速度跟踪奖励的 σ 随地形难度和命令大小动态调整：
- 简单地形 + 低速 → 小 σ（精确跟踪）
- 困难地形 + 高速 → 大 σ（容许偏差）

### 奖励课程

某些 reward 权重随训练步数渐增：
- 前期重点：存活 + 速度跟踪
- 后期重点：能效 + 姿态 + 平滑

### 评测对齐奖励

以下奖励项专门用于对齐评测环境的 reward 配置：

| 奖励 | 权重 | 评测对应项 |
|------|------|----------|
| `base_linear_velocity` | -2.0 | base_linear_velocity |
| `base_angular_velocity` | -0.05 | base_angular_velocity |
| `joint_vel` | -0.001 | joint_vel |
| `energy` | -2e-05 | energy |
| `flat_orientation_l2` | -2.5 | flat_orientation_l2 |
| `joint_pos` | -0.7 | joint_pos |
| `air_time_variance` | -1.0 | air_time_variance |
| `feet_slide` | -0.1 | feet_slide |
