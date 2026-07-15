# 导航策略设计

> 2026腾讯开悟人工智能全球公开赛 | D02 四足机器人强化学习挑战

---

## 核心设计决策：命令注入层

### 问题

Track 模式需要导航能力，但：
- 重新训练整个模型 → 破坏 Standard 阶段已训练好的运动能力
- 增加 goal 特征到 obs → 模型输入维度变化，需要重新训练

### 解决方案

**将导航转化为速度命令生成层**：

```
Standard Policy:  obs[301] → [vx, vy, wz] → 关节动作[12]
                  ↑
Track Extension:  goal_pos + robot_pos + nav_scan → [vx, vy, wz]
```

**优势**：
1. **保留 Standard 运动能力**：locomotion 策略不被破坏
2. **快速迁移**：Standard checkpoint 直接加载，只需微调导航层
3. **模型兼容**：不增加模型参数量，避免过拟合
4. **训练稳定**：导航层只负责高层决策，底层运动控制已成熟

---

## 导航命令计算流程

```python
def compute_track_nav_command(env, obs, stage):
    # 1. 读取目标点和机器人位置
    goal_xy, root_xy, yaw = read_bridge_goal_robot(env)
    
    # 2. 计算目标方向（机器人坐标系）
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
    
    # 5. 接近目标，减速并对齐
    if dist < slow_radius and goal_yaw available:
        wz = 0.5 * wz + 0.5 * yaw_gain * yaw_to_goal
    
    return [vx, vy, wz]
```

---

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `nav_cmd_target_speed` | 0.80 | 目标前进速度 |
| `nav_cmd_min_speed` | 0.18 | 最小速度 |
| `nav_cmd_max_speed` | 1.10 | 最大速度 |
| `nav_cmd_max_lateral` | 0.25 | 最大横向速度 |
| `nav_cmd_yaw_gain` | 1.35 | 偏航角增益 |
| `nav_cmd_max_yaw` | 0.95 | 最大偏航角速度 |
| `nav_cmd_goal_slow_radius` | 0.80 | 接近目标减速半径 |
| `nav_cmd_enable_scanner_avoidance` | True | 启用扫描避障 |
| `nav_cmd_scanner_max_m` | 2.5 | 扫描最大距离 |
| `nav_cmd_min_valid_ray_ratio` | 0.05 | 最小有效射线比例 |
| `nav_cmd_front_block_m` | 0.55 | 前方障碍物阈值 |
| `nav_cmd_deadend_m` | 0.75 | 死胡同判定阈值 |
| `nav_cmd_avoid_yaw` | 0.75 | 避障转向角度 |

---

## 避障扫描器集成

### nav_scanner 数据流

```
env.scene.sensors["nav_scanner"]
    → ray hits (distance, valid_mask)
    → sector reduction (16 sectors)
    → front / left / right clearance
    → wall_close / deadend flags
```

### 避障逻辑

```python
front_m, left_m, right_m = read_nav_clearance(env)

# 前方障碍物检测
blocked = front_m < front_block_m  # 0.55m

# 死胡同检测
deadend = max(left_m, right_m) < deadend_m  # 0.75m

# 避障动作
if blocked or deadend:
    # 选择转向方向（哪边空间大往哪边转）
    turn_sign = 1 if left_m >= right_m else -1
    wz += turn_sign * avoid_yaw  # 0.75 rad/s
    vx = min(vx, 0.28)          # 减速到 0.28
    vy = 0                       # 消除横向速度
```

### 扫描数据清洗

```python
def sanitize_distance_m(distance_m, valid_mask, cfg):
    """清洗原始扫描数据"""
    # 处理 NaN, Inf, 负值, 超大值
    finite = torch.isfinite(raw)
    nan_mask = torch.isnan(raw)
    inf_mask = torch.isinf(raw)
    negative_mask = finite & (raw < 0.0)
    huge_mask = finite & (raw > huge_limit)
    
    # 无效射线用 max_clearance 填充
    sanitized = torch.where(valid, raw.clamp_min(0.0), max_value)
    sanitized = torch.nan_to_num(sanitized, nan=max_m, posinf=max_m, neginf=0.0)
    
    return sanitized, valid, stats
```

---

## 坐标系变换

### 世界坐标系 → 机器人坐标系

```python
cos_yaw = torch.cos(yaw)
sin_yaw = torch.sin(yaw)

# 目标相对位置（机器人坐标系）
dx_b = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
dy_b = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]

# 目标方向误差
yaw_err = atan2(dy_b, dx_b)
```

### 速度命令 → 机器人动作

导航层生成的 `[vx, vy, wz]` 被写入：
1. `obs[:, 6:9]` — policy 观测的速度命令切片
2. `command_manager["base_velocity"]` — 奖励计算使用的命令

确保 actor 输入和 reward 使用的命令一致。

---

## Track 模式观测扩展

### Policy Obs（305维）

```
obs = [proprio(45) | height_scan(256) | goal(4)]
                              ↓
goal(4) = [goal_local_x, goal_local_y, dist, yaw_err]
```

- `goal_local_x, goal_local_y`：目标在机器人坐标系下的相对位置
- `dist`：到目标的欧氏距离
- `yaw_err`：目标方向与机器人朝向的夹角

### Critic Obs（320维）

与 policy 保持同步，同样拼接 goal(4) 特征。

---

## 失败防护

### 信号缺失处理

```python
if not valid.any():
    # 首次警告，保持原始命令
    logger.warning("TrackTensorBridge goal/robot invalid; keeping original command")
    return None  # 使用原始速度命令
```

### 形状探测兼容

```python
def _is_observation_shape_probe():
    """ObservationManager 初始化时 probing obs shape，不触发 fail-fast"""
    # 检查调用栈，如果在 _prepare_terms 中，返回 True
```

确保 ObservationManager 初始化时不会因为 goal_positions 未创建而崩溃。

---

## 训练配置

### TrackConfig 关键参数

```python
class TrackConfig(StageConfig):
    name = "navigation"
    task_type = "track"
    
    # 观测维度
    num_goal_obs = 4
    num_critic_observations = 320
    
    # 导航命令注入
    enable_track_nav_command = True
    nav_cmd_target_speed = 0.80
    nav_cmd_min_speed = 0.18
    nav_cmd_max_speed = 1.10
    nav_cmd_max_lateral = 0.25
    nav_cmd_yaw_gain = 1.35
    nav_cmd_max_yaw = 0.95
    nav_cmd_goal_slow_radius = 0.80
    nav_cmd_enable_scanner_avoidance = True
    
    # 保守微调
    lr = 1e-4
    num_steps_per_env = 64
    num_learning_epochs = 5
    num_mini_batches = 8
```

### 从 Standard 迁移到 Track

1. Standard 阶段训练好 locomotion 策略（~8000-15000 iterations）
2. 保存模型 checkpoint
3. 切换 `Config.CURRENT = TrackConfig`
4. `preload_model=true` 加载 Standard checkpoint
5. 继续训练导航能力（~5000-10000 iterations）
