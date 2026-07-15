# 训练策略与课程学习

> 2026腾讯开悟人工智能全球公开赛 | D02 四足机器人强化学习挑战

---

## 域随机化策略

### 平台可用参数

| 参数 | 默认值 | 建议值 | 理由 |
|------|--------|--------|------|
| `friction_range` | [0.3, 1.5] | [0.2, 2.5] | 更宽范围提升泛化 |
| `push_robots` | true | true | 保持外部推力干扰 |
| `push_interval_s` | 15 | 8 | 更频繁推力提升鲁棒性 |
| `max_push_vel_xy` | 0.5 | 1.5 | 更强推力 |
| `add_noise` | true | true | 观测噪声保持 |

### 通过奖励补偿不可用的域随机化

冠军方案有但平台没有的随机化 → 用奖励替代：

| 冠军随机化 | 替代方案 |
|-----------|----------|
| motor strength randomization | 加大 `joint_torques` 惩罚权重 |
| action delay | 加大 `action_smoothness` 权重 |
| PD gain randomization | 加大 `action_rate` 权重 |
| base COM randomization | 加大 `flat_orientation` 权重 |

---

## 速度命令课程

### 阶段 1（前 30% 训练时间）

保守速度，重点学习稳定行走：

```toml
[commands.ranges]
lin_vel_x = [0.0, 0.8]
lin_vel_y = [-0.3, 0.3]
ang_vel_yaw = [-0.5, 0.5]
```

### 阶段 2（中 40%）

扩大速度范围：

```toml
[commands.ranges]
lin_vel_x = [-0.5, 1.5]
lin_vel_y = [-0.5, 0.5]
ang_vel_yaw = [-1.0, 1.0]
```

### 阶段 3（后 30%）

全速度范围：

```toml
[commands.ranges]
lin_vel_x = [-1.0, 2.0]
lin_vel_y = [-1.0, 1.0]
ang_vel_yaw = [-1.5, 1.5]
```

> ⚠️ 平台不支持运行时动态修改 TOML。需要**分多次训练任务**（preload_model 继续训练）或在 reward 中实现 soft curriculum。

### Soft Curriculum 实现

```python
def _reward_track_lin_vel_xy(self):
    """带动态 sigma 的速度跟踪"""
    base_sigma = 0.25
    # 根据当前地形难度动态调整 sigma
    # difficulty_factor = self._get_terrain_difficulty()
    # sigma = base_sigma + 0.1 * difficulty_factor
    sigma = base_sigma
    error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    return torch.exp(-error / sigma)
```

---

## 超参调整时间表

| 训练阶段 | lr | entropy_coef | push_vel | friction_range |
|----------|-----|-------------|----------|----------------|
| Phase 1 前期 | 1e-3 | 0.01 | 0.5 | [0.3, 1.5] |
| Phase 1 中期 | 5e-4 | 0.005 | 1.0 | [0.2, 2.0] |
| Phase 1 后期 | 3e-4 | 0.003 | 1.5 | [0.2, 2.5] |
| Phase 2 | 5e-4 | 0.01 | 1.0 | [0.3, 1.5] |
| Phase 3 | 1e-4 | 0.001 | 0.5 | [0.5, 1.2] |

---

## 地形课程

```toml
[terrain]
curriculum = true
max_init_terrain_level = 3    # 初始只到 level 3
difficulty_range = [0.0, 1.0] # 逐步解锁全难度
```

平台机制：表现好的机器人会被升级到更难的地形。

### Standard 地形比例

```toml
[terrain.standard.pyramid_slope]
proportion = 0.2
[terrain.standard.pyramid_slope_inv]
proportion = 0.2
[terrain.standard.pyramid_stairs]
proportion = 0.25
[terrain.standard.pyramid_stairs_inv]
proportion = 0.25
[terrain.standard.maze]
proportion = 0.1
```

### Track 赛道配置

```toml
[terrain]
mode = "track"
num_rows = 10
curriculum = true

[terrain.track]
track_length = 5
sub_terrains = ["pyramid_slope", "pyramid_slope_inv", "pyramid_stairs", "pyramid_stairs_inv", "open_entry_maze"]
```

---

## 分阶段训练计划

### Phase 1: Standard Locomotion（8000~15000 iterations）

```
目标：全 5 种地形通过率 > 80%
配置：standard 模式, num_envs=4096, curriculum=true
重点：reward 工程 + 地形课程
验证：distance_score 和 completed_count
```

### Phase 2: Track Navigation（5000~10000 iterations）

```
前置：Phase 1 的模型作为预训练
目标：Track 完成率 > 60%
配置：track 模式, 加入 goal obs, 加入导航 reward
重点：goal_direction + reach_goal + navigation_progress
验证：completed_count / total, time_score
```

### Phase 3: 精调与鲁棒性（3000~5000 iterations）

```
前置：Phase 2 的模型
目标：提升能耗分和姿态分
配置：加大 energy/posture reward 权重
重点：降低域随机化强度，收敛优雅步态
验证：energy_score, pose_score 提升
```

---

## 关键监控指标

### 训练健康度

| 指标 | 健康标准 | 异常信号 |
|------|----------|----------|
| `total_loss` | 持续下降 | 震荡或上升 |
| `entropy_loss` | 缓慢下降 | 过快下降 = 探索不足 |
| `reward_mean` | 持续上升 | 停滞或下降 |
| `policy_loss` | 稳定下降 | 剧烈波动 |
| `value_loss` | 稳定下降 | 发散 |

### Standard 表现

| 指标 | 目标值 | 说明 |
|------|--------|------|
| `distance_score` | > 0.8 | 各地形 |
| `completed_count` | >> `abnormal_count` | 走穿 vs 摔倒 |
| `pose_score` | > 0.7 | 姿态稳定性 |
| `energy_score` | > 0.6 | 能耗效率 |

### Track 表现

| 指标 | 目标值 | 说明 |
|------|--------|------|
| `completed_count / total` | > 0.6 | 完成率 |
| `time_score` | > 0.5 | 时间效率 |
| `pose_score` | > 0.7 | 姿态稳定性 |

---

## 模型保存策略

```python
# conf.py
model_save_interval = 250  # Standard
model_save_interval = 100  # Track（保守微调，更频繁保存）
```

### 评估注意事项

- 评估关闭域随机化和噪声
- 评估 command limit 可能与训练不同
- 确保模型在**无噪声、无推力**下仍然稳定
- 建议训练后期降低噪声和推力强度

---

## NaN/Inf 三层防护

### Layer 1: Loss 检测

```python
if not torch.isfinite(loss):
    logger.warning(f"NaN/Inf loss at step {step}, skipping update")
    continue  # 跳过此 mini-batch
```

### Layer 2: Gradient 检测

```python
for p in actor_critic.parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        optimizer.zero_grad()
        continue  # 跳过 optimizer.step
```

### Layer 3: Std 钳制

```python
safe_std = torch.nan_to_num(std.data, nan=1.0, posinf=1.0e6, neginf=0.0)
std.data.copy_(torch.clamp(safe_std, min=min_std, max=1.0e6))
```

训练过程中**从未因数值问题导致崩溃**。
