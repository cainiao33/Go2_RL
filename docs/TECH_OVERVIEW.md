# 技术方案总览

> 2026腾讯开悟人工智能全球公开赛 | D02 四足机器人强化学习挑战

---

## 方案选型决策

### 冠军方案能力拆解

```
┌─────────────────────────────────────────────────┐
│ 1. 鲁棒 locomotion（PPO + 丰富域随机化 + 课程）  │ ← 可直接借鉴
│ 2. Teacher-Student 蒸馏（CTS）                   │ ← 本届弱化
│ 3. MoE 多专家路由                                │ ← 不推荐
│ 4. 导航能力（command curriculum + dynamic sigma）│ ← 改造适配
└─────────────────────────────────────────────────┘
```

**核心决策**：以纯 **PPO + 奖励工程 + 课程学习** 为主干。

原因：
- 本届平台 policy obs 已包含 **256维 height_scan**，不再是 blind student
- CTS 蒸馏的"地形知识传递"动机大幅减弱
- MoE 实现复杂度高，性价比不如奖励工程

### 技术路线对比

| 路线 | 复杂度 | 适用性 | 本届建议 |
|------|--------|--------|----------|
| 纯 PPO | ★☆☆ | ★★★★★ | **首选** |
| CTS 蒸馏 | ★★☆ | ★★★☆☆ | 可选第二阶段 |
| MoE | ★★★ | ★★☆☆☆ | 不推荐 |

---

## 分阶段训练路线

| 阶段 | 目标 | 时间占比 | 核心工作 |
|------|------|----------|----------|
| **Phase 1** | Standard 全地形通过率 > 80% | 40% | PPO baseline + 奖励工程 + 课程学习 |
| **Phase 2** | Track 完成率 > 60% | 30% | 加入目标点观测 + 导航奖励 + 命令注入层 |
| **Phase 3** | 提升泛化能力 | 20% | 尝试 CTS 蒸馏或 history buffer |
| **Phase 4** | 精调能耗/姿态 | 10% | 降低域随机化强度，收敛优雅步态 |

---

## 模型架构

### Actor-Critic 网络

```
Actor (Policy):
    obs[301] → Linear(301, 512) → ELU
           → Linear(512, 256) → ELU
           → Linear(256, 128) → ELU
           → Linear(128, 12)  → actions

Critic (Value):
    critic_obs[316] → Linear(316, 512) → ELU
                  → Linear(512, 256) → LayerNorm + ELU
                  → Linear(256, 128) → LayerNorm + ELU
                  → Linear(128, 1)   → value
```

### 观测维度

| 模式 | Policy Obs | Critic Obs | 说明 |
|------|-----------|-----------|------|
| Standard | 301 = 45 + 256 | 316 = 60 + 256 | proprio + height_scan |
| Track | 305 = 45 + 256 + 4 | 320 = 60 + 256 + 4 | + goal 特征 |

- **proprio(45)**：关节位置、速度、IMU 姿态、速度命令等本体感知
- **height_scan(256)**：16×16 前方地面高度扫描
- **goal(4)**：Track 模式下目标点相对位置（local_x, local_y, dist, yaw）

---

## PPO 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `clip_param` | 0.2 | PPO 裁剪参数 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE lambda |
| `lr` | 1e-3 → adaptive | 自适应学习率（基于 KL 散度） |
| `num_learning_epochs` | 5 | 每次更新 epoch 数 |
| `num_mini_batches` | 4 | mini-batch 数量 |
| `desired_kl` | 0.01 | 目标 KL 散度 |
| `entropy_coef` | 0.01 | 熵奖励系数 |
| `max_grad_norm` | 1.0 | 梯度裁剪最大范数 |

### 自适应学习率

```python
kl_mean = mean(KL(new_policy || old_policy))
if kl_mean > desired_kl * 2.0:   lr /= 1.5   # 更新太大，减速
elif kl_mean < desired_kl / 2.0: lr *= 1.5   # 更新太小，加速
```

### 价值损失归一化

按回报方差归一化 value_loss，保持尺度不变性：

```python
returns_var = returns_batch.detach().var() + 1e-8
value_loss = raw_loss / returns_var
```

---

## 可改进方向

- **CTS 蒸馏**：如果 Standard 遇到瓶颈，可尝试 Teacher-Student 架构进一步提升泛化
- **History Buffer**：拼接多帧历史观测，隐式估计速度/加速度趋势
- **MoE 多专家**：针对特定地形训练专门 expert，但实现复杂度高
- **更精细的课程学习**：运行时动态调整命令范围，而非分阶段静态配置
