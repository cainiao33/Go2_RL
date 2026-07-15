# 代码结构说明

> 2026腾讯开悟人工智能全球公开赛 | D02 四足机器人强化学习挑战

---

## 项目目录

```
Go2_RL/
├── README.md                      # 项目简介与亮点
├── train_test.py                  # 训练入口
├── kaiwu.json                     # 赛题配置
│
├── agent_diy/                     # DIY 算法实现（自定义 PPO）
│   ├── algorithm/
│   │   └── algorithm.py           # PPO 算法核心
│   │                              #   - 自适应学习率（基于 KL 散度）
│   │                              #   - 价值损失归一化
│   │                              #   - NaN/Inf 三层防护
│   │                              #   - GAE 优势估计
│   ├── feature/
│   │   ├── policy_observation_process.py    # Policy 观测处理
│   │   │                                    #   Standard: 301维 (proprio 45 + height_scan 256)
│   │   │                                    #   Track: 305维 (+ goal 4)
│   │   ├── critic_observation_process.py    # Critic 观测处理
│   │   │                                    #   Standard: 316维 (critic_proprio 60 + height_scan 256)
│   │   │                                    #   Track: 320维 (+ goal 4)
│   │   └── reward_process.py      # 自定义奖励函数（14+ 奖励项）
│   │                              #   - 基础运动驱动
│   │                              #   - 安全与约束
│   │                              #   - 能效优化
│   │                              #   - 步态质量
│   │                              #   - 下楼梯专项
│   │                              #   - Track 导航奖励
│   ├── model/
│   │   └── actor_critic.py        # Actor-Critic 网络
│   │                              #   - Actor: MLP + ELU
│   │                              #   - Critic: MLP + LayerNorm + ELU
│   │                              #   - 正交初始化
│   ├── conf/
│   │   └── conf.py                # 阶段配置
│   │                              #   - StageConfig（基类）
│   │                              #   - LocomotionConfig（Standard 训练）
│   │                              #   - TrackConfig（Track 导航训练）
│   └── workflow/
│       └── train_workflow.py      # 训练工作流
│                                    #   - 数据收集（run_episodes_）
│                                    #   - 策略更新（agent.learn）
│                                    #   - 监控指标上报
│                                    #   - 模型保存
│
├── agent_ppo/                     # PPO 算法实现（平台 baseline）
│   ├── algorithm/
│   │   └── algorithm_ppo.py       # PPO 算法（与 diy 相同架构）
│   ├── feature/
│   │   ├── nav_command.py         # 导航命令注入层（Track 核心）
│   │   │                          #   - compute_track_nav_command
│   │   │                          #   - 避障修正
│   │   │                          #   - 坐标系变换
│   │   ├── nav_signal.py          # 导航信号处理
│   │   │                          #   - NavMeterConfig
│   │   │                          #   - NavRuntimeSignal
│   │   │                          #   - 距离清洗与归一化
│   │   ├── track_tensor_bridge.py # Track 张量桥接
│   │   │                          #   - 读取 goal_positions
│   │   │                          #   - 读取 robot 位置/朝向
│   │   │                          #   - 读取 nav_scanner 数据
│   │   ├── feature_layout.py      # 特征布局定义
│   │   └── reward_process.py      # 奖励处理
│   ├── model/
│   │   └── actor_critic.py        # Actor-Critic 网络
│   ├── conf/
│   │   └── conf.py                # 配置管理
│   └── tool/
│       └── scan.py                # 扫描工具
│
├── docs/                          # 技术文档
│   ├── README.md                  # 项目简介（精简版）
│   ├── TECH_OVERVIEW.md           # 技术方案总览
│   ├── REWARD_ENGINEERING.md      # 奖励工程详解
│   ├── NAVIGATION.md              # 导航策略设计
│   ├── TRAINING.md                # 训练策略与课程学习
│   ├── CODE_STRUCTURE.md          # 代码结构说明（本文档）
│   ├── 开发指南/                  # 赛题开发指南
│   ├── 适配方案/                  # 冠军方案适配分析
│   └── 腾讯开悟强化学习框架/       # 框架文档
│
└── .vscode/
    └── launch.json                # VSCode 调试配置
```

---

## 核心模块职责

### `agent_diy/` vs `agent_ppo/`

| 模块 | 用途 | 说明 |
|------|------|------|
| `agent_diy` | 自定义算法 | 选手自行实现的 PPO 算法，可自由修改 |
| `agent_ppo` | 平台 baseline | 平台提供的 PPO 实现，作为参考和对比 |

两个模块的算法架构完全相同，区别仅在于：
- `agent_diy` 的 `reward_process.py` 包含更多自定义奖励
- `agent_ppo` 的 `nav_command.py` 包含导航命令注入层实现

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

---

## 关键文件速查

| 文件 | 功能 | 修改频率 |
|------|------|----------|
| `agent_diy/feature/reward_process.py` | 自定义奖励 | 高 |
| `agent_diy/conf/conf.py` | 阶段配置切换 | 中 |
| `agent_diy/model/actor_critic.py` | 模型架构 | 低 |
| `agent_diy/algorithm/algorithm.py` | PPO 算法 | 低 |
| `agent_ppo/feature/nav_command.py` | 导航命令 | 中 |
| `train_test.py` | 训练入口 | 低 |
