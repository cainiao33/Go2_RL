# 代码结构说明

> 2026腾讯开悟人工智能全球公开赛 | D02 四足机器人强化学习挑战

---

## 项目目录

```
Go2_RL/
├── README.md                      # 项目简介与亮点
├── AGENTS.md                      # AI 编码代理说明（结构/约定/平台依赖清单）
├── train_test.py                  # 训练入口（algorithm_name 当前为 "ppo"）
├── kaiwu.json                     # 平台项目元数据（project_code / version）
├── pyproject.toml                 # 工具配置（ruff / pytest；故意不声明可安装依赖）
│
├── agent_ppo/                     # ★ 主开发线：全部算法实现与 Track 导航自研代码
│   ├── agent.py                   # 智能体入口（Agent：predict / exploit / learn / save / load）
│   ├── algorithm/
│   │   └── algorithm_ppo.py       # PPO 核心（KL 自适应 lr / value loss 归一化 / NaN-Inf 三层防护 / GAE）
│   ├── model/
│   │   └── actor_critic.py        # Actor MLP + Critic MLP（LayerNorm），仅依赖 torch
│   ├── feature/                   # 特征处理
│   │   ├── definition.py          # ObsData / ActData / RolloutStorage（依赖平台 common_python）
│   │   ├── policy_observation_process.py   # Policy 观测（301 = proprio 45 + scan 256）
│   │   ├── critic_observation_process.py   # Critic 观测（316 = critic_proprio 60 + scan 256）
│   │   ├── reward_process.py      # 自定义奖励（20 个 _reward_ 函数，六层结构）
│   │   ├── nav_command.py         # Track 导航命令注入（改写 obs[:,6:9]，保持 301 维 ABI）
│   │   ├── track_tensor_bridge.py # 环境张量桥接（goal / robot / nav_scanner 提取）
│   │   ├── nav_signal.py          # 导航信号（NavMeterConfig / 距离清洗归一化）
│   │   └── feature_layout.py      # 观测布局常量（零依赖；tests/test_layout.py 锁定）
│   ├── conf/
│   │   ├── conf.py                # StageConfig / LocomotionConfig / TrackConfig + Config.CURRENT
│   │   ├── monitor_builder.py     # 监控面板配置（依赖平台 kaiwudrl）
│   │   ├── train_env_conf_standard_locomotion.toml  # Standard 训练配置（mode=standard）
│   │   └── train_env_conf_track_navigation.toml     # Track 训练配置（mode=track）
│   ├── workflow/
│   │   └── train_workflow.py      # 训练工作流（数据收集 → 策略更新 → 监控 → 保存）
│   └── tool/
│       └── scan.py                # 静态关键词扫描工具（纯 stdlib，无训练副作用）
│
├── agent_diy/                     # [diy] 算法槽位（框架别名，转发 agent_ppo）
│   ├── agent.py                   # 与 agent_ppo/agent.py 字节相同，import agent_ppo.*
│   ├── workflow/
│   │   └── train_workflow.py     # 同上（字节相同转发）
│   └── README.md                  # 槽位说明 + 2026-09-29 死代码清理记录
│
├── conf/                          # KaiwuDRL 框架路由层（平台强制布局）
│   ├── algo_conf_legged_robot_competition_26.toml   # [ppo]/[diy] → 入口模块映射
│   ├── app_conf_legged_robot_competition_26.toml    # rl_helper / policy builder 选择
│   └── configure_app.toml         # 样本池 / 批次 / 模型保存频率（[app] 节）
│
├── isaac_env/                     # 环境占位目录（仅含空 __init__.py，Isaac Lab 环境代码由平台注入）
│
├── tests/                         # 本地静态测试（平台无关，CI 两档运行）
│   ├── conftest.py                # sys.path 引导
│   ├── test_layout.py             # 观测布局 ABI（301/316，零依赖）
│   ├── test_agent_diy_slot.py     # agent_diy 转发槽位契约（零依赖）
│   ├── test_configs.py            # TOML / JSON 解析与关键值（stdlib 优先）
│   ├── test_conf_module.py        # agent_ppo.conf.conf 模块断言（需 toml 包）
│   └── test_actor_critic.py       # ActorCritic 前向 / 参数量（需 torch CPU）
│
├── .github/workflows/ci.yml       # CI：静态门 + torch-CPU 门
│
├── docs/                          # 技术文档（中文）
│   ├── TECH_OVERVIEW.md           # 技术方案总览
│   ├── REWARD_ENGINEERING.md      # 奖励工程详解
│   ├── NAVIGATION.md              # 导航策略设计
│   ├── TRAINING.md                # 训练策略与课程学习
│   ├── CODE_STRUCTURE.md          # 代码结构说明（本文档）
│   ├── 分布式计算框架.md           # KaiwuDRL 架构说明（平台文档）
│   ├── 强化学习系列系统技术标准.md  # 技术标准（平台文档）
│   ├── 开发指南/                   # 赛题开发指南（平台文档）
│   ├── 腾讯开悟强化学习框架/        # 框架文档（平台文档）
│   ├── 适配方案/                   # 自研适配方案 + 审查or建议/
│   └── 其他工具/                   # 日志与监控
│
└── .vscode/
    └── launch.json                # VS Code 调试配置（python 路径为平台容器内路径）
```

---

## 核心模块职责

### `agent_ppo/` vs `agent_diy/`

| 模块 | 用途 | 说明 |
|------|------|------|
| `agent_ppo` | ★ 主开发线 | 全部算法实现与 Track 导航自研代码（nav_command / track_tensor_bridge / nav_signal）所在；`train_test.py` 默认走此槽 |
| `agent_diy` | `[diy]` 槽位别名 | 框架强制保留的第二槽位；入口文件是 `agent_ppo` 的字节级转发副本，选择 `"diy"` 实际运行同一套代码（详见 `agent_diy/README.md`） |

历史上的「goal 拼接观测（policy 305 / critic 320 维）」实验路线代码已于 2026-09-29 从 `agent_diy` 清理（零引用死代码，完整存档于 git 初始提交 `a3ff697`）；现行路线为**命令注入**（观测保持 301/316 维，见 `docs/NAVIGATION.md`）。

### 训练入口

```python
# train_test.py
algorithm_name = "ppo"  # 或 "diy"（两者运行同一套 agent_ppo 代码）
```

### 切换训练阶段

```python
# agent_ppo/conf/conf.py（唯一生效位置——agent_diy 已无 conf）
class Config:
    CURRENT = TrackConfig  # LocomotionConfig → Standard 训练
                         # TrackConfig → Track 导航训练
```

---

## 关键文件速查

| 文件 | 功能 | 修改频率 |
|------|------|----------|
| `agent_ppo/feature/reward_process.py` | 自定义奖励 | 高 |
| `agent_ppo/conf/conf.py` | 阶段配置切换（`Config.CURRENT`） | 中 |
| `agent_ppo/feature/nav_command.py` | 导航命令注入 | 中 |
| `agent_ppo/model/actor_critic.py` | 模型架构 | 低 |
| `agent_ppo/algorithm/algorithm_ppo.py` | PPO 算法 | 低 |
| `train_test.py` | 训练入口 | 低 |
