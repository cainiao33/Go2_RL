# agent_diy/ — [diy] 算法槽位（框架别名，转发至 agent_ppo）

## 为什么这个目录只有 4 个文件

腾讯开悟框架的 `conf/algo_conf_legged_robot_competition_26.toml` 要求两个平行算法槽位：

- `[ppo]` → `agent_ppo.agent.Agent` + `agent_ppo.workflow.train_workflow.workflow`
- `[diy]` → `agent_diy.agent.Agent` + `agent_diy.workflow.train_workflow.workflow`

`train_test.py` 只接受 `"ppo"` / `"diy"` 两个算法名，且平台按模块路径加载入口——删除本目录会直接破坏 `[diy]` 槽，因此目录必须保留。

本目录的 `agent.py` 与 `workflow/train_workflow.py` 是 `agent_ppo` 对应文件的**字节级副本**（由 `tests/test_agent_diy_slot.py` 持续校验），且它们 import 的全部实现（definition / conf / model / algorithm）都来自 `agent_ppo.*`。也就是说：**选择 `algorithm_name = "diy"` 时实际运行的就是 `agent_ppo` 全套代码**，两个槽位行为完全一致。

## 2026-09-29 死代码清理记录

清理前本目录有 18 个文件，其中 14 个（`conf/`、`feature/`、`algorithm/`、`model/` 全部内容）是**零引用死代码**——全仓库没有任何 `from agent_diy` / `import agent_diy`，框架只加载 `agent_diy.agent` 与 `agent_diy.workflow.train_workflow` 两个入口。这批文件是被否弃的「goal 拼接观测（policy 305 / critic 320 维）」实验路线遗留，已被 `agent_ppo` 的「命令注入（观测保持 301/316 维）」路线取代：

- `conf/conf.py` 自己都硬编码加载 `agent_ppo/conf/train_env_conf_*.toml`，从不读本目录的 TOML——编辑本目录的 TOML 对训练零效果
- `conf/train_env_conf_standard_locomotion.toml` 名为 standard、内容实为 track 微调配置（误导性命名）
- `conf/train_env_conf_track_navigation.toml` 头注释引用的 `HierarchicalNavConfig` 类在仓库任何 `.py` 中都不存在
- `algorithm/__init__.py` import 了不存在的 `.algorithm_ppo`（目录里只有改名后的 `algorithm.py`）——潜伏 bug，任何 `import agent_diy.algorithm` 都会抛 `ModuleNotFoundError`

完整历史版本存档于 git 初始提交 `a3ff697`。如需复活 goal 拼接 / 层级导航实验：`git show a3ff697:agent_diy/conf/conf.py` 等按需恢复，并在本目录内恢复**独立 import**（不要继续转发 `agent_ppo`），同时更新 `tests/test_agent_diy_slot.py` 的转发契约断言。

## 平台侧验证清单（本地无法执行）

平台模块（`kaiwudrl` / `common_python` / `tools.*`）仅存在于腾讯开悟平台（<https://aiarena.tencent.com/>，**申请制**，需注册申请并通过审核），本地无法运行 `train_test.py`。本次清理后在平台侧应验证：

1. `train_test.py` 保持 `algorithm_name = "ppo"` → 启动训练，确认日志出现 `Stage: navigation, task_type: track`（agent_ppo 路线）
2. 改为 `algorithm_name = "diy"` → 再次启动，确认行为与 1 完全一致（同一套代码的转发槽位）
3. 若任一槽位启动失败，对照本目录两个入口文件的 import 是否仍指向 `agent_ppo.*`
