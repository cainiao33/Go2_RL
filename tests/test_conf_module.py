# -*- coding: UTF-8 -*-
"""agent_ppo.conf.conf 模块级导入与 StageConfig 断言（需 toml 包，不需要平台模块）。

注意：只允许 import 模块本身——conf.py 的平台依赖（common_python/kaiwudrl）
在 load_conf() 函数体内延迟导入，本地调用 Config.load_conf() 必然失败。
"""

import pytest

pytest.importorskip("toml")

from agent_ppo.conf.conf import Config, LocomotionConfig, StageConfig, TrackConfig


def test_current_stage_is_track():
    assert Config.CURRENT is TrackConfig


def test_stage_identities():
    assert TrackConfig.name == "navigation"
    assert TrackConfig.task_type == "track"
    assert LocomotionConfig.name == "locomotion"
    assert LocomotionConfig.task_type == "standard"
    assert LocomotionConfig.task_type in {"standard", "track"}


def test_obs_dims_match_layout_abi():
    # 命令注入路线：goal obs 恒为 0，policy 301 / critic 316 与 feature_layout 一致
    assert StageConfig.num_actions == 12
    assert StageConfig.num_proprio_obs == 45
    assert StageConfig.num_scan == 256
    assert TrackConfig.num_goal_obs == 0
    assert StageConfig.num_proprio_obs + StageConfig.num_scan + TrackConfig.num_goal_obs == 301
    assert TrackConfig.num_critic_observations == 316


def test_track_fine_tune_hyperparams():
    # Track 保守微调超参（与 docs/TECH_OVERVIEW.md 的记载挂钩，漂移即文档失真）
    assert TrackConfig.lr == 1e-4
    assert TrackConfig.num_steps_per_env == 64
    assert TrackConfig.num_mini_batches == 8
    assert TrackConfig.model_save_interval == 100


def test_scanner_avoidance_ships_off():
    # 出厂配置为 False（README/NAVIGATION 已如实标注）；开启只需改 agent_ppo/conf/conf.py 这一行
    assert TrackConfig.nav_cmd_enable_scanner_avoidance is False
    assert TrackConfig.nav_cmd_fail_fast is False
