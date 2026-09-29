# -*- coding: UTF-8 -*-
"""TOML / JSON 配置解析与关键值断言（stdlib 优先，toml 包兜底）。
配置错误是平台侧最常见的启动失败原因，这里在本地就拦住。
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TOML_FILES = [
    "conf/algo_conf_legged_robot_competition_26.toml",
    "conf/app_conf_legged_robot_competition_26.toml",
    "conf/configure_app.toml",
    "agent_ppo/conf/train_env_conf_standard_locomotion.toml",
    "agent_ppo/conf/train_env_conf_track_navigation.toml",
]


def _load_toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import toml

        with open(path, encoding="utf-8") as f:
            return toml.load(f)
    with open(path, "rb") as f:
        return tomllib.load(f)


def test_all_tomls_parse():
    for rel in TOML_FILES:
        _load_toml(REPO_ROOT / rel)  # 解析失败即抛异常


def test_kaiwu_json_metadata():
    meta = json.loads((REPO_ROOT / "kaiwu.json").read_text(encoding="utf-8"))
    assert meta["project_code"] == "legged_robot_competition_26"
    assert isinstance(meta.get("version"), str) and meta["version"]


def test_standard_toml_is_really_standard():
    cfg = _load_toml(REPO_ROOT / "agent_ppo" / "conf" / "train_env_conf_standard_locomotion.toml")
    assert cfg["terrain"]["mode"] == "standard"
    assert cfg["env"]["num_envs"] == 4096


def test_track_toml_is_really_track():
    cfg = _load_toml(REPO_ROOT / "agent_ppo" / "conf" / "train_env_conf_track_navigation.toml")
    assert cfg["terrain"]["mode"] == "track"


def test_configure_app_defaults():
    cfg = _load_toml(REPO_ROOT / "conf" / "configure_app.toml")
    app = cfg["app"]
    # 与 AGENTS.md §5.4 记载的默认值保持一致
    assert app["replay_buffer_capacity"] == 4096
    assert app["train_batch_size"] == 2048
    assert app["dump_model_freq"] == 200
    assert app["preload_model"] is False
