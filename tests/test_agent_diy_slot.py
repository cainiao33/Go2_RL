# -*- coding: UTF-8 -*-
"""agent_diy 转发槽位契约测试（零依赖）。

agent_diy 是腾讯开悟框架强制的 [diy] 槽位：入口文件转发 agent_ppo，目录内
不允许出现任何被 import 的实现代码（2026-09-29 已清理 14 个零引用死文件，
见 agent_diy/README.md）。本文件锁定该契约，防止死代码回潮或转发被悄悄改坏。
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIY = REPO_ROOT / "agent_diy"

EXPECTED_FILES = [
    "README.md",
    "__init__.py",
    "agent.py",
    "workflow/__init__.py",
    "workflow/train_workflow.py",
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


def test_diy_dir_contains_only_forwarding_files():
    files = sorted(
        p.relative_to(DIY).as_posix()
        for p in DIY.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    assert files == EXPECTED_FILES


def test_diy_entries_forward_to_agent_ppo():
    for rel in ("agent.py", "workflow/train_workflow.py"):
        src = (DIY / rel).read_text(encoding="utf-8")
        assert "agent_ppo." in src, (
            f"{rel} 不再转发 agent_ppo——若是有意独立化，请同步更新本测试与 agent_diy/README.md"
        )
        assert "from agent_diy" not in src and "import agent_diy" not in src


def test_forwarding_files_byte_identical_to_agent_ppo():
    pairs = [
        (DIY / "agent.py", REPO_ROOT / "agent_ppo" / "agent.py"),
        (DIY / "__init__.py", REPO_ROOT / "agent_ppo" / "__init__.py"),
        (DIY / "workflow" / "train_workflow.py", REPO_ROOT / "agent_ppo" / "workflow" / "train_workflow.py"),
        (DIY / "workflow" / "__init__.py", REPO_ROOT / "agent_ppo" / "workflow" / "__init__.py"),
    ]
    for diy_file, ppo_file in pairs:
        assert diy_file.read_bytes() == ppo_file.read_bytes(), (
            f"{diy_file.relative_to(REPO_ROOT)} 与 agent_ppo 版本出现漂移，需确认哪边是有意修改"
        )


def test_no_repo_module_imports_agent_diy():
    # agent_diy 只能被 algo_conf 按模块路径加载，任何 py 文件不得 import 它
    offenders = []
    for py in REPO_ROOT.rglob("*.py"):
        if ".git" in py.parts or "tests" in py.parts:
            continue
        if "agent_diy" in py.read_text(encoding="utf-8"):
            offenders.append(py.relative_to(REPO_ROOT).as_posix())
    assert offenders == [], f"以下文件引用了 agent_diy：{offenders}"


def test_algo_conf_slot_contract():
    algo = _load_toml(REPO_ROOT / "conf" / "algo_conf_legged_robot_competition_26.toml")
    for section in ("ppo", "diy"):
        assert section in algo, f"algo_conf 缺少 [{section}] 槽位"

    diy = algo["diy"]
    for key in ("actor_agent", "learner_agent", "aisrv_agent"):
        assert diy[key] == "agent_diy.agent.Agent"
    assert diy["train_workflow"] == "agent_diy.workflow.train_workflow.workflow"

    ppo = algo["ppo"]
    for key in ("actor_agent", "learner_agent", "aisrv_agent"):
        assert ppo[key] == "agent_ppo.agent.Agent"
    assert ppo["train_workflow"] == "agent_ppo.workflow.train_workflow.workflow"

    # 两个入口文件必须真实存在（槽位映射不可指向空路径）
    assert (DIY / "agent.py").is_file()
    assert (DIY / "workflow" / "train_workflow.py").is_file()
    assert (REPO_ROOT / "agent_ppo" / "agent.py").is_file()
    assert (REPO_ROOT / "agent_ppo" / "workflow" / "train_workflow.py").is_file()
