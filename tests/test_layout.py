# -*- coding: UTF-8 -*-
"""观测布局常量断言（零依赖）。

这些维度是网络输入 ABI：BasePolicyLayout.DIM / BaseCriticLayout.DIM 必须与
StageConfig 的观测维度、以及 checkpoint 的网络输入层完全一致——漂移即崩。

注意：agent_ppo/feature/__init__.py 会急切 import 观测处理（其依赖平台侧
tools.*，本地必然 ImportError），因此这里按文件路径直接加载 feature_layout.py
（该文件本身零 import）。
"""

import importlib.util
from pathlib import Path

_LAYOUT_PATH = (
    Path(__file__).resolve().parent.parent / "agent_ppo" / "feature" / "feature_layout.py"
)


def _load_layout_module():
    spec = importlib.util.spec_from_file_location("feature_layout_under_test", _LAYOUT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fl = _load_layout_module()

BasePolicyLayout = fl.BasePolicyLayout
BaseCriticLayout = fl.BaseCriticLayout
GoalFeatureLayout = fl.GoalFeatureLayout
HistoryFeatureLayout = fl.HistoryFeatureLayout
NavFeatureLayout = fl.NavFeatureLayout
AuxTargetLayout = fl.AuxTargetLayout


def test_policy_base_layout_is_301():
    assert BasePolicyLayout.DIM == 301
    assert BasePolicyLayout.PROPRIO == slice(0, 45)
    assert BasePolicyLayout.HEIGHT_SCAN == slice(45, 301)


def test_policy_slices_contiguous_no_gap_no_overlap():
    segments = [
        BasePolicyLayout.BASE_ANG_VEL,
        BasePolicyLayout.PROJECTED_GRAVITY,
        BasePolicyLayout.VELOCITY_COMMANDS,
        BasePolicyLayout.JOINT_POS_REL,
        BasePolicyLayout.JOINT_VEL_REL,
        BasePolicyLayout.LAST_ACTION,
        BasePolicyLayout.HEIGHT_SCAN,
    ]
    cursor = 0
    for seg in segments:
        assert seg.start == cursor, f"slice {seg} 起点与上一段终点 {cursor} 不衔接"
        assert seg.stop > seg.start
        cursor = seg.stop
    assert cursor == BasePolicyLayout.DIM


def test_velocity_command_slice_is_nav_injection_abi():
    # 命令注入路线的 ABI：nav_command.py 通过改写 obs[:, 6:9] 注入导航速度命令
    assert BasePolicyLayout.VELOCITY_COMMANDS == slice(6, 9)


def test_critic_base_layout_is_316():
    assert BaseCriticLayout.DIM == 316
    assert BaseCriticLayout.CRITIC_PROPRIO == slice(0, 60)
    assert BaseCriticLayout.HEIGHT_SCAN == slice(60, 316)


def test_extended_layout_dims():
    # v2.1 协议预留的扩展布局（当前 301/316 模型未使用，但常量必须自洽）
    assert GoalFeatureLayout.DIM == 16
    assert NavFeatureLayout.DIM == 84
    assert HistoryFeatureLayout.DIM == 72
    assert AuxTargetLayout.DIM == 12


def test_aux_layout_indices_dense():
    indices = [getattr(AuxTargetLayout, name) for name in AuxTargetLayout.CANONICAL_NAMES]
    assert sorted(indices) == list(range(AuxTargetLayout.DIM))
