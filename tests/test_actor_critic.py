# -*- coding: UTF-8 -*-
"""ActorCritic 前向与参数量断言（CPU 即可；actor_critic.py 仅依赖 torch，平台无关）。"""

import pytest

torch = pytest.importorskip("torch")

from agent_ppo.model.actor_critic import ActorCritic


def _model():
    torch.manual_seed(0)
    return ActorCritic(num_obs=301, num_critic_obs=316, num_actions=12)


def test_forward_shapes():
    m = _model()
    obs = torch.zeros(4, 301)
    critic_obs = torch.zeros(4, 316)

    with torch.no_grad():
        assert m.act(obs).shape == (4, 12)
        assert m.act_inference(obs).shape == (4, 12)
        assert m.evaluate(critic_obs).shape == (4, 1)

        m.update_distribution(obs)
        actions = torch.zeros(4, 12)
        assert m.get_actions_log_prob(actions).shape == (4,)
        assert m.action_mean.shape == (4, 12)
        assert m.action_std.shape == (4, 12)


def test_param_count():
    # actor 320,396 + critic 327,425（含 LayerNorm）+ scalar std 12
    m = _model()
    total = sum(p.numel() for p in m.parameters())
    assert total == 647_833


def test_critic_uses_layernorm():
    assert any(isinstance(mod, torch.nn.LayerNorm) for mod in _model().modules())


def test_actor_hidden_dims_match_stage_config():
    # 与 StageConfig.actor_hidden_dims 对齐（conf 改维度时此测试应变红）
    from agent_ppo.conf.conf import Config

    m = _model()
    linears = [mod for mod in m.actor.modules() if isinstance(mod, torch.nn.Linear)]
    expect = list(Config.CURRENT.actor_hidden_dims) + [Config.CURRENT.num_actions]
    assert [ln.out_features for ln in linears] == expect
