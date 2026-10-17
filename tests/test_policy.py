"""One policy forward pass: output shapes, the std clamp, and action ranges."""

import math

import pytest
import torch

from hive_rl import config
from hive_rl.model import ActorCritic, TanhNormal, scale_actions


def test_policy_forward_pass_shapes_and_action_ranges() -> None:
    torch.manual_seed(0)
    model = ActorCritic(config.state_dim, config.action_dim)
    batch = torch.randn(4, config.state_dim)

    with torch.no_grad():
        mean, std, value = model(batch)
        raw_action, _ = TanhNormal(mean, std).sample()
        action = scale_actions(raw_action)

    assert tuple(mean.shape) == (4, config.action_dim)
    assert tuple(std.shape) == (config.action_dim,)
    assert tuple(value.shape) == (4,)
    assert torch.isfinite(mean).all() and torch.isfinite(value).all()

    assert std.min().item() >= math.exp(config.log_std_min) - 1e-6
    assert std.max().item() <= math.exp(config.log_std_max) + 1e-6
    with torch.no_grad():
        model.log_std.fill_(5.0)
        _, std_high, _ = model(batch)
    assert std_high.max().item() == pytest.approx(math.exp(config.log_std_max))

    steer, throttle, brake = action[:, 0], action[:, 1], action[:, 2]
    assert steer.abs().max().item() <= 1.0
    assert throttle.min().item() >= 0.0 and throttle.max().item() <= 1.0
    assert brake.min().item() >= 0.0 and brake.max().item() <= 1.0
