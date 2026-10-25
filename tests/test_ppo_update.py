"""One PPO update on a small random batch reports finite losses and changes the weights."""

import math

import torch

from hive_rl import config
from hive_rl.model import ActorCritic, TanhNormal
from hive_rl.ppo import ppo_update


def test_ppo_update_reports_finite_losses() -> None:
    torch.manual_seed(0)
    model = ActorCritic(config.state_dim, config.action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    states = torch.randn(16, config.state_dim)
    with torch.no_grad():
        mean, std, _ = model(states)
        dist = TanhNormal(mean, std)
        actions, pre_tanhs = dist.sample()
        logps = dist.log_prob(actions, pre_tanhs)
    before = [p.detach().clone() for p in model.parameters()]

    stats = ppo_update(
        model,
        optimizer,
        states.tolist(),
        actions.tolist(),
        pre_tanhs.tolist(),
        logps.tolist(),
        torch.randn(16).tolist(),
        torch.randn(16).tolist(),
        config.entropy_coef_start,
    )

    assert set(stats) == {"policy", "value", "entropy", "approx_kl", "clip_fraction"}
    assert all(math.isfinite(v) for v in stats.values())
    assert 0.0 <= stats["clip_fraction"] <= 1.0
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))
