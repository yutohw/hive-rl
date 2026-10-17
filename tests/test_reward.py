"""Reward processing: per-step rewards become discounted returns and GAE advantages.

The per-step reward itself is VEX inside the Houdini scene (the wrangles that
feed the ``Reward`` node), so it can't run here. This tests the Python half:
what ``process_rollout`` does with the rewards an episode produced."""

import numpy as np
import pytest

from hive_rl import config
from hive_rl.ppo import compute_gae, process_rollout


def test_rewards_become_discounted_returns_and_normalised_advantages() -> None:
    g = config.gamma
    rewards = [1.0, 0.0, 2.0]
    values = [0.5, 0.5, 0.5]

    returns, advantages = process_rollout(rewards, values)

    assert returns == pytest.approx([1.0 + g * 0.0 + g * g * 2.0, 0.0 + g * 2.0, 2.0])

    assert compute_gae(rewards, values)[-1] == pytest.approx(2.0 - 0.5)

    assert np.mean(advantages) == pytest.approx(0.0, abs=1e-6)
    assert np.std(advantages) == pytest.approx(1.0, rel=1e-4)
