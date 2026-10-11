"""Guards against renaming network attributes, which would break saved .pth files."""

from hive_rl import config
from hive_rl.model import ActorCritic

EXPECTED_KEYS = [
    "log_std",
    "actor.0.weight",
    "actor.0.bias",
    "actor.2.weight",
    "actor.2.bias",
    "actor.4.weight",
    "actor.4.bias",
    "actor.6.weight",
    "actor.6.bias",
    "critic.0.weight",
    "critic.0.bias",
    "critic.2.weight",
    "critic.2.bias",
    "critic.4.weight",
    "critic.4.bias",
    "critic.6.weight",
    "critic.6.bias",
]


def test_state_dict_keys_unchanged() -> None:
    model = ActorCritic(config.state_dim, config.action_dim)
    assert sorted(model.state_dict().keys()) == sorted(EXPECTED_KEYS)
