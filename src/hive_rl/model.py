"""Policy network and action distribution.

The attribute names in ``ActorCritic`` (``actor``, ``critic``, ``log_std``) are the
state-dict keys saved in every ``.pth`` file. Do not rename them, or existing
checkpoints and the ``Inference_Script`` node will stop loading.
"""

import torch
import torch.nn as nn

from hive_rl import config


class TanhNormal:
    """Normal distribution squashed through ``tanh`` into the range (-1, 1)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.normal = torch.distributions.Normal(mean, std)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tanh(u), u)``."""
        u = self.normal.sample()
        return torch.tanh(u), u

    def log_prob(self, raw_action: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        lp = self.normal.log_prob(u)
        correction = torch.log(1.0 - raw_action.pow(2) + 1e-6)
        return (lp - correction).sum(-1)

    def entropy_approx(self, u: torch.Tensor) -> torch.Tensor:
        raw_action = torch.tanh(u)
        correction = torch.log(1.0 - raw_action.pow(2) + 1e-6)
        return (self.normal.entropy() - correction).sum(-1).mean()


def scale_actions(raw_action: torch.Tensor) -> torch.Tensor:
    """Map raw (-1, 1) actions to steer (-1, 1), throttle (0, 1), brake (0, 1)."""
    steer = raw_action[..., 0:1]
    throttle = (raw_action[..., 1:2] + 1.0) / 2.0
    brake = (raw_action[..., 2:3] + 1.0) / 2.0
    return torch.cat([steer, throttle, brake], dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, action_dim),
        )

        # Initial bias: no steer, some throttle, almost no brake.
        with torch.no_grad():
            self.actor[-1].bias[0].fill_(0.0)
            self.actor[-1].bias[1].fill_(1.5)
            self.actor[-1].bias[2].fill_(-3.0)

        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(mean, std, value)``."""
        mean = self.actor(x)
        std = torch.exp(torch.clamp(self.log_std, config.log_std_min, config.log_std_max))
        value = self.critic(x).squeeze(-1)
        return mean, std, value
