"""PPO maths: schedules, returns and GAE, transition capping, and the update step."""

import random
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from hive_rl import config
from hive_rl.model import TanhNormal


# ===========================
# Schedules
# ===========================
def cosine_lr(episode: int, num_episodes: int, lr_start: float, lr_min: float) -> float:
    progress = (episode - 1) / max(num_episodes - 1, 1)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return lr_min + (lr_start - lr_min) * cosine


def linear_entropy_coef(
    episode: int, num_episodes: int, coef_start: float, coef_end: float
) -> float:
    progress = (episode - 1) / max(num_episodes - 1, 1)
    return coef_start + (coef_end - coef_start) * progress


# ===========================
# Returns and GAE
# ===========================
def compute_gae(rewards: list[float], values: list[float]) -> list[float]:
    advantages = []
    gae = 0.0
    next_values = values[1:] + [0.0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + config.gamma * next_values[t] - values[t]
        gae = delta + config.gamma * config.gae_lambda * gae
        advantages.insert(0, gae)

    return advantages


def process_rollout(rewards: list[float], values: list[float]) -> tuple[list[float], list[float]]:
    """Return ``(discounted returns, normalised GAE advantages)`` for one episode."""
    returns = []
    discounted = 0.0
    for r in reversed(rewards):
        discounted = r + config.gamma * discounted
        returns.insert(0, discounted)

    advantages = compute_gae(rewards, values)

    if len(advantages) > 1:
        adv_arr = np.array(advantages, dtype=np.float32)
        adv_mean = adv_arr.mean()
        adv_std = adv_arr.std()
        advantages = ((adv_arr - adv_mean) / (adv_std + 1e-8)).tolist()

    return returns, advantages


# ===========================
# Transition cap per episode
# ===========================
def cap_transitions(
    states: list,
    actions: list,
    pre_tanhs: list,
    logps: list,
    returns: list,
    advantages: list,
) -> tuple[list, list, list, list, list, list]:
    """Randomly keep at most ``max_transitions_per_episode`` steps, in order."""
    n = len(states)
    if n <= config.max_transitions_per_episode:
        return states, actions, pre_tanhs, logps, returns, advantages

    idx = sorted(random.sample(range(n), config.max_transitions_per_episode))

    def sub(lst: list) -> list:
        return [lst[i] for i in idx]

    return sub(states), sub(actions), sub(pre_tanhs), sub(logps), sub(returns), sub(advantages)


# ===========================
# PPO update
# ===========================
def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    states: Sequence,
    actions: Sequence,
    pre_tanhs: Sequence,
    log_probs_old: Sequence[float],
    returns: Sequence[float],
    advantages: Sequence[float],
    entropy_coef: float,
) -> dict[str, float]:
    """Run ``ppo_epochs`` of clipped PPO on one batch; return losses averaged over epochs."""
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    pre_tanhs = torch.tensor(pre_tanhs, dtype=torch.float32)
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)

    stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
    for _ in range(config.ppo_epochs):
        mean, std, values = model(states)
        dist = TanhNormal(mean, std)

        log_probs = dist.log_prob(actions, pre_tanhs)
        entropy = dist.entropy_approx(pre_tanhs)

        ratios = torch.exp(log_probs - log_probs_old)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (returns - values).pow(2).mean()
        loss = policy_loss + config.value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            stats["policy"] += policy_loss.item()
            stats["value"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["approx_kl"] += (log_probs_old - log_probs).mean().item()
            clipped = (ratios - 1.0).abs() > config.clip_epsilon
            stats["clip_fraction"] += clipped.float().mean().item()

    return {k: v / config.ppo_epochs for k, v in stats.items()}
