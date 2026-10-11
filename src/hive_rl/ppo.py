import random
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from hive_rl import config
from hive_rl.model import TanhNormal


def cap_transitions(states, actions, pre_tanhs, logps, returns, advantages):
    n = len(states)
    if n <= config.max_transitions_per_episode:
        return states, actions, pre_tanhs, logps, returns, advantages

    idx = sorted(random.sample(range(n), config.max_transitions_per_episode))
    sub = lambda lst: [lst[i] for i in idx]
    return sub(states), sub(actions), sub(pre_tanhs), sub(logps), sub(returns), sub(advantages)

# ===========================
# Schedules
# ===========================
def cosine_lr(episode, num_episodes, lr_start, lr_min):
    progress = (episode - 1) / max(num_episodes - 1, 1)
    cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
    return lr_min + (lr_start - lr_min) * cosine

def linear_entropy_coef(episode, num_episodes, coef_start, coef_end):
    progress = (episode - 1) / max(num_episodes - 1, 1)
    return coef_start + (coef_end - coef_start) * progress


# ===========================
# GAE
# ===========================
def compute_gae(rewards, values):
    advantages  = []
    gae         = 0.0
    next_values = values[1:] + [0.0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + config.gamma * next_values[t] - values[t]
        gae   = delta + config.gamma * config.gae_lambda * gae
        advantages.insert(0, gae)

    return advantages


def process_rollout(rewards, values):
    returns    = []
    discounted = 0.0
    for r in reversed(rewards):
        discounted = r + config.gamma * discounted
        returns.insert(0, discounted)

    advantages = compute_gae(rewards, values)

    if len(advantages) > 1:
        adv_arr  = np.array(advantages, dtype=np.float32)
        adv_mean = adv_arr.mean()
        adv_std  = adv_arr.std()
        advantages = ((adv_arr - adv_mean) / (adv_std + 1e-8)).tolist()

    return returns, advantages


# ===========================
# PPO Update
# ===========================
def ppo_update(model, optimizer, states, actions, pre_tanhs, log_probs_old, returns, advantages, entropy_coef):
    states        = torch.tensor(states,        dtype=torch.float32)
    actions       = torch.tensor(actions,       dtype=torch.float32)
    pre_tanhs     = torch.tensor(pre_tanhs,     dtype=torch.float32)
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
    returns       = torch.tensor(returns,       dtype=torch.float32)
    advantages    = torch.tensor(advantages,    dtype=torch.float32)

    for _ in range(config.ppo_epochs):
        mean, std, values = model(states)
        dist = TanhNormal(mean, std)

        log_probs = dist.log_prob(actions, pre_tanhs)
        entropy   = dist.entropy_approx(pre_tanhs)

        ratios = torch.exp(log_probs - log_probs_old)
        surr1  = ratios * advantages
        surr2  = torch.clamp(ratios, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss  = (returns - values).pow(2).mean()
        loss        = policy_loss + config.value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
