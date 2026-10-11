"""Paths, environment constants, PPO hyperparameters and seeding.

Edit ``base_dir`` and ``num_workers`` before training.
"""

import os
import random

import numpy as np
import torch

# ===========================
# Paths
# ===========================
base_dir = r"D:\Users\YUTO\Documents\Self_Driving\Model\Agent"
comms_dir = os.path.join(base_dir, "comms")
model_dir = os.path.join(base_dir, "model")

model_path = os.path.join(model_dir, "Self_Driving_Agent_Example_01.pth")


def ensure_dirs() -> None:
    """Create ``comms/`` and ``model/`` under ``base_dir`` if they do not exist."""
    os.makedirs(comms_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)


# ===========================
# Houdini environment
# ===========================
geo_node = "/obj/Training_Environment"

state_dim = 18
action_dim = 3  # steer, throttle, brake

# ===========================
# Seeding
# ===========================
base_seed = 27


def seed_process(offset: int | str) -> None:
    """Seed torch, numpy and random with ``base_seed + offset * 1000``."""
    s = base_seed + int(offset) * 1000
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


# ===========================
# PPO hyperparameters
# ===========================
learning_rate = 5e-4
learning_rate_min = 5e-5
gamma = 0.997
gae_lambda = 0.95
clip_epsilon = 0.1
ppo_epochs = 6

entropy_coef_start = 0.05
entropy_coef_end = 0.001

value_coef = 0.5
max_grad_norm = 0.5

log_std_min = -2.0
log_std_max = 0.0

num_episodes = 100000
max_steps = 1440

# ===========================
# Multi-worker
# ===========================
num_workers = 5
worker_ids = [str(i) for i in range(1, num_workers + 1)]

# ===========================
# Transition cap per episode
# ===========================
max_transitions_per_episode = 300
