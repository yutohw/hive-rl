"""File-based messaging between the trainer and the workers.

Every file lives in ``config.comms_dir``. Writes go through a temporary file
and ``os.replace()`` so a reader never sees a half-written file.
"""

import contextlib
import glob
import json
import os
import time
from typing import Any

import torch

from hive_rl import config


# ===========================
# Paths
# ===========================
def assignment_path(worker_id: str, round_num: int) -> str:
    return os.path.join(config.comms_dir, f"assignment_{worker_id}_{round_num}.json")


def rollout_path(worker_id: str, round_num: int) -> str:
    return os.path.join(config.comms_dir, f"rollout_{worker_id}_{round_num}.pt")


def weights_round_path(round_num: int) -> str:
    return os.path.join(config.comms_dir, f"weights_round_{round_num}.pth")


def done_flag_path(worker_id: str, round_num: int) -> str:
    return os.path.join(config.comms_dir, f"done_{worker_id}_{round_num}.flag")


def worker_hip_path(worker_id: str) -> str:
    return os.path.join(config.base_dir, f"worker_{worker_id}.hip")


# ===========================
# Atomic write helpers
# ===========================
def atomic_write_json(
    path: str, data: dict[str, Any], retries: int = 50, delay: float = 0.05
) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)

    for _ in range(retries):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            time.sleep(delay)

    os.replace(tmp_path, path)


def atomic_torch_save(path: str, obj: Any, retries: int = 50, delay: float = 0.05) -> None:
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)

    for _ in range(retries):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            time.sleep(delay)

    os.replace(tmp_path, path)


def safe_remove(path: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(path)


def cleanup_old_rounds(round_num: int, keep: int = 3) -> None:
    """Delete comms files from rounds older than ``round_num - keep``."""
    cutoff = round_num - keep
    if cutoff < 1:
        return
    patterns = ("assignment_*_*.json", "rollout_*_*.pt", "weights_round_*.pth", "done_*_*.flag")
    for pattern in patterns:
        for f in glob.glob(os.path.join(config.comms_dir, pattern)):
            name = os.path.basename(f)
            try:
                r = int(name.rsplit("_", 1)[-1].split(".")[0])
            except ValueError:
                continue
            if r <= cutoff:
                safe_remove(f)
