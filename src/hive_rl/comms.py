import contextlib
import glob
import json
import os
import time
from typing import Any

import torch

from hive_rl import config



def assignment_path(worker_id, round_num):
    return os.path.join(config.config.comms_dir, f"assignment_{worker_id}_{round_num}.json")

def rollout_path(worker_id, round_num):
    return os.path.join(config.comms_dir, f"rollout_{worker_id}_{round_num}.pt")

def weights_round_path(round_num):
    return os.path.join(config.comms_dir, f"weights_round_{round_num}.pth")

def done_flag_path(worker_id, round_num):
    return os.path.join(config.comms_dir, f"done_{worker_id}_{round_num}.flag")

def worker_hip_path(worker_id):
    return os.path.join(config.base_dir, f"worker_{worker_id}.hip")

# ===========================
# Atomic write helpers
# ===========================
def atomic_write_json(path, data, retries=50, delay=0.05):
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

def atomic_torch_save(path, obj, retries=50, delay=0.05):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)

    for _ in range(retries):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            time.sleep(delay)

    os.replace(tmp_path, path)

def safe_remove(path):
    try:
        os.remove(path)
    except (OSError, PermissionError):
        pass

def cleanup_old_rounds(round_num, keep=3):
    cutoff = round_num - keep
    if cutoff < 1:
        return
    for pattern in ("assignment_*_*.json", "rollout_*_*.pt", "weights_round_*.pth", "done_*_*.flag"):
        for f in glob.glob(os.path.join(config.comms_dir, pattern)):
            name = os.path.basename(f)
            try:
                r = int(name.rsplit("_", 1)[-1].split(".")[0])
            except ValueError:
                continue
            if r <= cutoff:
                safe_remove(f)