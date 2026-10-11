"""Rollout worker: one headless Houdini instance that plays episodes.

Run with hython, passing the worker ID::

    hython -m hive_rl.worker 1
"""

import json
import os
import sys
import time

import hou
import torch

from hive_rl import comms, config
from hive_rl.model import ActorCritic, TanhNormal, scale_actions
from hive_rl.ppo import cap_transitions, process_rollout


# ===========================
# Houdini helpers
# ===========================
def get_attr(node: str, name: str) -> object:
    """Read a detail attribute from the geometry of the node at ``node``."""
    return hou.node(node).geometry().attribValue(name)


def set_parm(node: str, parm: str, value: float) -> None:
    hou.node(node).parm(parm).set(value)


def cook(node: str) -> None:
    hou.node(node).cook(force=True)


def reset_environment() -> None:
    set_parm(f"{config.geo_node}/Steer", "value1v1", 0.0)
    set_parm(f"{config.geo_node}/Throttle", "value1v1", 0.0)
    set_parm(f"{config.geo_node}/Brake", "value1v1", 0.0)

    hou.node(f"{config.geo_node}/Solver").parm("resimulate").pressButton()

    hou.setFrame(1)
    cook(f"{config.geo_node}/Solver")
    cook(f"{config.geo_node}/Reward")
    cook(f"{config.geo_node}/Stop")
    cook(f"{config.geo_node}/State")
    cook(f"{config.geo_node}/Output")


def execute_training_recorder(round_number: int, return_value: float, track_value: int) -> None:
    set_parm(f"{config.geo_node}/Return", "value1v1", float(return_value))
    cook(f"{config.geo_node}/Return")
    set_parm(f"{config.geo_node}/Track", "value1v1", float(track_value))
    cook(f"{config.geo_node}/Track")
    recorder = hou.node(f"{config.geo_node}/Training_Recorder_01")
    if recorder:
        set_parm(f"{config.geo_node}/Training_Recorder_01", "loadfromdisk", 0)
        recorder.parm("frameoverride").set(round_number)
        recorder.cook(force=True)
        recorder.parm("execute").pressButton()


def select_track(track_index: int) -> None:
    set_parm(f"{config.geo_node}/switch1", "input", track_index)


# ===========================
# Episode
# ===========================
def run_episode(
    model: ActorCritic,
) -> tuple[list, list, list, list[float], list[float], list[float], float, int]:
    """Play one episode with a sampled policy.

    Returns ``(states, actions, pre_tanhs, rewards, values, logps,
    episode_return, last_step_index)``.
    """
    state = get_attr(f"{config.geo_node}/State", "state")

    states, actions, pre_tanhs, rewards, values, logps = [], [], [], [], [], []
    episode_return = 0.0
    speed_zero_counter = 0
    step = 0

    # `step` is read after the loop, so B007 does not apply here.
    for step in range(config.max_steps):  # noqa: B007
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            mean, std, value = model(state_t)
            dist = TanhNormal(mean, std)
            raw_action, u = dist.sample()
            logp = dist.log_prob(raw_action, u)

        final = scale_actions(raw_action)
        steer_delta = float(final[0, 0])
        throttle = float(final[0, 1])
        brake = float(final[0, 2])

        set_parm(f"{config.geo_node}/Steer", "value1v1", steer_delta)
        set_parm(f"{config.geo_node}/Throttle", "value1v1", throttle)
        set_parm(f"{config.geo_node}/Brake", "value1v1", brake)

        hou.setFrame(hou.frame() + 1)

        cook(f"{config.geo_node}/Reward")
        cook(f"{config.geo_node}/State")
        cook(f"{config.geo_node}/Output")
        cook(f"{config.geo_node}/Stop")

        reward = float(get_attr(f"{config.geo_node}/Reward", "reward"))
        stop_flag = int(get_attr(f"{config.geo_node}/Stop", "stop"))
        speed = float(get_attr(f"{config.geo_node}/State", "speed"))

        if speed <= 10:
            speed_zero_counter += 1
        else:
            speed_zero_counter = 0

        done = stop_flag > 0 or speed_zero_counter >= 20

        states.append(state)
        actions.append(raw_action.squeeze(0).numpy())
        pre_tanhs.append(u.squeeze(0).numpy())
        rewards.append(reward)
        values.append(value.item())
        logps.append(logp.item())

        episode_return += reward

        if done:
            break

        state = get_attr(f"{config.geo_node}/State", "state")

    return states, actions, pre_tanhs, rewards, values, logps, episode_return, step


# ===========================
# Trainer communication
# ===========================
def wait_for_assignment(worker_id: str, round_num: int, poll_interval: float = 0.05) -> dict:
    path = comms.assignment_path(worker_id, round_num)
    while True:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(poll_interval)


def load_weights(
    model: ActorCritic, weights_file: str, retries: int = 100, delay: float = 0.05
) -> bool:
    for _ in range(retries):
        try:
            model.load_state_dict(torch.load(weights_file, weights_only=True))
            return True
        except (RuntimeError, EOFError, OSError):
            time.sleep(delay)
    return False


def main() -> None:
    worker_id = sys.argv[1]

    config.ensure_dirs()
    config.seed_process(worker_id)

    hou.hipFile.load(comms.worker_hip_path(worker_id))

    model = ActorCritic(config.state_dim, config.action_dim)

    print(f"=== Worker {worker_id} started, waiting for assignments ===")

    round_num = 0
    while True:
        round_num += 1

        assignment = wait_for_assignment(worker_id, round_num)

        track = assignment["track"]
        should_record = assignment["record"]
        weights_file = assignment["weights_file"]

        if weights_file and os.path.exists(weights_file) and not load_weights(model, weights_file):
            print(
                f"Worker {worker_id}: failed to load weights for round {round_num}, "
                "using previous weights"
            )

        select_track(track)
        reset_environment()

        states, actions, pre_tanhs, rewards, values, logps, episode_return, steps_taken = (
            run_episode(model)
        )

        returns, advantages = process_rollout(rewards, values)
        states, actions, pre_tanhs, logps, returns, advantages = cap_transitions(
            states, actions, pre_tanhs, logps, returns, advantages
        )

        comms.atomic_torch_save(
            comms.rollout_path(worker_id, round_num),
            {
                "states": states,
                "actions": actions,
                "pre_tanhs": pre_tanhs,
                "logps": logps,
                "returns": returns,
                "advantages": advantages,
                "episode_return": episode_return,
                "steps_taken": steps_taken,
                "track": track,
                "worker_id": worker_id,
            },
        )

        if should_record:
            execute_training_recorder(round_num, episode_return, track)

        with open(comms.done_flag_path(worker_id, round_num), "w") as f:
            f.write("done")


if __name__ == "__main__":
    main()
