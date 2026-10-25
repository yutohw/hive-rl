"""PPO trainer: hands out episodes to the workers and updates the shared policy.

Run with hython::

    hython -m hive_rl.trainer
"""

import os
import time
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from hive_rl import comms, config, tracking
from hive_rl.curriculum import next_track_batch
from hive_rl.model import ActorCritic
from hive_rl.ppo import cosine_lr, linear_entropy_coef, ppo_update


def wait_for_all(round_num: int, poll_interval: float = 0.05) -> None:
    """Block until every worker has written its done flag for ``round_num``."""
    flags = [comms.done_flag_path(wid, round_num) for wid in config.worker_ids]
    while not all(os.path.exists(f) for f in flags):
        time.sleep(poll_interval)


def load_rollout(
    worker_id: str, round_num: int, retries: int = 100, delay: float = 0.05
) -> dict[str, Any]:
    path = comms.rollout_path(worker_id, round_num)
    for _ in range(retries):
        try:
            return torch.load(path, weights_only=False)
        except (RuntimeError, EOFError, OSError):
            time.sleep(delay)
    raise RuntimeError(f"Could not read rollout for worker {worker_id}, round {round_num}")


def main() -> None:
    config.ensure_dirs()
    config.seed_process(0)

    model = ActorCritic(config.state_dim, config.action_dim)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    run = tracking.start_run()

    episode_returns = []
    best_return = -float("inf")
    best_state_dict = None
    global_episode = 0
    last_10_start = int(config.num_episodes * 0.9)
    checkpoint_every_rounds = 100

    print("=== PPO Trainer started (multi-worker) ===")

    round_num = 0
    while global_episode < config.num_episodes:
        round_num += 1
        round_start = time.perf_counter()

        weights_file = comms.weights_round_path(round_num)
        comms.atomic_torch_save(weights_file, model.state_dict())

        tracks = next_track_batch(global_episode + 1, config.num_episodes, config.num_workers)

        current_lr = cosine_lr(
            global_episode + 1, config.num_episodes, config.learning_rate, config.learning_rate_min
        )
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        current_entropy_coef = linear_entropy_coef(
            global_episode + 1,
            config.num_episodes,
            config.entropy_coef_start,
            config.entropy_coef_end,
        )

        should_record = round_num % 2 == 0

        for wid, track in zip(config.worker_ids, tracks, strict=True):
            comms.atomic_write_json(
                comms.assignment_path(wid, round_num),
                {
                    "round": round_num,
                    "track": track,
                    "record": should_record,
                    "weights_file": weights_file,
                },
            )

        wait_for_all(round_num)

        batch_states, batch_actions, batch_pre_tanhs = [], [], []
        batch_logps, batch_returns, batch_advantages = [], [], []

        round_summary = []
        round_results = []
        for wid, track in zip(config.worker_ids, tracks, strict=True):
            rollout = load_rollout(wid, round_num)

            steps_taken = rollout["steps_taken"]
            episode_return = rollout["episode_return"]

            batch_states.extend(rollout["states"])
            batch_actions.extend(rollout["actions"])
            batch_pre_tanhs.extend(rollout["pre_tanhs"])
            batch_logps.extend(rollout["logps"])
            batch_returns.extend(rollout["returns"])
            batch_advantages.extend(rollout["advantages"])

            episode_returns.append(episode_return)
            round_summary.append((track, steps_taken))
            round_results.append((track, episode_return, steps_taken + 1))

            global_episode += 1
            if global_episode >= last_10_start and episode_return > best_return:
                best_return = episode_return
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        losses = None
        if len(batch_states) > 1:
            losses = ppo_update(
                model,
                optimizer,
                batch_states,
                batch_actions,
                batch_pre_tanhs,
                batch_logps,
                batch_returns,
                batch_advantages,
                current_entropy_coef,
            )

        if round_num % checkpoint_every_rounds == 0:
            checkpoint_path = os.path.join(config.model_dir, f"checkpoint_{global_episode}.pth")
            torch.save(model.state_dict(), checkpoint_path)

        comms.cleanup_old_rounds(round_num, keep=3)

        if len(episode_returns) >= 100:
            avg_100 = np.mean(episode_returns[-100:])
        else:
            avg_100 = np.mean(episode_returns)
        summary = " | ".join(f"T{t}:{s}" for t, s in round_summary)
        print(
            f"Round {round_num:5d} | Episodes {global_episode:5d} | {summary} | "
            f"Avg100 {avg_100:8.2f} | LR {current_lr:.2e} | Ent {current_entropy_coef:.4f} | "
            f"LogStd {model.log_std.data.mean().item():.3f}"
        )

        if run is not None:
            metrics = tracking.round_metrics(
                episodes=global_episode,
                results=round_results,
                avg_100=avg_100,
                lr=current_lr,
                entropy_coef=current_entropy_coef,
                log_std=model.log_std.data.mean().item(),
                seconds=time.perf_counter() - round_start,
                losses=losses,
            )
            run.log(metrics, step=global_episode)

    torch.save(best_state_dict, config.model_path)
    print("\nTraining complete.")
    print(f"Best model from last 10% saved (Return {best_return:.2f})")

    if run is not None:
        run.summary["best_return"] = best_return
        run.finish()


if __name__ == "__main__":
    main()
