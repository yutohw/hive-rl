"""Optional Weights & Biases logging for the trainer.

Training never depends on W&B. If ``wandb`` is not installed, or ``config.use_wandb`` is
False, :func:`start_run` returns ``None`` and the trainer skips logging. W&B's own
``WANDB_MODE`` environment variable still applies: ``offline`` keeps runs on disk for a
later ``wandb sync``, ``disabled`` turns W&B off.

Runs start with machine details switched off, so a public W&B project shows the
metrics and hyperparameters but not the PC name, the Windows user folder, the git
remote or the source code.
"""

import asyncio
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from hive_rl import config

PUBLIC_HOST = "hive-rl-machine"

_CONFIG_KEYS = (
    "learning_rate",
    "learning_rate_min",
    "gamma",
    "gae_lambda",
    "clip_epsilon",
    "ppo_epochs",
    "entropy_coef_start",
    "entropy_coef_end",
    "value_coef",
    "max_grad_norm",
    "log_std_min",
    "log_std_max",
    "num_episodes",
    "max_steps",
    "num_workers",
    "max_transitions_per_episode",
    "state_dim",
    "action_dim",
    "base_seed",
)


def run_config() -> dict[str, Any]:
    """Hyperparameters and environment constants recorded with each run."""
    return {key: getattr(config, key) for key in _CONFIG_KEYS}


def start_run(name: str | None = None) -> Any | None:
    """Start a W&B run, or return ``None`` when W&B is switched off or not installed."""
    if not config.use_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed: training continues without W&B logging.")
        return None

    # hython installs Houdini's own asyncio event loop, which refuses to run outside the
    # main thread. W&B's background thread needs a normal loop, or it dies and
    # run.finish() hangs. Plain Python already uses this policy, so this changes nothing there.
    # Deprecated from Python 3.14, but Houdini 20 runs 3.10 and this is the only hook.
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())  # type: ignore

    settings = wandb.Settings(
        host=PUBLIC_HOST,  # shown instead of the real PC name
        x_disable_meta=True,  # no OS, user folder, executable or command line
        disable_git=True,  # no git remote, commit or diff
        disable_code=True,  # no source files
        save_code=False,
        x_save_requirements=False,
        console="off",  # terminal output stays local (tracebacks carry local paths)
    )
    return wandb.init(
        project=config.wandb_project,
        name=name,
        config=run_config(),
        dir=config.base_dir,  # local run files go next to comms/ and model/, not the repo
        settings=settings,
    )


def round_metrics(
    episodes: int,
    results: Sequence[tuple[int, float, int]],
    avg_100: float,
    lr: float,
    entropy_coef: float,
    log_std: float,
    seconds: float,
    losses: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Metrics for one training round.

    ``results`` holds one ``(track, episode_return, steps_driven)`` tuple per worker.
    """
    returns = [ret for _, ret, _ in results]
    steps = [n for _, _, n in results]
    metrics: dict[str, float] = {
        "episodes": episodes,
        "return/mean": fmean(returns),
        "return/avg100": float(avg_100),
        "steps/mean": fmean(steps),
        "steps/completion": fmean(steps) / config.max_steps,
        "schedule/lr": lr,
        "schedule/entropy_coef": entropy_coef,
        "policy/log_std": log_std,
        "time/round_seconds": seconds,
    }
    by_track: dict[int, list[float]] = {}
    for track, ret, _ in results:
        by_track.setdefault(track, []).append(ret)
    for track, rets in by_track.items():
        metrics[f"return/{config.track_names[track].lower()}"] = fmean(rets)
    if losses:
        metrics.update({f"loss/{key}": value for key, value in losses.items()})
    return metrics