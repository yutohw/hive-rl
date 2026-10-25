"""W&B logging: privacy settings on the run, the no-W&B fallback, and per-round metrics.

A fake ``wandb`` module stands in for the real one, so these tests need no account,
no network and no wandb install.
"""

import asyncio
import sys
import types

import pytest

from hive_rl import config, tracking


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[int | None, dict]] = []
        self.summary: dict = {}
        self.finished = False

    def log(self, data: dict, step: int | None = None) -> None:
        self.logged.append((step, data))

    def finish(self) -> None:
        self.finished = True


def make_fake_wandb() -> types.ModuleType:
    wandb = types.ModuleType("wandb")
    wandb.calls = {}

    def settings(**kwargs: object) -> dict:
        wandb.calls["settings"] = kwargs
        return kwargs

    def init(**kwargs: object) -> FakeRun:
        wandb.calls["init"] = kwargs
        wandb.run = FakeRun()
        return wandb.run

    wandb.Settings = settings
    wandb.init = init
    return wandb


def test_start_run_hides_machine_details(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = make_fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setattr(config, "use_wandb", True)

    run = tracking.start_run()

    assert run is fake.run
    settings = fake.calls["settings"]
    assert settings["host"] == tracking.PUBLIC_HOST
    assert settings["x_disable_meta"] is True
    assert settings["disable_git"] is True
    assert settings["disable_code"] is True
    assert settings["console"] == "off"
    init = fake.calls["init"]
    assert init["project"] == config.wandb_project
    assert init["config"]["learning_rate"] == config.learning_rate
    assert init["config"]["num_workers"] == config.num_workers


def test_start_run_resets_houdinis_event_loop_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    class HoudiniLikePolicy(asyncio.DefaultEventLoopPolicy):
        pass

    original = asyncio.get_event_loop_policy()
    monkeypatch.setitem(sys.modules, "wandb", make_fake_wandb())
    monkeypatch.setattr(config, "use_wandb", True)
    asyncio.set_event_loop_policy(HoudiniLikePolicy())
    try:
        tracking.start_run()
        assert type(asyncio.get_event_loop_policy()) is asyncio.DefaultEventLoopPolicy
    finally:
        asyncio.set_event_loop_policy(original)


def test_start_run_returns_none_without_wandb(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)  # makes `import wandb` fail
    monkeypatch.setattr(config, "use_wandb", True)
    assert tracking.start_run() is None
    assert "without W&B" in capsys.readouterr().out

    monkeypatch.setitem(sys.modules, "wandb", make_fake_wandb())
    monkeypatch.setattr(config, "use_wandb", False)
    assert tracking.start_run() is None


def test_round_metrics_summarise_one_round() -> None:
    # Three workers: one on Monza (track 0), two on Spa (track 1).
    results = [(0, 100.0, 1440), (1, 50.0, 720), (1, 70.0, 360)]
    losses = {"policy": 0.1, "value": 2.0}

    m = tracking.round_metrics(
        episodes=15,
        results=results,
        avg_100=80.0,
        lr=5e-4,
        entropy_coef=0.05,
        log_std=-0.5,
        seconds=12.5,
        losses=losses,
    )

    assert m["episodes"] == 15
    assert m["return/mean"] == pytest.approx(220.0 / 3)
    assert m["return/monza"] == pytest.approx(100.0)
    assert m["return/spa"] == pytest.approx(60.0)
    assert "return/barcelona" not in m
    assert m["steps/completion"] == pytest.approx(840.0 / config.max_steps)
    assert m["loss/policy"] == pytest.approx(0.1)
    assert m["loss/value"] == pytest.approx(2.0)
