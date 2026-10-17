"""Environment step loop: ``worker.run_episode`` driven against a fake Houdini.

``hou`` only exists inside Houdini, so this test puts a small stand-in into
``sys.modules`` before importing the worker. The fake car earns a reward of 1.0
per frame and crashes (``stop = 1``) when the playhead reaches frame 6.
"""

import importlib
import sys
import types

import pytest
import torch

from hive_rl import config
from hive_rl.model import ActorCritic

CRASH_FRAME = 6


class FakeParm:
    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value

    def pressBuitton(self) -> None:
        pass


class FakeNode:
    def __init__(self, hou: "FakeHou", path: str) -> None:
        self.hou = hou
        self.path = path

    def parm(self, name: str) -> None:
        return self.hou.parms.setdefault((self.path, name), FakeParm())

    def cook(self, force: bool = False) -> None:
        pass

    def geometry(self) -> "FakeNode":
        return self

    def attribValue(self, name: str) -> object:
        return self.hou.attrib(name)


class FakeHou(types.ModuleType):
    """Just enough of the ``hou`` module for ``run_episode``."""

    def __init__(self) -> None:
        super().__init__("hou")
        self.current_frame = 1
        self.parms: dict[tuple[str, str], FakeParm] = {}

    def node(self, path: str) -> FakeNode:
        return FakeNode(self, path)

    def frame(self) -> int:
        return self.current_frame

    def setFrame(self, frame: int) -> None:
        self.current_frame = frame

    def attrib(self, name: str) -> object:
        values = {
            "state": [0.1 * self.current_frame] * config.state_dim,
            "reward": 1.0,
            "stop": int(self.current_frame >= CRASH_FRAME),
            "speed": 50.0,
        }
        return values[name]

    def parm_value(self, node: str, parm: str = "value1v1") -> float | None:
        return self.parms[(f"{config.geo_node}/{node}", parm)].value


@pytest.fixture
def fake_hou(monkeypatch: pytest.MonkeyPatch) -> FakeHou:
    hou = FakeHou()
    monkeypatch.setitem(sys.modules, "hou", hou)
    monkeypatch.delitem(sys.modules, "hive_rl.worker", raising=False)
    return hou


def test_env_step_writes_actions_advances_frames_and_stops_on_crash(fake_hou: FakeHou) -> None:
    worker = importlib.import_module("hive_rl.worker")
    torch.manual_seed(0)
    model = ActorCritic(config.state_dim, config.action_dim)

    states, actions, pre_tanhs, rewards, values, logps, episode_return, last_step = (
        worker.run_episode(model)
    )

    assert fake_hou.frame() == CRASH_FRAME
    assert last_step == CRASH_FRAME - 2
    assert len(rewards) == CRASH_FRAME - 1
    assert all(len(x) == len(rewards) for x in (states, actions, pre_tanhs, values, logps))
    assert episode_return == pytest.approx(sum(rewards))
    assert episode_return == pytest.approx(5.0)

    assert states[0] == pytest.approx([0.1] * config.state_dim)

    assert -1.0 <= fake_hou.parm_value("Steer") <= 1.0
    assert 0.0 <= fake_hou.parm_value("Throttle") <= 1.0
    assert 0.0 <= fake_hou.parm_value("Brake") <= 1.0
