"""Turn the recorded inference runs into the README's results table.

Each inference cache folder holds one ``.bgeo.sc`` file per frame, with the car's state
stored as detail attributes (``speed``, ``collision``, ``return`` and more). Run with
hython, pointing at the folder that contains the five track folders::

    hython -m hive_rl.results "C:/path/to/inference cache"
"""

import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from hive_rl import config

FRAME_RE = re.compile(r"\.(\d+)\.bgeo\.sc$")
ATTRIBUTES = ("speed", "collision", "return")


@dataclass
class TrackResult:
    track: str
    steps: int
    collisions: int
    episode_return: float
    mean_speed: float
    slowest_speed: float  # lowest speed after the car first reaches 95% of its top speed


def summarise(track: str, frames: Sequence[dict[str, float]]) -> TrackResult:
    """Summarise one recorded run, given its frames in order."""
    speeds = [f["speed"] for f in frames]
    top = max(speeds)
    launch_end = next(i for i, s in enumerate(speeds) if s >= 0.95 * top)
    return TrackResult(
        track=track,
        steps=len(frames),
        collisions=sum(1 for f in frames if f["collision"]),
        episode_return=frames[-1]["return"],
        mean_speed=fmean(speeds),
        slowest_speed=min(speeds[launch_end:]),
    )


def markdown_table(results: Sequence[TrackResult]) -> str:
    lines = [
        "| Track | Steps driven | Collisions | Episode return | Mean speed "
        "| Slowest after launch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.track} | {r.steps:,} | {r.collisions} | {r.episode_return:,.1f} | "
            f"{r.mean_speed:.1f} | {r.slowest_speed:.1f} |"
        )
    return "\n".join(lines)


def find_track_folders(root: str) -> dict[str, str]:
    """Map each track name to the folder under ``root`` holding its frames."""
    found: dict[str, str] = {}
    for dirpath, _, files in os.walk(root):
        if not any(FRAME_RE.search(f) for f in files):
            continue
        rel = os.path.relpath(dirpath, root)
        for name in config.track_names:
            if re.search(rf"(?:^|[_\W]){name}(?:[_\W]|$)", rel, re.IGNORECASE):
                found[name] = dirpath
    return found


def load_frames(folder: str) -> list[dict[str, float]]:
    """Read the detail attributes of every frame in ``folder``. Needs hython."""
    import hou

    names = sorted(
        (f for f in os.listdir(folder) if FRAME_RE.search(f)),
        key=lambda f: int(FRAME_RE.search(f).group(1)),
    )
    frames = []
    for name in names:
        geo = hou.Geometry()
        geo.loadFromFile(os.path.join(folder, name))
        frames.append({a: float(geo.attribValue(a)) for a in ATTRIBUTES})
    return frames


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit('usage: hython -m hive_rl.results "<inference cache folder>"')
    folders = find_track_folders(args[0])
    results = [summarise(n, load_frames(folders[n])) for n in config.track_names if n in folders]
    print(markdown_table(results))


if __name__ == "__main__":
    main()
