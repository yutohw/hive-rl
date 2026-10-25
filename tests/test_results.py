"""Results table: summarising a recorded run and finding the track folders."""

import pytest

from hive_rl.results import find_track_folders, markdown_table, summarise


def test_summarise_skips_the_launch_when_finding_the_slowest_speed() -> None:
    speeds = [10.0, 60.0, 100.0, 80.0, 99.0]  # top speed 100, reached on the third frame
    frames = [{"speed": s, "collision": 0.0, "return": float(i)} for i, s in enumerate(speeds)]
    frames[3]["collision"] = 1.0

    r = summarise("Spa", frames)

    assert r.steps == 5
    assert r.collisions == 1
    assert r.episode_return == pytest.approx(4.0)
    assert r.mean_speed == pytest.approx(69.8)
    assert r.slowest_speed == pytest.approx(80.0)  # not 10 or 60: those are the launch
    assert "| Spa | 5 | 1 | 4.0 | 69.8 | 80.0 |" in markdown_table([r])


def _frames_in(folder) -> None:
    folder.mkdir(parents=True)
    (folder / "rec.0002_v1.0002.bgeo.sc").write_bytes(b"")


def test_find_track_folders_matches_whole_names_only(tmp_path) -> None:
    root = tmp_path / "inference cache"
    spa = root / "260903_Inference_Spa_01" / "rec.0002" / "v2"
    monza = root / "260906_Inference_Monza_01" / "rec.0002" / "v1"
    _frames_in(spa)
    _frames_in(monza)
    (root / "notes").mkdir()
    assert find_track_folders(str(root)) == {"Spa": str(spa), "Monza": str(monza)}

    # "Spa" in a parent folder, or inside another word, is not the Spa track.
    root = tmp_path / "Spa_backup" / "inference cache"
    monza = root / "260906_Inference_Monza_01" / "v1"
    _frames_in(monza)
    _frames_in(root / "workspace_tests")
    assert find_track_folders(str(root)) == {"Monza": str(monza)}
