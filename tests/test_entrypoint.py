from pathlib import Path

from fsoc_tracker import __main__ as entrypoint
from fsoc_tracker import main_dashboard
from fsoc_tracker.math3d_engine import Trajectory


def test_desktop_arguments_reach_mission_launch(tmp_path: Path, monkeypatch):
    received = {}

    def launch(*args, **kwargs):
        received.update(kwargs)
        return 0

    monkeypatch.setattr(main_dashboard, "launch", launch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_tracker.py",
            "--video",
            str(tmp_path / "source.mp4"),
            "--truth",
            str(tmp_path / "labels.csv"),
            "--speed",
            "40",
            "--seed",
            "9",
            "--trajectory",
            "Circular",
            "--output",
            str(tmp_path),
        ],
    )
    assert entrypoint.main() == 0
    assert received["video_path"] == tmp_path / "source.mp4"
    assert received["truth_path"] == tmp_path / "labels.csv"
    assert received["initial_config"].speed == 40
    assert received["initial_config"].seed == 9
    assert received["initial_config"].trajectory is Trajectory.CIRCULAR
