import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

from fsoc_tracker.main_dashboard import MissionControl


def test_dashboard_processes_frames_pause_resume_and_shutdown(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    window = MissionControl(output_directory=tmp_path)
    window.show()
    deadline = time.monotonic() + 8
    while window.last_packet is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert window.last_packet is not None
    window.pause_button.click()
    window.pause_button.click()
    window.close()
    deadline = time.monotonic() + 8
    while window.isVisible() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not window.isVisible()
    assert not window.worker.isRunning()


def test_weather_worker_survives_reset_without_duplicate_request(tmp_path: Path, monkeypatch):
    from threading import Event

    from fsoc_tracker import main_dashboard as dashboard

    release = Event()

    class OfflineWeather(dashboard.WeatherWorker):
        def run(self) -> None:
            release.wait(8)

    monkeypatch.setattr(dashboard, "WeatherWorker", OfflineWeather)
    app = QApplication.instance() or QApplication([])
    window = MissionControl(tmp_path)
    window.show()
    try:
        window.api_key.setText("test-only-no-network")
        window._fetch_weather()
        weather = window.weather_worker
        previous_mission = window.worker
        window.reset_simulation()
        deadline = time.monotonic() + 5
        while window.worker is previous_mission and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert window.worker is not previous_mission
        assert not window.weather_button.isEnabled()
        window._fetch_weather()
        assert window.weather_worker is weather
    finally:
        release.set()
        window.weather_worker.wait(2000)
        window.close()
        deadline = time.monotonic() + 5
        while window.isVisible() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
    assert not window.worker.isRunning()


@pytest.mark.parametrize("startup_video", [False, True])
def test_dashboard_switches_video_mode_and_preserves_eof_audit(tmp_path: Path, startup_video: bool):
    import cv2
    import numpy as np

    video = tmp_path / "benchmark.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    assert writer.isOpened()
    for _ in range(6):
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[235:245, 315:325] = 255
        writer.write(frame)
    writer.release()
    app = QApplication.instance() or QApplication([])
    truth = tmp_path / "explicit-labels.csv"
    truth.write_text("frame,x,y\n" + "".join(f"{i},319.5,239.5\n" for i in range(6)))
    window = MissionControl(
        output_directory=tmp_path / "audit",
        video_path=video if startup_video else None,
        truth_path=truth if startup_video else None,
    )
    window.show()
    if not startup_video:
        window.load_video(str(video))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        app.processEvents()
        if window.mode == "video" and not window.worker.isRunning() and window.logger is not None:
            break
        time.sleep(0.01)
    assert window.mode == "video"
    assert window.logger.summary()["frames"] == 6
    assert window.last_packet["result"].row["pan_px_s"] == 0
    if startup_video:
        assert window.last_packet["result"].row["centroid_error_px"] < 0.1
    else:
        assert window.last_packet["result"].row["centroid_error_px"] is None
    window._export()
    deadline = time.monotonic() + 10
    while window.export_worker.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert list((tmp_path / "audit").glob("*.pdf"))
    window.close()
    app.processEvents()
    assert not window.isVisible()
