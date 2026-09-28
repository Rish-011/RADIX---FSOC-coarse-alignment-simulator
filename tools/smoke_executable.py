"""Validate the actual Windows package, including GUI startup and clean exit."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def run(executable: Path, arguments: list[str]) -> None:
    process = subprocess.Popen([str(executable), *arguments])
    try:
        code = process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        # Only the process tree launched by this test is stopped.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True)
        raise RuntimeError("Packaged app did not exit within 60 seconds.") from None
    if code != 0:
        raise RuntimeError(f"Packaged application exited with code {code}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path, nargs="?", default=Path("dist/FSOCTracker.exe"))
    parser.add_argument("--output", type=Path, default=Path("output/package-smoke"))
    parser.add_argument("--video", type=Path, default=Path("examples/annotated_benchmark.mp4"))
    args = parser.parse_args()
    exe = args.executable.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run(exe, ["--headless", "--frames", "90", "--output", str(output / "headless")])
    if not list((output / "headless").glob("*.pdf")):
        raise RuntimeError("Packaged audit PDF was not generated.")
    screenshot = output / "dashboard.png"
    run(exe, ["--smoke-seconds", "4", "--screenshot", str(screenshot), "--output", str(output / "gui")])
    if not screenshot.is_file():
        raise RuntimeError("Packaged GUI did not produce its smoke screenshot.")
    video = args.video.resolve(strict=True)
    run(
        exe,
        [
            "--video",
            str(video),
            "--truth",
            str(video.with_suffix(".csv")),
            "--smoke-seconds",
            "8",
            "--output",
            str(output / "video-gui"),
        ],
    )
    logs = list((output / "video-gui").glob("*_live.csv"))
    if not logs:
        raise RuntimeError("Packaged video GUI did not produce telemetry.")
    with max(logs, key=lambda path: path.stat().st_mtime).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row["mode"] != "video" or float(row["pan_px_s"]) != 0 for row in rows):
        raise RuntimeError("Packaged video startup failed to bypass the simulation.")
    if not any(row["centroid_error_px"] for row in rows) or not all(row["detector_backend"] for row in rows):
        raise RuntimeError("Packaged video GUI did not record labeled centroids and detector provenance.")
    print(
        f"PASS: packaged headless audit, simulation GUI, labeled video GUI and clean exit. Screenshot: {screenshot}"
    )


if __name__ == "__main__":
    main()
