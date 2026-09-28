"""Reproducible measurement matrix; no pass metrics are hard-coded."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from fsoc_tracker.audit_exporter import AuditLogger
from fsoc_tracker.disturbances import DisturbanceConfig
from fsoc_tracker.math3d_engine import Trajectory
from fsoc_tracker.pipeline import RunConfig, TrackingPipeline, VideoSource


def main() -> None:
    output = Path("output/scenarios")
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    results = []
    configs = [(t.value, RunConfig(trajectory=t, decoys=2)) for t in Trajectory]
    configs.extend(
        [
            ("10% salt-pepper", RunConfig(disturbances=DisturbanceConfig(salt_pepper=0.1))),
            ("Maximum jitter", RunConfig(disturbances=DisturbanceConfig(jitter=20))),
            (
                "Dense fog",
                RunConfig(disturbances=DisturbanceConfig(fog=0.85, gaussian_sigma=8, salt_pepper=0.05)),
            ),
            ("Occlusion", RunConfig()),
        ]
    )
    for name, config in configs:
        pipe = TrackingPipeline(config)
        log = AuditLogger(output, asdict(config))
        rows = []
        for i in range(360):
            if name == "Occlusion":
                pipe.config = replace(config, occluded=150 <= i < 165)
            start = time.perf_counter()
            row = pipe.step().row
            row["wall_dt_s"] = time.perf_counter() - start
            log.append(row)
            rows.append(row)
        summary = log.summary()
        summary["scenario"] = name
        summary["last_pointing_error_px"] = rows[-1]["pointing_error_px"]
        summary["steady_pointing_rmse_px"] = float(
            np.sqrt(np.mean([r["pointing_error_px"] ** 2 for r in rows[90:]]))
        )
        results.append(summary)
        log.close()
        pipe.close()
    video = output / "annotated_benchmark.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("MP4 encoder unavailable")
    labels = ["frame,x,y"]
    for index in range(180):
        frame = np.full((720, 1280, 3), 5, np.uint8)
        x, y = 350 + 2 * index, round(360 + 60 * np.sin(index / 40))
        frame[y - 5 : y + 5, x - 5 : x + 5] = 250
        writer.write(frame)
        labels.append(f"{index},{x - 0.5},{y - 0.5}")
    writer.release()
    video.with_suffix(".csv").write_text("\n".join(labels), encoding="utf-8")
    source = VideoSource(video)
    pipe = TrackingPipeline(RunConfig(mode="video"), source)
    log = AuditLogger(
        output, {"mode": "video", "video": str(video), "ground_truth": str(video.with_suffix(".csv"))}
    )
    for _ in range(180):
        start = time.perf_counter()
        row = pipe.step().row
        row["wall_dt_s"] = time.perf_counter() - start
        log.append(row)
    summary = log.summary()
    summary["scenario"] = "Annotated 1280x720 video"
    results.append(summary)
    log.export()
    log.close()
    pipe.close()
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        print(
            result["scenario"],
            "retention",
            result["lock_retention_pct"],
            "centroid RMSE",
            result["centroid_rmse_px"],
            "FPS",
            result["average_fps"],
        )


if __name__ == "__main__":
    main()
