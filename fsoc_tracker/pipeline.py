"""UI-independent pipeline; all image processing and video I/O run in its owner thread."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np

from .disturbances import DisturbanceConfig, DisturbanceInjector
from .math3d_engine import Frame, Trajectory, Vec, VirtualWorldEngine
from .perception_ekf import BeaconDetector, Detection, PriorityTracker
from .pid_fsm import GainScheduledPID, SpiralSearch, TrackingFSM, TrackingState


@dataclass(frozen=True)
class RunConfig:
    mode: Literal["simulation", "video"] = "simulation"
    trajectory: Trajectory = Trajectory.STRAIGHT
    speed: float = 25.0
    slew_deg_s: float = 10.0
    seed: int = 42
    decoys: int = 0
    occluded: bool = False
    disturbances: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    model_path: Path | None = None


@dataclass(frozen=True)
class VideoTransform:
    width: int
    height: int
    resized_width: int
    resized_height: int
    left: int
    top: int

    def to_source(self, point: Vec) -> Vec:
        # OpenCV resize uses pixel-center alignment, not corner alignment.
        return (point - [self.left, self.top] + 0.5) * [
            self.width / self.resized_width,
            self.height / self.resized_height,
        ] - 0.5

    def to_sensor(self, point: Vec) -> Vec:
        return (
            (point + 0.5) * [self.resized_width / self.width, self.resized_height / self.height]
            - 0.5
            + [self.left, self.top]
        )


class VideoSource:
    """Complete-frame letterboxing, no dropped frames; Benchmark-2 uses 30 Hz time."""

    def __init__(self, path: Path, truth_path: Path | None = None) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"Cannot open video: {path.name}")
        self.frame_index = 0
        self.reported_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.labels: dict[int, Vec] = {}
        self.transform: VideoTransform | None = None
        sidecar = truth_path if truth_path is not None else path.with_suffix(".csv")
        self.truth_path = sidecar if sidecar.is_file() else None
        try:
            if truth_path is not None and not truth_path.is_file():
                raise ValueError("Ground-truth file does not exist.")
            if sidecar.is_file():
                with sidecar.open(newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    if not {"frame", "x", "y"}.issubset(reader.fieldnames or []):
                        raise ValueError("Ground truth must have frame,x,y columns.")
                    for row in reader:
                        index = int(row["frame"])
                        point = np.array([float(row["x"]), float(row["y"])])
                        if index < 0 or index in self.labels or not np.isfinite(point).all():
                            raise ValueError("Ground truth contains invalid or duplicate frame labels.")
                        self.labels[index] = point
        except Exception:
            self.capture.release()
            raise

    def read(self) -> tuple[Frame, Vec | None, VideoTransform]:
        ok, frame = self.capture.read()
        if not ok:
            expected = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if expected > 0 and self.frame_index < expected - 1:
                raise OSError(f"Video decoding stopped at frame {self.frame_index} of {int(expected)}.")
            raise EOFError("Video completed.")
        height, width = frame.shape[:2]
        scale = min(640 / width, 480 / height)
        rw, rh = round(width * scale), round(height * scale)
        left, top = (640 - rw) // 2, (480 - rh) // 2
        self.transform = VideoTransform(width, height, rw, rh, left, top)
        sensor = np.zeros((480, 640), np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sensor[top : top + rh, left : left + rw] = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA)
        truth = self.labels.get(self.frame_index)
        if truth is not None and not (0 <= truth[0] < width and 0 <= truth[1] < height):
            raise ValueError(f"Ground truth outside source frame at frame {self.frame_index}.")
        self.frame_index += 1
        return sensor, truth, self.transform

    def close(self) -> None:
        self.capture.release()


@dataclass
class FrameResult:
    image: Frame
    binary: Frame
    row: dict[str, Any]
    backend: str
    detector_status: str


class TrackingPipeline:
    DT = 1 / 30

    def __init__(self, config: RunConfig, video: VideoSource | None = None) -> None:
        self.config = config
        self.video = video
        if config.mode == "video" and video is None:
            raise ValueError("Benchmark mode requires a video source.")
        self.world = VirtualWorldEngine(config.seed, config.decoys) if config.mode == "simulation" else None
        if self.world is not None:
            self.world.set_trajectory(config.trajectory)
        self.injector = DisturbanceInjector(config.seed + 1)
        self.detector = BeaconDetector(config.model_path)
        self.tracker = PriorityTracker()
        self.fsm = TrackingFSM()
        self.pid = GainScheduledPID()
        self.search = SpiralSearch()
        self.frame_index = 0
        self.last_origin = np.zeros(2)

    def designate(self, sensor_point: Vec) -> None:
        self.tracker.designate(sensor_point + self.last_origin)
        self.fsm = TrackingFSM()
        self.pid.reset()
        self.search = SpiralSearch()

    def step(self) -> FrameResult:
        started = time.perf_counter()
        dt = self.DT
        truth: Vec | None = None
        transform: VideoTransform | None = None
        origin = np.zeros(2)
        if self.world is not None:
            self.world.decoy_count = self.config.decoys
            self.world.step(dt, self.config.speed, self.config.occluded)
            origin = self.world.origin
            sensor, audit_shift = self.injector.apply(self.world.render_viewport(), self.config.disturbances)
            # Square core spans rounded center-5 through center+4: centroid at -0.5.
            truth = np.rint(self.world.target) - 0.5 - origin + audit_shift
        else:
            assert self.video is not None
            sensor, truth, transform = self.video.read()
        self.last_origin = origin
        # Use the configured jitter variance, never the sampled audit-only shift.
        # Uniform(-a,a) has variance a^2/3 on each axis.
        jitter = self.config.disturbances.jitter if self.world is not None else 0.0
        self.tracker.ekf.R = np.eye(2) * (4.0 + jitter**2 / 3.0)
        detections, binary = self.detector.detect(sensor)
        detection = self.tracker.step(detections, origin, dt)
        previous_state = self.fsm.state
        state = self.fsm.step(detection is not None, self.tracker.ekf.initialized, dt)
        velocity = np.zeros(2)
        if self.world is not None:
            limit = float(np.clip(self.config.slew_deg_s, 5, 10)) * 10
            if state == TrackingState.SEARCHING:
                if previous_state != state:
                    self.search = SpiralSearch()
                    self.pid.reset()
                center = self.tracker.ekf.position if self.tracker.ekf.initialized else self.world.camera
                command = self.search.command(self.world.camera, center, dt, limit)
            elif self.tracker.ekf.initialized:
                error = self.tracker.ekf.position - self.world.camera
                command = self.pid.compute(error, self.tracker.ekf.velocity, dt, limit)
            else:
                command = np.zeros(2)
            velocity = self.world.slew_camera(command, dt, self.config.slew_deg_s)
        measured = detection.center.copy() if detection is not None else None
        estimated = self.tracker.ekf.position - origin if self.tracker.ekf.initialized else None
        source_measurement = transform.to_source(measured) if measured is not None and transform else measured
        source_estimate = transform.to_source(estimated) if estimated is not None and transform else estimated
        centroid_error = (
            float(np.linalg.norm(source_measurement - truth))
            if source_measurement is not None and truth is not None
            else None
        )
        estimate_error = (
            float(np.linalg.norm(source_estimate - truth))
            if source_estimate is not None and truth is not None
            else None
        )
        # Pointing error is sensor-domain truth in simulation and observed offset in video.
        pointing = truth if self.world is not None else measured
        pointing_error = float(np.linalg.norm(pointing - [320, 240])) if pointing is not None else None
        image = cast(Frame, cv2.cvtColor(sensor, cv2.COLOR_GRAY2BGR))
        self._overlay(image, origin, detections, detection)
        row: dict[str, Any] = {
            "frame": self.frame_index,
            "time_s": (self.frame_index + 1) * dt,
            "dt_s": dt,
            "mode": self.config.mode,
            "detector_backend": self.detector.backend,
            "detector_status": self.detector.status,
            "state": state.value,
            "detected": detection is not None,
            "locked": self.fsm.locked and detection is not None,
            "centroid_x": float(source_measurement[0]) if source_measurement is not None else None,
            "centroid_y": float(source_measurement[1]) if source_measurement is not None else None,
            "truth_x": float(truth[0]) if truth is not None else None,
            "truth_y": float(truth[1]) if truth is not None else None,
            "centroid_error_px": centroid_error,
            "estimate_error_px": estimate_error,
            "pointing_error_px": pointing_error,
            "pan_px_s": float(velocity[0]),
            "tilt_px_s": float(velocity[1]),
            "acquisition_s": self.fsm.acquisition_s,
            "reacquisition_s": self.fsm.reacquisition_s,
            "processing_ms": (time.perf_counter() - started) * 1000,
            "wall_dt_s": dt,
        }
        self.frame_index += 1
        return FrameResult(image, binary, row, self.detector.backend, self.detector.status)

    def _overlay(
        self, image: Frame, origin: Vec, detections: list[Detection], selected: Detection | None
    ) -> None:
        cyan, amber, green = (220, 210, 40), (40, 185, 245), (135, 235, 85)
        cv2.circle(image, (320, 240), 10, cyan, 1, cv2.LINE_AA)
        for a, b in [
            ((296, 240), (314, 240)),
            ((326, 240), (344, 240)),
            ((320, 216), (320, 234)),
            ((320, 246), (320, 264)),
        ]:
            cv2.line(image, a, b, cyan, 1, cv2.LINE_AA)
        if len(self.tracker.history) > 1:
            points = np.clip(np.array(self.tracker.history) - origin, -10000, 10000).astype(np.int32)
            cv2.polylines(image, [points], False, amber, 1, cv2.LINE_AA)
        for d in detections:
            x, y, w, h = d.bbox
            color = green if d is selected else (100, 110, 120)
            for px, py, sx, sy in [
                (x - 4, y - 4, 1, 1),
                (x + w + 4, y - 4, -1, 1),
                (x - 4, y + h + 4, 1, -1),
                (x + w + 4, y + h + 4, -1, -1),
            ]:
                cv2.line(image, (px, py), (px + sx * 7, py), color, 1)
                cv2.line(image, (px, py), (px, py + sy * 7), color, 1)
        if self.tracker.ekf.initialized:
            prediction = np.clip(self.tracker.ekf.position - origin, -10000, 10000).astype(int)
            cv2.drawMarker(image, tuple(prediction), amber, cv2.MARKER_DIAMOND, 10, 1)
        cv2.putText(
            image,
            f"{self.fsm.state.value} / {self.detector.backend}",
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            cyan,
            1,
            cv2.LINE_AA,
        )

    def close(self) -> None:
        if self.video is not None:
            self.video.close()
