# Complete FSOC Tracker source

## Directory structure

```text
.gitignore
docs/design.md
docs/superpowers/plans/2026-09-19-fsoc.md
docs/USER_MANUAL.md
docs/VALIDATION.md
examples/annotated_benchmark.csv
examples/annotated_benchmark.mp4
examples/README.md
fsoc_tracker/__init__.py
fsoc_tracker/__main__.py
fsoc_tracker/audit_exporter.py
fsoc_tracker/disturbances.py
fsoc_tracker/main_dashboard.py
fsoc_tracker/math3d_engine.py
fsoc_tracker/perception_ekf.py
fsoc_tracker/pid_fsm.py
fsoc_tracker/pipeline.py
fsoc_tracker/scene_view.py
fsoc_tracker.spec
models/README.md
pyproject.toml
README.md
requirements-dev.txt
requirements-lock-windows-py313.txt
requirements.txt
run_tracker.py
Start-FSOC.ps1
tests/test_core.py
tests/test_dashboard.py
tests/test_entrypoint.py
tests/test_onnx.py
tools/package_source.py
tools/smoke_executable.py
tools/validate_scenarios.py
```

## fsoc_tracker/math3d_engine.py

```python
"""Historical module name; the tracking physics is strictly a 2D sliding window.

Units: position in world pixels, velocity in pixels/second, time in seconds.
The user-requested actuator scale is 10 pixels/degree (not an optical FOV model).
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
from numpy.typing import NDArray

Vec = NDArray[np.float64]
Frame = NDArray[np.uint8]


class Trajectory(str, Enum):
    STRAIGHT = "Straight Line"
    CIRCULAR = "Circular"
    FIGURE_EIGHT = "Figure of 8"
    RANDOM = "Random Walk"
    CUSTOM = "Custom B-Spline"


def cubic_bspline(points: Vec, samples_per_span: int = 40) -> Vec:
    """Clamped, open uniform cubic B-spline evaluated by the de Boor algorithm."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("A B-spline requires at least four finite 2D control points.")
    n, degree = len(points), 3
    knots = np.r_[np.zeros(4), np.arange(1, n - 3) / (n - 3), np.ones(4)]
    result: list[Vec] = []
    for u in np.linspace(0, 1, (n - 3) * samples_per_span + 1):
        k = min(n - 1, max(degree, int(np.searchsorted(knots, u, side="right") - 1)))
        d = points[k - degree : k + 1].copy()
        for r in range(1, degree + 1):
            for j in range(degree, r - 1, -1):
                idx = j + k - degree
                denom = knots[j + 1 + k - r] - knots[idx]
                alpha = (u - knots[idx]) / denom if denom else 0.0
                d[j] = (1 - alpha) * d[j - 1] + alpha * d[j]
        result.append(d[degree])
    return np.asarray(result)


class VirtualWorldEngine:
    WORLD_SIZE = 2000
    WIDTH = 640
    HEIGHT = 480
    PIXELS_PER_DEGREE = 10.0

    def __init__(self, seed: int = 42, decoys: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.camera = np.array([1000.0, 1000.0])
        self.target = np.array([1080.0, 1040.0])
        self.target_velocity = np.zeros(2)
        self.camera_velocity = np.zeros(2)
        self.decoy_count = int(np.clip(decoys, 0, 5))
        self.decoy_positions: list[Vec] = []
        self.time = 0.0
        self.distance = 0.0
        self.direction = np.array([1.0, 0.45]) / np.hypot(1, 0.45)
        self.world_frame = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE), np.uint8)
        self.custom_points = np.array([[850, 900], [1100, 750], [1250, 1150], [900, 1250]], float)
        self.trajectory = Trajectory.STRAIGHT
        self.path = np.zeros((2, 2))
        self.arc = np.array([0.0, 1.0])
        self.set_trajectory(self.trajectory)

    @property
    def origin(self) -> Vec:
        """Integer slice origin, also used for exact sensor/world conversion."""
        return np.rint(self.camera).astype(np.float64) - [320.0, 240.0]

    def set_trajectory(self, trajectory: Trajectory, points: Vec | None = None) -> None:
        self.trajectory = Trajectory(trajectory)
        self.distance = 0.0
        if points is not None:
            self.custom_points = np.clip(np.asarray(points, dtype=float), 20, 1980)
        angle = np.linspace(0, 2 * np.pi, 1601)
        if trajectory == Trajectory.CIRCULAR:
            self.path = np.column_stack((1000 + 180 * np.cos(angle), 1000 + 180 * np.sin(angle)))
        elif trajectory == Trajectory.FIGURE_EIGHT:
            self.path = np.column_stack((1000 + 250 * np.sin(angle), 1000 + 140 * np.sin(2 * angle)))
        elif trajectory == Trajectory.CUSTOM:
            forward = cubic_bspline(self.custom_points)
            self.path = np.vstack((forward, forward[-2::-1]))
        else:
            return
        self.arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
        if self.arc[-1] < 1:
            raise ValueError("Draw a path with at least one pixel of length.")
        self.target = self.path[0].copy()

    def step(self, dt: float, speed: float, occluded: bool = False) -> None:
        if not 0 < dt <= 1 or not np.isfinite(speed) or speed < 0:
            raise ValueError("dt must be in (0,1] seconds and speed nonnegative.")
        self.time += dt
        before = self.target.copy()
        if self.trajectory in (Trajectory.STRAIGHT, Trajectory.RANDOM):
            if self.trajectory == Trajectory.RANDOM:
                angle = np.arctan2(self.direction[1], self.direction[0]) + self.rng.normal(
                    0, 0.7 * np.sqrt(dt)
                )
                self.direction = np.array([np.cos(angle), np.sin(angle)])
            candidate = self.target + self.direction * speed * dt
            for axis in range(2):
                if candidate[axis] < 20 or candidate[axis] > 1980:
                    self.direction[axis] *= -1
            self.target = np.clip(self.target + self.direction * speed * dt, 20, 1980)
        else:
            self.distance = (self.distance + speed * dt) % self.arc[-1]
            self.target = np.array([np.interp(self.distance, self.arc, self.path[:, i]) for i in range(2)])
        self.target_velocity = (self.target - before) / dt
        self.decoy_positions = [
            np.array(
                [1000 + 280 * np.cos(self.time * 0.18 + i * 2), 1000 + 170 * np.sin(self.time * 0.31 + i * 2)]
            )
            for i in range(self.decoy_count)
        ]
        self.world_frame.fill(6)
        if not occluded:
            self._draw_spot(self.target, 255)
        for position in self.decoy_positions:
            self._draw_spot(position, 195)

    def _draw_spot(self, position: Vec, intensity: int) -> None:
        x, y = np.rint(position).astype(int)
        cv2.circle(self.world_frame, (x, y), 9, int(intensity * 0.13), -1)
        self.world_frame[y - 5 : y + 5, x - 5 : x + 5] = intensity

    def slew_camera(self, command: Vec, dt: float, limit_deg_s: float = 5.0) -> Vec:
        if dt <= 0 or not np.isfinite(command).all():
            raise ValueError("Camera command and timestep must be finite, with dt > 0.")
        limit = float(np.clip(limit_deg_s, 5.0, 10.0)) * self.PIXELS_PER_DEGREE
        velocity = np.clip(command, -limit, limit)
        previous = self.camera.copy()
        self.camera = np.clip(self.camera + velocity * dt, [320.0, 240.0], [1680.0, 1760.0])
        self.camera_velocity = (self.camera - previous) / dt
        return self.camera_velocity.copy()

    def render_viewport(self) -> Frame:
        cx, cy = np.rint(self.camera).astype(int)
        return self.world_frame[cy - 240 : cy + 240, cx - 320 : cx + 320].copy()
```

## fsoc_tracker/disturbances.py

```python
"""Seeded sensor-domain disturbances; ground-truth shift is returned only for audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from .math3d_engine import Frame, Vec


@dataclass(frozen=True)
class DisturbanceConfig:
    jitter: float = 0.0
    fog: float = 0.0
    salt_pepper: float = 0.0
    gaussian_sigma: float = 0.0
    poisson: bool = False
    scintillation: float = 0.0

    def __post_init__(self) -> None:
        for name, maximum in [
            ("jitter", 20.0),
            ("fog", 1.0),
            ("salt_pepper", 0.1),
            ("gaussian_sigma", 20.0),
            ("scintillation", 1.0),
        ]:
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be in [0, {maximum}].")


class DisturbanceInjector:
    def __init__(self, seed: int = 43) -> None:
        self.rng = np.random.default_rng(seed)

    def apply(self, frame: Frame, config: DisturbanceConfig) -> tuple[Frame, Vec]:
        out = frame.copy()
        shift = np.zeros(2)
        if config.jitter:
            shift = self.rng.uniform(-config.jitter, config.jitter, 2)
            matrix = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], np.float32)
            out = cast(
                Frame,
                cv2.warpAffine(out, matrix, (out.shape[1], out.shape[0]), borderMode=cv2.BORDER_REFLECT_101),
            )
        if config.fog:
            f = config.fog
            out = cast(Frame, cv2.addWeighted(out, 1 - 0.82 * f, np.full_like(out, 110), 0.7 * f, 0))
            out = cast(Frame, cv2.GaussianBlur(out, (0, 0), 0.3 + 3.0 * f))
        if config.scintillation:
            scale = np.exp(self.rng.normal(-0.5 * config.scintillation**2, config.scintillation))
            out = np.clip(out.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        if config.poisson:
            out = np.clip(self.rng.poisson(out.astype(np.float32)), 0, 255).astype(np.uint8)
        if config.gaussian_sigma:
            out = np.clip(
                out.astype(np.float32) + self.rng.normal(0, config.gaussian_sigma, out.shape), 0, 255
            ).astype(np.uint8)
        if config.salt_pepper:
            count = int(out.size * config.salt_pepper) // 2
            indices = self.rng.choice(out.size, 2 * count, replace=False)
            flat = out.reshape(-1)
            flat[indices[:count]] = 0
            flat[indices[count:]] = 255
        return out, shift
```

## fsoc_tracker/perception_ekf.py

```python
"""Beacon detection, gated temporal association and nonlinear coordinated-turn EKF."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .math3d_engine import Frame, Vec

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    center: Vec
    bbox: tuple[int, int, int, int]
    area: float
    brightness: float
    confidence: float = 1.0


class BeaconDetector:
    """YOLOv8 raw [1,4+classes,N] float32 RGB export, with classical fallback.

    Use beacon-trained weights, exported with nms=False. Generic COCO weights are
    not beacon detectors. Absent/invalid weights use intensity-weighted components.
    """

    def __init__(self, model_path: Path | None = None, confidence: float = 0.25) -> None:
        self.session: Any = None
        self.confidence = confidence
        self.backend = "OpenCV moments"
        self.status = "No beacon ONNX weights; classical fallback active."
        self.input_name = ""
        self.input_size = (640, 640)
        if model_path is not None and model_path.is_file():
            try:
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                self.session = ort.InferenceSession(
                    str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
                )
                node = self.session.get_inputs()[0]
                if node.type != "tensor(float)" or len(node.shape) != 4:
                    raise ValueError("Expected NCHW float32 YOLOv8 input.")
                self.input_name = node.name
                self.input_size = (
                    node.shape[3] if isinstance(node.shape[3], int) else 640,
                    node.shape[2] if isinstance(node.shape[2], int) else 640,
                )
                self.backend = "YOLOv8 ONNX"
                self.status = f"Loaded {model_path.name}"
            except Exception as exc:
                self.session = None
                self.status = f"ONNX unavailable ({type(exc).__name__}); classical fallback."
                LOG.warning(self.status, exc_info=True)

    @staticmethod
    def classical(frame: Frame) -> tuple[list[Detection], Frame]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        clean = cv2.medianBlur(gray, 3)
        otsu, _ = cv2.threshold(clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        background = float(np.median(clean))
        mad = float(np.median(np.abs(clean.astype(np.float32) - background)))
        threshold = max(float(otsu), background + max(8.0, 5 * mad))
        binary = (clean > threshold).astype(np.uint8) * 255
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        detections: list[Detection] = []
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if not 9 <= area <= 2500 or w > 90 or h > 90:
                continue
            region = clean[y : y + h, x : x + w].astype(np.float32)
            weights = np.maximum(region - background, 0) * (labels[y : y + h, x : x + w] == label)
            moments = cv2.moments(weights)
            if moments["m00"] <= 0:
                continue
            center = np.array([x + moments["m10"] / moments["m00"], y + moments["m01"] / moments["m00"]])
            detections.append(Detection(center, (x, y, w, h), float(area), float(region.max())))
        return detections, binary

    def detect(self, frame: Frame) -> tuple[list[Detection], Frame]:
        classical, binary = self.classical(frame)
        if self.session is None:
            return classical, binary
        try:
            return self._onnx(frame), binary
        except Exception as exc:
            LOG.exception("ONNX inference failed; using classical detector")
            self.session = None
            self.backend = "OpenCV moments"
            self.status = f"ONNX inference failure ({type(exc).__name__}); classical fallback."
            return classical, binary

    def _onnx(self, frame: Frame) -> list[Detection]:
        height, width = frame.shape[:2]
        iw, ih = self.input_size
        scale = min(iw / width, ih / height)
        rw, rh = round(width * scale), round(height * scale)
        ox, oy = (iw - rw) // 2, (ih - rh) // 2
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
        canvas = np.full((ih, iw, 3), 114, np.uint8)
        canvas[oy : oy + rh, ox : ox + rw] = cv2.resize(bgr, (rw, rh))
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        raw = np.asarray(self.session.run(None, {self.input_name: tensor})[0])
        if raw.ndim != 3 or raw.shape[0] != 1:
            raise ValueError("Expected YOLOv8 raw output [1,4+classes,N].")
        rows = raw[0]
        if rows.shape[0] < rows.shape[1]:
            rows = rows.T
        if rows.shape[1] < 5:
            raise ValueError("YOLOv8 output has no class scores.")
        scores = np.max(rows[:, 4:], axis=1)
        boxes: list[list[int]] = []
        confidences: list[float] = []
        for row, score in zip(rows, scores):
            if score < self.confidence or not np.isfinite(row).all():
                continue
            cx, cy, w, h = row[:4]
            x0 = int(np.clip((cx - w / 2 - ox) / scale, 0, width - 1))
            y0 = int(np.clip((cy - h / 2 - oy) / scale, 0, height - 1))
            x1 = int(np.clip((cx + w / 2 - ox) / scale, x0 + 1, width))
            y1 = int(np.clip((cy + h / 2 - oy) / scale, y0 + 1, height))
            boxes.append([x0, y0, x1 - x0, y1 - y0])
            confidences.append(float(score))
        indices = np.asarray(cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, 0.45)).reshape(-1)
        result: list[Detection] = []
        for idx in indices:
            x, y, w, h = boxes[int(idx)]
            crop = frame[y : y + h, x : x + w]
            spots, _ = self.classical(crop)
            if spots:
                spot = max(spots, key=lambda d: d.area)
                center = spot.center + [x, y]
                area, brightness = spot.area, spot.brightness
            else:
                center = np.array([x + w / 2, y + h / 2])
                area, brightness = float(w * h), float(crop.max())
            result.append(Detection(center, (x, y, w, h), area, brightness, confidences[int(idx)]))
        return result


class TargetTrackerEKF:
    """State [x,y,vx,vy,omega], nonlinear constant-turn-rate transition.

    Stable sinc expressions handle omega=0 continuously. Central-difference
    Jacobian and Joseph covariance update preserve numerical conditioning.
    """

    def __init__(self) -> None:
        self.state = np.zeros(5, dtype=np.float64)
        self.P = np.diag([25.0, 25.0, 2500.0, 2500.0, 0.25])
        self.R = np.eye(2) * 4.0
        self.H = np.eye(2, 5)
        self.initialized = False

    @property
    def position(self) -> Vec:
        return self.state[:2].copy()

    @property
    def velocity(self) -> Vec:
        return self.state[2:4].copy()

    def initialize(self, position: Vec) -> None:
        TargetTrackerEKF.__init__(self)
        self.state[:2] = position
        self.initialized = True

    @staticmethod
    def transition(state: Vec, dt: float) -> Vec:
        x, y, vx, vy, omega = state
        theta = omega * dt
        a = dt * np.sinc(theta / np.pi)
        b = dt * 0.5 * theta * np.sinc(theta / (2 * np.pi)) ** 2
        c, s = np.cos(theta), np.sin(theta)
        return np.array([x + a * vx - b * vy, y + b * vx + a * vy, c * vx - s * vy, s * vx + c * vy, omega])

    def predict(self, dt: float) -> Vec:
        if not 0 < dt <= 1:
            raise ValueError("EKF dt must be in (0,1].")
        if not self.initialized:
            return self.position
        jacobian = np.empty((5, 5))
        for i in range(5):
            eps = 1e-5 * max(1.0, abs(self.state[i]))
            delta = np.zeros(5)
            delta[i] = eps
            jacobian[:, i] = (
                self.transition(self.state + delta, dt) - self.transition(self.state - delta, dt)
            ) / (2 * eps)
        self.state = self.transition(self.state, dt)
        q = np.zeros((5, 5))
        for position, velocity in [(0, 2), (1, 3)]:
            q[np.ix_([position, velocity], [position, velocity])] = 100 * np.array(
                [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]
            )
        q[4, 4] = 0.025 * dt
        self.P = jacobian @ self.P @ jacobian.T + q
        self.P = (self.P + self.P.T) * 0.5
        return self.position

    def innovation_distance(self, position: Vec) -> float:
        residual = position - self.position
        covariance = self.P[:2, :2] + self.R
        return float(residual @ np.linalg.solve(covariance, residual))

    def update(self, position: Vec) -> None:
        if not self.initialized:
            self.initialize(position)
            return
        residual = position - self.H @ self.state
        innovation = self.H @ self.P @ self.H.T + self.R
        gain = np.linalg.solve(innovation, self.H @ self.P).T
        self.state += gain @ residual
        self.state[4] = np.clip(self.state[4], -3.0, 3.0)
        correction = np.eye(5) - gain @ self.H
        self.P = correction @ self.P @ correction.T + gain @ self.R @ gain.T
        self.P = (self.P + self.P.T) * 0.5


class PriorityTracker:
    def __init__(self) -> None:
        self.ekf = TargetTrackerEKF()
        self.history: deque[Vec] = deque(maxlen=90)
        self.seed: Vec | None = None
        self.area: float | None = None
        self.brightness: float | None = None
        self.missing_s = 0.0

    def designate(self, position: Vec) -> None:
        PriorityTracker.__init__(self)
        self.seed = np.asarray(position, dtype=float)

    def step(self, detections: list[Detection], origin: Vec, dt: float) -> Detection | None:
        self.ekf.predict(dt)
        selected: Detection | None = None
        if not self.ekf.initialized:
            cue = self.seed if self.seed is not None else origin + [320.0, 240.0]
            if detections:
                candidate = min(detections, key=lambda d: float(np.linalg.norm(d.center + origin - cue)))
                if self.seed is None or np.linalg.norm(candidate.center + origin - cue) < 45:
                    selected = candidate
        else:
            candidates: list[tuple[float, Detection]] = []
            for detection in detections:
                distance = self.ekf.innovation_distance(detection.center + origin)
                pixel_distance = np.linalg.norm(detection.center + origin - self.ekf.position)
                ratio = detection.area / max(self.area or detection.area, 1.0)
                # Merged blobs and out-of-gate candidates cause coasting, not identity switches.
                if (
                    distance > 25
                    or pixel_distance > min(160.0, 35 + self.missing_s * 80)
                    or not 0.5 < ratio < 1.65
                ):
                    continue
                appearance = abs(detection.brightness - (self.brightness or detection.brightness)) / 35
                candidates.append((distance + appearance**2, detection))
            candidates.sort(key=lambda pair: pair[0])
            if candidates and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] > 2.0):
                selected = candidates[0][1]
        if selected is not None:
            self.ekf.update(selected.center + origin)
            self.area = selected.area if self.area is None else 0.95 * self.area + 0.05 * selected.area
            self.brightness = (
                selected.brightness
                if self.brightness is None
                else 0.95 * self.brightness + 0.05 * selected.brightness
            )
            self.missing_s = 0.0
        else:
            self.missing_s += dt
        if self.ekf.initialized:
            self.history.append(self.ekf.position)
        return selected
```

## fsoc_tracker/pid_fsm.py

```python
"""Gain-scheduled dual-axis PID, seven-state PAT machine and spiral search."""

from __future__ import annotations

from enum import Enum

import numpy as np

from .math3d_engine import Vec


class TrackingState(str, Enum):
    SEARCHING = "SEARCHING"
    ACQUIRING = "ACQUIRING"
    LOCKED = "LOCKED"
    TRACKING = "TRACKING"
    TARGET_LOST = "TARGET_LOST"
    PREDICTING = "PREDICTING"
    REACQUIRING = "REACQUIRING"


class TrackingFSM:
    def __init__(self, confirmations: int = 3, coast_s: float = 2.0) -> None:
        self.state = TrackingState.SEARCHING
        self.confirmations = confirmations
        self.coast_s = coast_s
        self.hits = 0
        self.lost_s = 0.0
        self.elapsed = 0.0
        self.acquisition_s: float | None = None
        self.loss_started: float | None = None
        self.reacquisition_s: float | None = None

    @property
    def locked(self) -> bool:
        return self.state in (TrackingState.LOCKED, TrackingState.TRACKING)

    def step(self, detected: bool, initialized: bool, dt: float) -> TrackingState:
        self.elapsed += dt
        self.hits = self.hits + 1 if detected else 0
        self.lost_s = 0.0 if detected else self.lost_s + dt
        previous = self.state
        if detected:
            if previous == TrackingState.SEARCHING:
                self.state = (
                    TrackingState.ACQUIRING if self.acquisition_s is None else TrackingState.REACQUIRING
                )
            elif previous in (TrackingState.TARGET_LOST, TrackingState.PREDICTING):
                self.state = TrackingState.REACQUIRING
            elif previous in (TrackingState.ACQUIRING, TrackingState.REACQUIRING):
                if self.hits >= self.confirmations:
                    self.state = TrackingState.LOCKED
                    if self.acquisition_s is None:
                        self.acquisition_s = self.elapsed
                    if self.loss_started is not None:
                        self.reacquisition_s = self.elapsed - self.loss_started
                        self.loss_started = None
            elif previous == TrackingState.LOCKED:
                self.state = TrackingState.TRACKING
        elif previous in (TrackingState.LOCKED, TrackingState.TRACKING, TrackingState.REACQUIRING):
            self.state = TrackingState.TARGET_LOST
            if self.loss_started is None:
                self.loss_started = self.elapsed - dt
        elif previous == TrackingState.TARGET_LOST:
            self.state = TrackingState.PREDICTING if initialized else TrackingState.SEARCHING
        elif (
            previous == TrackingState.PREDICTING
            and self.lost_s > self.coast_s
            or previous == TrackingState.ACQUIRING
        ):
            self.state = TrackingState.SEARCHING
        return self.state


class GainScheduledPID:
    def __init__(self) -> None:
        self.integral = np.zeros(2)
        self.previous: Vec | None = None
        self.derivative = np.zeros(2)

    def reset(self) -> None:
        GainScheduledPID.__init__(self)

    def compute(self, error: Vec, feedforward: Vec, dt: float, limit: float) -> Vec:
        if dt <= 0 or limit <= 0:
            raise ValueError("PID dt and limit must be positive.")
        kp = 0.85 + 2.15 * np.clip(np.abs(error) / 100.0, 0.0, 1.0)
        derivative = np.zeros(2) if self.previous is None else (error - self.previous) / dt
        alpha = dt / (0.15 + dt)
        self.derivative += alpha * (derivative - self.derivative)
        candidate = np.clip(self.integral + error * dt, -80.0, 80.0)
        raw = feedforward + kp * error + 0.08 * candidate + 0.04 * self.derivative
        command = np.clip(raw, -limit, limit)
        # Conditional integration prevents windup while saturated in the error direction.
        integrate = (raw == command) | (np.sign(error) != np.sign(raw))
        self.integral = np.where(integrate, candidate, self.integral)
        self.previous = error.copy()
        return command


class SpiralSearch:
    """Archimedean spiral with approximately bounded tangential waypoint speed."""

    def __init__(self) -> None:
        self.theta = 0.0
        self.center: Vec | None = None

    def command(self, camera: Vec, center: Vec, dt: float, limit: float) -> Vec:
        if self.center is None:
            self.center = np.clip(center.copy(), [320.0, 240.0], [1680.0, 1760.0])
        pitch = 90.0
        self.theta += limit * 0.8 * dt / (pitch * np.sqrt(1 + self.theta**2))
        radius = pitch * self.theta
        waypoint = self.center + radius * np.array([np.cos(self.theta), np.sin(self.theta)])
        waypoint = np.clip(waypoint, [320.0, 240.0], [1680.0, 1760.0])
        if radius > 2300:
            self.theta = 0.0
        return np.clip(2.0 * (waypoint - camera), -limit, limit)
```

## fsoc_tracker/main_dashboard.py

```python
"""PyQt6 mission control. Widgets stay on the GUI thread; workers own their data."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from typing import Any

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QCloseEvent,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .audit_exporter import AuditLogger
from .disturbances import DisturbanceConfig
from .math3d_engine import Frame, Trajectory, cubic_bspline
from .pipeline import FrameResult, RunConfig, TrackingPipeline, VideoSource
from .scene_view import render_scene

LOG = logging.getLogger(__name__)
STYLE = """
QMainWindow, QDialog { background: #090e13; }
QWidget { color: #c9d6df; font-family: 'Segoe UI'; font-size: 12px; }
QGroupBox { background: #101820; border: 1px solid #24303c; border-radius: 6px;
 margin-top: 16px; padding: 12px 8px 8px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; color: #758b9b; }
QPushButton { background: #182733; border: 1px solid #304653; border-radius: 4px; padding: 7px 12px; }
QPushButton:hover { border-color: #59d5e3; background: #203742; }
QPushButton:disabled { color: #536471; border-color: #26323c; }
QPushButton#primary { background: #173e45; color: #8eeaf0; border-color: #327c88; font-weight: 600; }
QLineEdit, QComboBox, QSpinBox { background: #0b1219; border: 1px solid #30414d; padding: 5px; border-radius: 3px; }
QComboBox QAbstractItemView { background: #14212b; color: #c9d6df; selection-background-color: #245568; }
QSlider::groove:horizontal { height: 4px; background: #273642; border-radius: 2px; }
QSlider::handle:horizontal { background: #58cad7; width: 12px; margin: -5px 0; border-radius: 6px; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #47606d; background: #111b24; }
QCheckBox::indicator:checked { background: #56cbd7; }
QScrollArea { border: none; background: transparent; }
QStatusBar { background: #101820; color: #8ca0ae; }
"""


class MissionWorker(QThread):
    packet_ready = pyqtSignal(object)
    status = pyqtSignal(str)
    failed = pyqtSignal(str)
    audit_ready = pyqtSignal(object)

    def __init__(
        self,
        config: RunConfig,
        output: Path,
        video_path: Path | None = None,
        custom_points: np.ndarray | None = None,
        truth_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.output = output
        self.video_path = video_path
        self.custom_points = custom_points
        self.truth_path = truth_path
        self.commands: Queue[tuple[str, Any]] = Queue()
        self.display_pending = Event()
        self.wake = Event()
        self.logger: AuditLogger | None = None

    def submit(self, name: str, value: Any = None) -> None:
        self.commands.put((name, value))
        self.wake.set()

    def stop(self) -> None:
        self.requestInterruption()
        self.wake.set()

    def run(self) -> None:
        pipeline: TrackingPipeline | None = None
        source: VideoSource | None = None
        paused = False
        try:
            cv2.setNumThreads(2)
            source = VideoSource(self.video_path, self.truth_path) if self.video_path is not None else None
            pipeline = TrackingPipeline(self.config, source)
            if pipeline.world is not None and self.custom_points is not None:
                pipeline.world.set_trajectory(self.config.trajectory, self.custom_points)
            metadata = asdict(self.config)
            metadata["detector"] = pipeline.detector.status
            metadata["video"] = str(self.video_path) if self.video_path else None
            if source:
                metadata["reported_video_fps"] = source.reported_fps
                metadata["ground_truth"] = str(source.truth_path) if source.truth_path else None
            self.logger = AuditLogger(self.output, metadata)
            self.audit_ready.emit(self.logger)
            self.status.emit(pipeline.detector.status)
            while not self.isInterruptionRequested():
                self.wake.clear()
                while True:
                    try:
                        name, value = self.commands.get_nowait()
                    except Empty:
                        break
                    if name == "pause":
                        paused = bool(value)
                    elif name == "config":
                        pipeline.config = value
                        self.logger.event(pipeline.frame_index / 30, asdict(value))
                    elif name == "designate":
                        pipeline.designate(np.asarray(value, dtype=float))
                        self.logger.event(pipeline.frame_index / 30, {"designation_sensor": list(value)})
                if paused:
                    self.wake.wait(0.1)
                    continue
                started = time.perf_counter()
                try:
                    result = pipeline.step()
                except EOFError:
                    self.status.emit(
                        "BENCHMARK COMPLETE - all decoded frames logged. Generate ISRO Audit to export."
                    )
                    break
                scene = None
                if not self.display_pending.is_set() and pipeline.world is not None:
                    scene = render_scene(pipeline.world, list(pipeline.tracker.history))
                remaining = 1 / 30 - (time.perf_counter() - started)
                if remaining > 0:
                    # A stop/config command can wake this wait without touching worker-owned state.
                    self.wake.wait(remaining)
                result.row["wall_dt_s"] = time.perf_counter() - started
                self.logger.append(result.row)
                if not self.display_pending.is_set():
                    self.display_pending.set()
                    self.packet_ready.emit(
                        {"result": result, "scene": scene, "summary": self.logger.summary()}
                    )
        except Exception as exc:
            LOG.exception("Mission worker stopped")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if pipeline is not None:
                pipeline.close()
            elif source is not None:
                source.close()
            if self.logger is not None:
                self.logger.close()


class ExportWorker(QThread):
    exported = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, logger: AuditLogger) -> None:
        super().__init__()
        self.logger = logger

    def run(self) -> None:
        try:
            self.exported.emit(self.logger.export())
        except Exception as exc:
            LOG.exception("Audit export failed")
            self.failed.emit(f"Export failed: {exc}")


class WeatherWorker(QThread):
    weather_ready = pyqtSignal(float, float, str)
    failed = pyqtSignal(str)

    def __init__(self, city: str, api_key: str) -> None:
        super().__init__()
        self.city, self.api_key = city, api_key

    def run(self) -> None:
        query = urllib.parse.urlencode({"q": self.city, "appid": self.api_key, "units": "metric"})
        try:
            request = urllib.request.Request(
                "https://api.openweathermap.org/data/2.5/weather?" + query,
                headers={"User-Agent": "FSOCTracker/1.0"},
            )
            with urllib.request.urlopen(request, timeout=8) as response:
                data = json.load(response)
            wind = float(data.get("wind", {}).get("speed", 0))
            visibility = float(data.get("visibility", 10000))
            jitter = float(np.clip(wind * 1.5, 0, 20))
            fog = float(np.clip(1 - visibility / 10000, 0, 1))
            self.weather_ready.emit(
                jitter,
                fog,
                f"{self.city}: wind {wind:.1f} m/s, visibility {visibility:.0f} m (heuristic mapping)",
            )
        except urllib.error.HTTPError as exc:
            self.failed.emit(f"Weather HTTP {exc.code}. Check city, key activation and API allowance.")
        except (OSError, ValueError, KeyError, TypeError):
            # urllib exceptions can contain credential-bearing URLs; never surface them.
            self.failed.emit("Weather unavailable. Check network connectivity and try again.")


class ImagePanel(QLabel):
    clicked = pyqtSignal(float, float)

    def __init__(self, text: str, width: int, height: int) -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(width, height)
        self.setStyleSheet("background:#060b10; border:1px solid #263541; color:#627888;")
        self.sensor_clicks = False
        self.image_size = (width, height)
        self.display_rect = (0.0, 0.0, float(width), float(height))

    def show_frame(self, array: Frame) -> None:
        height, width = array.shape[:2]
        fmt = QImage.Format.Format_Grayscale8 if array.ndim == 2 else QImage.Format.Format_BGR888
        contiguous = np.ascontiguousarray(array)
        image = QImage(contiguous.tobytes(), width, height, contiguous.strides[0], fmt).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.image_size = (width, height)
        self.display_rect = (
            (self.width() - pixmap.width()) / 2,
            (self.height() - pixmap.height()) / 2,
            float(pixmap.width()),
            float(pixmap.height()),
        )
        self.setPixmap(pixmap)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and self.sensor_clicks and event.button() == Qt.MouseButton.LeftButton:
            x, y, width, height = self.display_rect
            p = event.position()
            if x <= p.x() < x + width and y <= p.y() < y + height:
                self.clicked.emit(
                    (p.x() - x) * self.image_size[0] / width, (p.y() - y) * self.image_size[1] / height
                )
        super().mousePressEvent(event)


class DropZone(QLabel):
    video_dropped = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__(
            "DROP EVALUATOR .MP4 HERE\nFull frame input | 30 Hz source clock | optional matching .csv"
        )
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(66)
        self.setStyleSheet("border:1px dashed #3b7885; background:#112129; color:#82c9d4; padding:8px;")

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        mime = event.mimeData() if event is not None else None
        if event is not None and mime is not None:
            urls = mime.urls()
            if (
                len(urls) == 1
                and urls[0].isLocalFile()
                and Path(urls[0].toLocalFile()).suffix.lower() == ".mp4"
            ):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent | None) -> None:
        mime = event.mimeData() if event is not None else None
        if event is not None and mime is not None:
            urls = mime.urls()
            if (
                len(urls) == 1
                and urls[0].isLocalFile()
                and Path(urls[0].toLocalFile()).suffix.lower() == ".mp4"
            ):
                self.video_dropped.emit(urls[0].toLocalFile())
                event.acceptProposedAction()


class PathCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(520, 520)
        self.points: list[tuple[float, float]] = []

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.points.append((event.position().x(), event.position().y()))
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.buttons() & Qt.MouseButton.LeftButton:
            p = (event.position().x(), event.position().y())
            if not self.points or np.linalg.norm(np.array(p) - self.points[-1]) > 18:
                self.points.append(p)
                self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b131a"))
        painter.setPen(QPen(QColor("#213340"), 1))
        for value in range(0, 521, 52):
            painter.drawLine(value, 0, value, 520)
            painter.drawLine(0, value, 520, value)
        painter.setPen(QPen(QColor("#ecbd65"), 2))
        for x, y in self.points:
            painter.drawEllipse(QPointF(x, y), 3, 3)
        if len(self.points) >= 4:
            curve = cubic_bspline(np.asarray(self.points))
            painter.setPen(QPen(QColor("#60d5df"), 2))
            for a, b in pairwise(curve):
                painter.drawLine(QPointF(*a), QPointF(*b))
        painter.end()


class PathDialog(QDialog):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Draw a custom B-spline trajectory")
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Click at least four control points, or drag. The beacon traverses the curve and returns.")
        )
        self.canvas = PathCanvas()
        layout.addWidget(self.canvas)
        clear = QPushButton("Clear path")
        clear.clicked.connect(self.clear)
        layout.addWidget(clear)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def clear(self) -> None:
        self.canvas.points.clear()
        self.canvas.update()

    def validate(self) -> None:
        if len(self.canvas.points) < 4:
            QMessageBox.information(self, "Path needs control points", "Draw or click at least four points.")
        else:
            self.accept()


class MissionControl(QMainWindow):
    def __init__(
        self,
        output_directory: Path,
        model_path: Path | None = None,
        *,
        initial_config: RunConfig | None = None,
        video_path: Path | None = None,
        truth_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.output_directory = output_directory
        self.initial_config = initial_config or RunConfig(model_path=model_path)
        self.model_path = model_path or self.initial_config.model_path
        self.worker: MissionWorker
        self.logger: AuditLogger | None = None
        self.export_worker: ExportWorker | None = None
        self.weather_worker: WeatherWorker | None = None
        self.last_packet: dict[str, Any] | None = None
        self.pending_start: tuple[RunConfig, Path | None] | None = None
        self.custom_points: np.ndarray | None = None
        self.closing = False
        self.mode = "simulation"
        self.sliders: dict[str, QSlider] = {}
        self.kpis: dict[str, QLabel] = {}
        self.history: deque[tuple[float, float, float, float]] = deque(maxlen=450)
        self.setWindowTitle("RADIX / FSOC Mission Control - ISRO 26169")
        self.setStyleSheet(STYLE)
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.resize(1460, 960)
        self._build_ui()
        config = self.initial_config
        self.trajectory.setCurrentText(config.trajectory.value)
        self.sliders["speed"].setValue(round(config.speed))
        self.slew.setValue(round(config.slew_deg_s))
        self.decoys.setValue(config.decoys)
        self.occlude.setChecked(config.occluded)
        disturbance = config.disturbances
        for key, value in {
            "jitter": disturbance.jitter,
            "fog": disturbance.fog * 100,
            "salt": disturbance.salt_pepper * 100,
            "gaussian": disturbance.gaussian_sigma,
            "scintillation": disturbance.scintillation * 100,
        }.items():
            self.sliders[key].setValue(round(value))
        self.poisson.setChecked(disturbance.poisson)
        self._start(
            replace(config, mode="video" if video_path else "simulation", model_path=self.model_path),
            video_path,
            truth_path,
        )

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 16, 20, 10)
        header = QHBoxLayout()
        title = QLabel("RADIX  /  FSOC MISSION CONTROL")
        title.setStyleSheet("font-size:21px; font-weight:600; color:#e8f3f8; letter-spacing:2px;")
        header.addWidget(title)
        header.addStretch()
        self.mode_label = QLabel("MODE A  /  LIVE CLOSED LOOP")
        self.mode_label.setStyleSheet("color:#68d3df; font-weight:600;")
        header.addWidget(self.mode_label)
        outer.addLayout(header)
        subtitle = QLabel(
            "COARSE ALIGNMENT LABORATORY     /     2000 x 2000 WORLD     /     640 x 480 FOCAL PLANE     /     PS 26169"
        )
        subtitle.setStyleSheet("color:#657e8f; font-size:10px; padding-bottom:8px;")
        outer.addWidget(subtitle)
        metrics = QHBoxLayout()
        for key, heading in [
            ("error", "TRACKING ERROR / <= 10 px"),
            ("retention", "LOCK RETENTION / > 95%"),
            ("acquisition", "ACQUISITION / <= 2.0 s"),
            ("fps", "THROUGHPUT / >= 20 FPS"),
        ]:
            box = QGroupBox(heading)
            layout = QVBoxLayout(box)
            value = QLabel("--")
            value.setStyleSheet("font-size:26px; font-weight:600; color:#8195a4;")
            layout.addWidget(value)
            self.kpis[key] = value
            metrics.addWidget(box)
        outer.addLayout(metrics)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 6, 0)
        sensor_box = QGroupBox("01 / SENSOR FEED - CLICK A BEACON TO DESIGNATE PRIORITY")
        sensor_layout = QVBoxLayout(sensor_box)
        self.sensor = ImagePanel("Initializing sensor pipeline...", 640, 480)
        self.sensor.setFixedSize(640, 480)
        self.sensor.sensor_clicks = True
        self.sensor.clicked.connect(lambda x, y: self.worker.submit("designate", (x, y)))
        sensor_layout.addWidget(self.sensor, alignment=Qt.AlignmentFlag.AlignCenter)
        self.state_label = QLabel("SEARCHING   |   waiting for observations")
        self.state_label.setStyleSheet("color:#e6b963; padding:4px;")
        sensor_layout.addWidget(self.state_label)
        ll.addWidget(sensor_box)
        self.error_plot = self._plot("TRACKING ERROR CONVERGENCE", "px")
        self.error_curve = self.error_plot.plot(pen=pg.mkPen("#64d5e0", width=1.5))
        self.error_plot.addItem(
            pg.InfiniteLine(pos=10, angle=0, pen=pg.mkPen("#b78c4b", style=Qt.PenStyle.DashLine))
        )
        ll.addWidget(self.error_plot)
        self.velocity_plot = self._plot("MOTOR VELOCITY / PAN - CYAN / TILT - AMBER", "px/s")
        self.pan_curve = self.velocity_plot.plot(pen=pg.mkPen("#64d5e0", width=1.5))
        self.tilt_curve = self.velocity_plot.plot(pen=pg.mkPen("#e6b963", width=1.5))
        ll.addWidget(self.velocity_plot)
        splitter.addWidget(left)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        top = QHBoxLayout()
        debug_box = QGroupBox("02 / VISION DEBUGGER")
        dl = QVBoxLayout(debug_box)
        self.debug = ImagePanel("Binary / Otsu feed", 240, 180)
        self.debug.setFixedSize(240, 180)
        dl.addWidget(self.debug, alignment=Qt.AlignmentFlag.AlignCenter)
        top.addWidget(debug_box)
        scenario_box = QGroupBox("03 / SCENARIO")
        sl = QVBoxLayout(scenario_box)
        self.trajectory = QComboBox()
        self.trajectory.addItems([t.value for t in Trajectory])
        self.trajectory.currentTextChanged.connect(self._trajectory_changed)
        sl.addWidget(self.trajectory)
        draw = QPushButton("Draw B-spline path")
        draw.clicked.connect(self._draw_path)
        sl.addWidget(draw)
        self.decoys = QSpinBox()
        self.decoys.setRange(0, 5)
        self.decoys.setPrefix("Decoys: ")
        self.decoys.valueChanged.connect(self._controls_changed)
        sl.addWidget(self.decoys)
        self.slew = QSpinBox()
        self.slew.setRange(5, 10)
        self.slew.setValue(10)
        self.slew.setSuffix(" deg/s max slew")
        self.slew.valueChanged.connect(self._controls_changed)
        sl.addWidget(self.slew)
        self.occlude = QCheckBox("Occlude priority beacon")
        self.occlude.toggled.connect(self._controls_changed)
        sl.addWidget(self.occlude)
        top.addWidget(scenario_box)
        rl.addLayout(top)
        controls = QGroupBox("04 / DISTURBANCE CONTROL DECK")
        cl = QGridLayout(controls)
        for i, (name, label, maximum, initial, unit) in enumerate(
            [
                ("jitter", "Platform jitter", 20, 0, " px"),
                ("fog", "Fog / contrast loss", 100, 0, "%"),
                ("salt", "Salt & pepper", 10, 0, "%"),
                ("speed", "Target speed", 150, 25, " px/s"),
                ("gaussian", "Gaussian sigma", 20, 0, " DN"),
                ("scintillation", "Scintillation", 100, 0, "%"),
            ]
        ):
            label_widget = QLabel(label)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, maximum)
            slider.setValue(initial)
            readout = QLabel(f"{initial}{unit}")
            readout.setMinimumWidth(62)
            slider.valueChanged.connect(lambda v, out=readout, suffix=unit: out.setText(f"{v}{suffix}"))
            slider.valueChanged.connect(self._controls_changed)
            cl.addWidget(label_widget, i, 0)
            cl.addWidget(slider, i, 1)
            cl.addWidget(readout, i, 2)
            self.sliders[name] = slider
        self.poisson = QCheckBox("Poisson shot noise")
        self.poisson.toggled.connect(self._controls_changed)
        cl.addWidget(self.poisson, 6, 0, 1, 3)
        presets = QHBoxLayout()
        for name in ["Clear Sky", "High Jitter", "Dense Fog"]:
            button = QPushButton(name)
            button.clicked.connect(lambda checked=False, preset=name: self._preset(preset))
            presets.addWidget(button)
        cl.addLayout(presets, 7, 0, 1, 3)
        rl.addWidget(controls)
        weather_box = QGroupBox("05 / LIVE WEATHER - OPENWEATHERMAP")
        wl = QHBoxLayout(weather_box)
        self.city = QLineEdit("Bengaluru,IN")
        self.city.setPlaceholderText("City,country")
        self.api_key = QLineEdit(os.environ.get("OPENWEATHERMAP_API_KEY", ""))
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("API key (not saved)")
        self.weather_button = QPushButton("Fetch weather")
        self.weather_button.clicked.connect(self._fetch_weather)
        wl.addWidget(self.city)
        wl.addWidget(self.api_key)
        wl.addWidget(self.weather_button)
        rl.addWidget(weather_box)
        self.dropzone = DropZone()
        self.dropzone.video_dropped.connect(self.load_video)
        rl.addWidget(self.dropzone)
        scene_box = QGroupBox("06 / WORLD CONTEXT - OPENCV PERSPECTIVE PROJECTION")
        scene_layout = QVBoxLayout(scene_box)
        self.scene = ImagePanel("Perspective scene", 420, 210)
        self.scene.setMaximumHeight(260)
        scene_layout.addWidget(self.scene)
        rl.addWidget(scene_box, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(right)
        scroll.setMinimumWidth(500)
        splitter.addWidget(scroll)
        splitter.setSizes([690, 660])
        outer.addWidget(splitter, 1)
        actions = QHBoxLayout()
        self.pause_button = QPushButton("Pause")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._pause)
        self.reset_button = QPushButton("Reset simulation")
        self.reset_button.clicked.connect(self.reset_simulation)
        browse = QPushButton("Open benchmark video")
        browse.clicked.connect(self._browse)
        self.export_button = QPushButton("Generate ISRO Audit")
        self.export_button.setObjectName("primary")
        self.export_button.clicked.connect(self._export)
        self.export_button.setEnabled(False)
        for button in [self.pause_button, self.reset_button, browse]:
            actions.addWidget(button)
        actions.addStretch()
        actions.addWidget(QLabel("30 Hz SOURCE CLOCK  /  SEE AUDIT FOR MEASURED PERFORMANCE"))
        actions.addWidget(self.export_button)
        outer.addLayout(actions)
        self.setCentralWidget(root)
        self.status_bar.showMessage("Initializing mission...")

    @staticmethod
    def _plot(title: str, units: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(background="#0d151d")
        plot.setTitle(title, color="#859aa9", size="9pt")
        plot.setLabel("left", units, color="#728693")
        plot.setLabel("bottom", "Mission time", units="s", color="#728693")
        plot.showGrid(x=True, y=True, alpha=0.13)
        plot.setMinimumHeight(115)
        plot.setMaximumHeight(155)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        return plot

    def _config(self) -> RunConfig:
        values = {key: slider.value() for key, slider in self.sliders.items()}
        return RunConfig(
            trajectory=Trajectory(self.trajectory.currentText()),
            speed=float(values["speed"]),
            seed=self.initial_config.seed,
            slew_deg_s=float(self.slew.value()),
            decoys=self.decoys.value(),
            occluded=self.occlude.isChecked(),
            model_path=self.model_path,
            disturbances=DisturbanceConfig(
                jitter=float(values["jitter"]),
                fog=values["fog"] / 100,
                salt_pepper=values["salt"] / 100,
                gaussian_sigma=float(values["gaussian"]),
                poisson=self.poisson.isChecked(),
                scintillation=values["scintillation"] / 100,
            ),
        )

    def _controls_changed(self, *_args: Any) -> None:
        if hasattr(self, "worker") and self.mode == "simulation":
            self.worker.submit("config", self._config())

    def _start(self, config: RunConfig, video: Path | None, truth_path: Path | None = None) -> None:
        self.mode = config.mode
        self.history.clear()
        self.last_packet = None
        self.logger = None
        self.export_button.setEnabled(False)
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(False)
        self.pause_button.setText("Pause")
        self.pause_button.blockSignals(False)
        self.mode_label.setText(
            "MODE A  /  LIVE CLOSED LOOP" if video is None else "MODE B  /  VIDEO BENCHMARK"
        )
        for widget in [
            *self.sliders.values(),
            self.trajectory,
            self.decoys,
            self.slew,
            self.occlude,
            self.poisson,
        ]:
            widget.setEnabled(video is None)
        self.weather_button.setEnabled(
            video is None and not (self.weather_worker is not None and self.weather_worker.isRunning())
        )
        self.worker = MissionWorker(config, self.output_directory, video, self.custom_points, truth_path)
        self.worker.packet_ready.connect(self._packet)
        self.worker.status.connect(self.status_bar.showMessage)
        self.worker.failed.connect(self._error)
        self.worker.audit_ready.connect(self._audit_ready)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _request_start(self, config: RunConfig, video: Path | None = None) -> None:
        if self.worker.isRunning():
            self.pending_start = (config, video)
            self.worker.stop()
            self.status_bar.showMessage("Finishing current run and releasing its resources...")
        else:
            self._start(config, video)

    def _worker_finished(self) -> None:
        if self.pending_start is not None and not self.closing:
            config, video = self.pending_start
            self.pending_start = None
            self._start(config, video)

    def _audit_ready(self, logger: AuditLogger) -> None:
        self.logger = logger
        self.export_button.setEnabled(True)

    def _packet(self, packet: dict[str, Any]) -> None:
        self.last_packet = packet
        result: FrameResult = packet["result"]
        row, summary = result.row, packet["summary"]
        self.sensor.show_frame(result.image)
        self.debug.show_frame(result.binary)
        if packet["scene"] is not None:
            self.scene.show_frame(packet["scene"])
        elif self.mode == "video":
            self.scene.clear()
            self.scene.setText(
                "BENCHMARK BYPASS\nWorld and gimbal disabled\nCentroid RMSE uses original-pixel ground truth"
            )
        error = row["pointing_error_px"] if self.mode == "simulation" else row["centroid_error_px"]
        self._kpi("error", error, "px", error is not None and error <= 10)
        retention = summary["lock_retention_pct"]
        self._kpi("retention", retention, "%", retention is not None and retention > 95)
        acquisition = summary["acquisition_s"]
        self._kpi("acquisition", acquisition, "s", acquisition is not None and acquisition <= 2)
        fps = summary["average_fps"]
        self._kpi("fps", fps, "FPS", fps is not None and fps >= 20)
        rmse = summary["centroid_rmse_px"]
        rmse_text = f"{rmse:.3f} px" if rmse is not None else "N/A (no labeled observations)"
        reacq = summary["max_reacquisition_s"]
        reacq_text = f"{reacq:.2f}s" if reacq is not None else "--"
        self.state_label.setText(
            f"{row['state']}  |  {result.backend}  |  Centroid RMSE: {rmse_text}  |  Reacq: {reacq_text}"
        )
        self.history.append(
            (row["time_s"], error if error is not None else float("nan"), row["pan_px_s"], row["tilt_px_s"])
        )
        data = np.array(self.history)
        self.error_curve.setData(data[:, 0], data[:, 1], connect="finite")
        self.pan_curve.setData(data[:, 0], data[:, 2])
        self.tilt_curve.setData(data[:, 0], data[:, 3])
        self.worker.display_pending.clear()

    def _kpi(self, key: str, value: float | None, unit: str, passed: bool) -> None:
        color = "#79d6a0" if passed else "#e8b46a" if value is not None else "#7a8e9c"
        text = f"{value:.2f} {unit}  {'PASS' if passed else 'FAIL'}" if value is not None else "N/A"
        self.kpis[key].setText(text)
        self.kpis[key].setStyleSheet(f"font-size:23px; font-weight:600; color:{color};")

    def _pause(self, checked: bool) -> None:
        self.pause_button.setText("Resume" if checked else "Pause")
        self.worker.submit("pause", checked)

    def reset_simulation(self) -> None:
        self._request_start(self._config())

    def _trajectory_changed(self, _text: str) -> None:
        if hasattr(self, "worker"):
            self.reset_simulation()

    def _draw_path(self) -> None:
        dialog = PathDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.custom_points = np.asarray(dialog.canvas.points) / 520 * 1960 + 20
            self.trajectory.blockSignals(True)
            self.trajectory.setCurrentText(Trajectory.CUSTOM.value)
            self.trajectory.blockSignals(False)
            self.reset_simulation()

    def _preset(self, name: str) -> None:
        presets = {
            "Clear Sky": (0, 0, 0, 0, 0),
            "High Jitter": (20, 5, 2, 3, 10),
            "Dense Fog": (3, 85, 5, 8, 30),
        }
        for key, value in zip(["jitter", "fog", "salt", "gaussian", "scintillation"], presets[name]):
            self.sliders[key].setValue(value)

    def _browse(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Open Benchmark-2 video", "", "MP4 video (*.mp4)")
        if filename:
            self.load_video(filename)

    def load_video(self, filename: str) -> None:
        path = Path(filename)
        if not path.is_file() or path.suffix.lower() != ".mp4":
            self._error("Select an existing .mp4 video file.")
            return
        self._request_start(replace(self._config(), mode="video"), path)

    def _fetch_weather(self) -> None:
        if (
            self.closing
            or self.mode != "simulation"
            or (self.weather_worker is not None and self.weather_worker.isRunning())
        ):
            return
        if not self.api_key.text().strip() or not self.city.text().strip():
            self._error("Enter your OpenWeatherMap API key and city, or set OPENWEATHERMAP_API_KEY.")
            return
        self.weather_button.setEnabled(False)
        self.weather_worker = WeatherWorker(self.city.text().strip(), self.api_key.text().strip())
        self.weather_worker.weather_ready.connect(self._weather_received)
        self.weather_worker.failed.connect(self._error)
        self.weather_worker.finished.connect(
            lambda: self.weather_button.setEnabled(self.mode == "simulation" and not self.closing)
        )
        self.weather_worker.start()

    def _weather_received(self, jitter: float, fog: float, message: str) -> None:
        if self.mode == "simulation" and not self.closing:
            self.sliders["jitter"].setValue(round(jitter))
            self.sliders["fog"].setValue(round(fog * 100))
            self.status_bar.showMessage(message)

    def _export(self) -> None:
        if self.logger is None or (self.export_worker is not None and self.export_worker.isRunning()):
            return
        self.export_button.setEnabled(False)
        self.export_worker = ExportWorker(self.logger)
        self.export_worker.exported.connect(self._exported)
        self.export_worker.failed.connect(self._error)
        self.export_worker.finished.connect(lambda: self.export_button.setEnabled(self.logger is not None))
        self.export_worker.start()

    def _exported(self, paths: list[Path]) -> None:
        self.status_bar.showMessage(f"Audit saved: {next(p for p in paths if p.suffix == '.pdf')}")

    def _error(self, message: str) -> None:
        self.status_bar.showMessage(message)
        if not self.closing:
            QMessageBox.warning(self, "Mission control", message)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        if event is None:
            return
        self.closing = True
        self.pending_start = None
        self.worker.stop()
        busy = self.worker.isRunning() or any(
            w is not None and w.isRunning() for w in [self.export_worker, self.weather_worker]
        )
        if busy:
            event.ignore()
            self.status_bar.showMessage("Closing: waiting for workers to release capture and files...")
            QTimer.singleShot(100, self.close)
        else:
            event.accept()


def launch(
    output: Path,
    model_path: Path | None = None,
    smoke_seconds: float = 0,
    screenshot: Path | None = None,
    *,
    initial_config: RunConfig | None = None,
    video_path: Path | None = None,
    truth_path: Path | None = None,
) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("FSOCTracker")
    window = MissionControl(
        output, model_path, initial_config=initial_config, video_path=video_path, truth_path=truth_path
    )
    window.show()
    if smoke_seconds > 0:

        def finish_smoke() -> None:
            if screenshot is not None:
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                window.grab().save(str(screenshot))
            window.close()

        QTimer.singleShot(round(smoke_seconds * 1000), finish_smoke)
    return app.exec()
```

## fsoc_tracker/pipeline.py

```python
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
```

## fsoc_tracker/audit_exporter.py

```python
"""Incremental telemetry with snapshot-safe, independently named audit exports."""

from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class AuditLogger:
    def __init__(self, directory: Path, metadata: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.directory = directory
        self.session = f"FSOC_{stamp}"
        self.metadata = metadata
        self.path = directory / f"{self.session}_live.csv"
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer: csv.DictWriter[str] | None = None
        self.lock = RLock()
        self.count = 0
        self.duration = self.wall_duration = self.processing_total = 0.0
        self.detected = self.locked = self.after_acquisition = self.retained = 0
        self.point_sum = self.point_sq = self.point_peak = 0.0
        self.point_count = self.centroid_count = 0
        self.centroid_sq = self.centroid_peak = 0.0
        self.acquisition: float | None = None
        self.reacquisition: float | None = None
        self.closed = False

    def append(self, row: dict[str, Any]) -> None:
        with self.lock:
            if self.closed:
                raise RuntimeError("Audit log is closed.")
            if self.writer is None:
                self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
            self.count += 1
            self.duration += float(row["dt_s"])
            self.wall_duration += float(row["wall_dt_s"])
            self.processing_total += float(row["processing_ms"])
            self.detected += bool(row["detected"])
            self.locked += bool(row["locked"])
            if self.acquisition is None and row["locked"]:
                self.acquisition = float(row["time_s"])
            if self.acquisition is not None:
                self.after_acquisition += 1
                self.retained += bool(row["locked"])
            if row.get("reacquisition_s") is not None:
                value = float(row["reacquisition_s"])
                self.reacquisition = max(self.reacquisition or 0.0, value)
            point = row.get("pointing_error_px")
            if point is not None and math.isfinite(point):
                self.point_count += 1
                self.point_sum += point
                self.point_sq += point**2
                self.point_peak = max(self.point_peak, point)
            centroid = row.get("centroid_error_px")
            if centroid is not None and math.isfinite(centroid):
                self.centroid_count += 1
                self.centroid_sq += centroid**2
                self.centroid_peak = max(self.centroid_peak, centroid)
            if self.count % 30 == 0:
                self.handle.flush()

    def event(self, time_s: float, config: dict[str, Any]) -> None:
        with self.lock:
            self.metadata.setdefault("events", []).append({"time_s": time_s, "config": config})

    def summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "mode": self.metadata.get("mode", "unknown"),
                "frames": self.count,
                "duration_s": self.duration,
                "average_fps": self.count / self.wall_duration if self.wall_duration else None,
                "processing_fps": 1000 * self.count / self.processing_total
                if self.processing_total
                else None,
                "mean_processing_ms": self.processing_total / self.count if self.count else None,
                "acquisition_s": self.acquisition,
                "max_reacquisition_s": self.reacquisition,
                "average_pointing_error_px": self.point_sum / self.point_count if self.point_count else None,
                "peak_pointing_error_px": self.point_peak if self.point_count else None,
                "pointing_rmse_px": math.sqrt(self.point_sq / self.point_count) if self.point_count else None,
                "centroid_rmse_px": math.sqrt(self.centroid_sq / self.centroid_count)
                if self.centroid_count
                else None,
                "peak_centroid_error_px": self.centroid_peak if self.centroid_count else None,
                "ground_truth_matched_frames": self.centroid_count,
                "detection_rate_pct": 100 * self.detected / self.count if self.count else None,
                "lock_retention_pct": 100 * self.retained / self.after_acquisition
                if self.after_acquisition
                else None,
                "whole_run_lock_pct": 100 * self.locked / self.count if self.count else None,
            }

    def export(self) -> list[Path]:
        stamp = datetime.now(timezone.utc).strftime("%H%M%S_%f")
        stem = self.directory / f"{self.session}_audit_{stamp}"
        raw_path = Path(f"{stem}_frames.csv")
        summary_path = Path(f"{stem}_summary.csv")
        metadata_path = Path(f"{stem}_config.json")
        pdf_path = Path(f"{stem}.pdf")
        with self.lock:
            if not self.closed:
                self.handle.flush()
            summary = self.summary()
            metadata = json.loads(json.dumps(self.metadata, default=str))
            shutil.copyfile(self.path, raw_path)
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._pdf(pdf_path, summary, metadata)
        return [raw_path, summary_path, metadata_path, pdf_path]

    @staticmethod
    def _pdf(path: Path, summary: dict[str, Any], metadata: dict[str, Any]) -> None:
        styles = getSampleStyleSheet()
        document = SimpleDocTemplate(
            str(path),
            pagesize=(8.27 * inch, 11.69 * inch),
            rightMargin=42,
            leftMargin=42,
            topMargin=40,
            bottomMargin=40,
        )
        story: list[Any] = [
            Paragraph("FSOC | Coarse Alignment Audit", styles["Title"]),
            Paragraph("ISRO Problem Statement 26169 - measured software run", styles["Normal"]),
            Spacer(1, 14),
        ]
        rows: list[list[Any]] = [["Metric", "Measured value"]]
        for name, value in summary.items():
            formatted = "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
            rows.append([name.replace("_", " "), formatted])
        table = Table(rows, colWidths=[330, 175], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#13232d")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f5f7"), colors.white]),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.extend(
            [table, Spacer(1, 16), Paragraph("Metric definitions and interpretation", styles["Heading2"])]
        )
        notes = [
            "Acquisition: first three consecutive associated observations. Reacquisition: loss onset to confirmed lock.",
            "Lock retention: confirmed visual lock frames after first acquisition / all frames after acquisition. Prediction-only frames count as lost.",
            "Centroid RMSE: sqrt(mean(dx^2 + dy^2)) over frames with both accepted detection and ground truth. Missing observations are reported by detection rate, not assigned zero error.",
            "Simulation centroid and pointing errors use sensor pixels, including injected jitter. Benchmark centroid error uses original video pixels; video pointing error is observed displacement from the 640 x 480 boresight, not centroid accuracy.",
            "Average FPS includes processing and pacing but excludes operator pauses. Processing FPS excludes pacing. Source time advances by 1/30 s per frame.",
            "Targets: pointing error <= 10 px; retention > 95%; acquisition <= 2 s; reacquisition <= 1 s; throughput >= 20 FPS. These are acceptance criteria, not guaranteed outcomes.",
            "No ground truth means N/A centroid RMSE. This report is a software measurement, not an ISRO certification.",
            f"Configuration seed: {metadata.get('seed', 'N/A')}. Mode: {metadata.get('mode', 'N/A')}. See companion JSON for configuration changes and detector provenance.",
        ]
        for note in notes:
            story.extend([Paragraph(escape(note), styles["BodyText"]), Spacer(1, 5)])
        document.build(story)

    def close(self) -> None:
        with self.lock:
            if not self.closed:
                self.handle.flush()
                self.handle.close()
                self.closed = True
```

## fsoc_tracker/scene_view.py

```python
"""AstraTrack-inspired perspective context view, drawn entirely by OpenCV.

This illustration projects the planar world into a tilted 3D reference plane;
it is not another dynamics model and never supplies perception measurements.
"""

from __future__ import annotations

import cv2
import numpy as np

from .math3d_engine import Frame, Vec, VirtualWorldEngine


def render_scene(world: VirtualWorldEngine, trail: list[Vec], width: int = 640, height: int = 330) -> Frame:
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = (20, 16, 12)
    source = np.array([[0, 0], [2000, 0], [2000, 2000], [0, 2000]], dtype=np.float32)
    destination = np.array([[190, 55], [460, 55], [630, 295], [10, 295]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, destination)

    def project(points: list[Vec] | Vec) -> np.ndarray:
        return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(1, -1, 2), homography)[
            0
        ].astype(np.int32)

    for value in range(0, 2001, 200):
        for points in [[[value, 0], [value, 2000]], [[0, value], [2000, value]]]:
            a, b = project(np.array(points))
            cv2.line(image, tuple(a), tuple(b), (47, 43, 34), 1, cv2.LINE_AA)
    origin = world.origin
    footprint = project(np.array([origin, origin + [640, 0], origin + [640, 480], origin + [0, 480]]))
    cv2.polylines(image, [footprint], True, (210, 195, 65), 1, cv2.LINE_AA)
    camera, target = project(np.array([world.camera, world.target]))
    elevated = camera + [0, -75]
    for corner in footprint:
        cv2.line(image, tuple(elevated), tuple(corner), (93, 90, 38), 1, cv2.LINE_AA)
    cv2.line(image, tuple(elevated), tuple(target), (135, 235, 85), 1, cv2.LINE_AA)
    cv2.circle(image, tuple(elevated), 6, (210, 195, 65), -1, cv2.LINE_AA)
    cv2.circle(image, tuple(target), 5, (135, 235, 85), -1, cv2.LINE_AA)
    if len(trail) > 1:
        cv2.polylines(image, [project(trail)], False, (40, 185, 245), 1, cv2.LINE_AA)
    for decoy in world.decoy_positions:
        p = project([decoy])[0]
        cv2.circle(image, tuple(p), 3, (110, 110, 130), -1)
    cv2.putText(
        image,
        "PLANAR WORLD / PERSPECTIVE CONTEXT",
        (15, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (170, 180, 185),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "2000 x 2000 px    |    CYAN: SENSOR FOOTPRINT",
        (15, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    return image
```

## fsoc_tracker/__main__.py

```python
"""Desktop entrypoint and repeatable headless benchmark CLI."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .audit_exporter import AuditLogger
from .math3d_engine import Trajectory
from .pipeline import RunConfig, TrackingPipeline, VideoSource


def main() -> int:
    parser = argparse.ArgumentParser(description="FSOC coarse alignment mission control")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, default=Path.home() / "Documents" / "FSOCTracker" / "audits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=25.0)
    parser.add_argument(
        "--trajectory", choices=[t.value for t in Trajectory], default=Trajectory.STRAIGHT.value
    )
    parser.add_argument("--realtime", action="store_true", help="Pace headless runs to 30 Hz.")
    parser.add_argument("--smoke-seconds", type=float, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--screenshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.frames <= 0 or not 0 <= args.speed <= 150:
        parser.error("frames must be positive; speed must be between 0 and 150 px/s.")
    args.output.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        args.output / "fsoc.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO, handlers=[handler], format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    model = args.model or bundle / "models" / "yolov8n_fsoc.onnx"
    config = RunConfig(
        mode="video" if args.video else "simulation",
        trajectory=Trajectory(args.trajectory),
        speed=args.speed,
        seed=args.seed,
        model_path=model,
    )
    if args.truth and not args.video:
        parser.error("--truth requires --video.")
    if not args.headless:
        from .main_dashboard import launch

        return launch(
            args.output,
            model,
            args.smoke_seconds,
            args.screenshot,
            initial_config=config,
            video_path=args.video,
            truth_path=args.truth,
        )
    pipeline: TrackingPipeline | None = None
    logger: AuditLogger | None = None
    try:
        source = VideoSource(args.video, args.truth) if args.video else None
        pipeline = TrackingPipeline(config, source)
        metadata = asdict(config)
        metadata["video"] = str(args.video) if args.video else None
        metadata["detector"] = pipeline.detector.status
        if source:
            metadata["ground_truth"] = str(source.truth_path) if source.truth_path else None
            metadata["reported_video_fps"] = source.reported_fps
        logger = AuditLogger(args.output, metadata)
        for _ in range(args.frames):
            started = time.perf_counter()
            try:
                result = pipeline.step()
            except EOFError:
                break
            if args.realtime:
                time.sleep(max(0, 1 / 30 - (time.perf_counter() - started)))
            result.row["wall_dt_s"] = time.perf_counter() - started
            logger.append(result.row)
        paths = logger.export()
        output = {"summary": logger.summary(), "artifacts": [str(path) for path in paths]}
        if sys.stdout is not None:
            print(json.dumps(output, indent=2))
        return 0
    except Exception as exc:
        logging.getLogger(__name__).exception("Headless run failed")
        if sys.stderr is not None:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
```

## fsoc_tracker/__init__.py

```python
"""FSOC coarse-alignment simulation and benchmark tools."""

__version__ = "1.0.0"
```

## tests/test_core.py

```python
from pathlib import Path

import cv2
import numpy as np
import pytest

from fsoc_tracker.audit_exporter import AuditLogger
from fsoc_tracker.disturbances import DisturbanceConfig, DisturbanceInjector
from fsoc_tracker.math3d_engine import Trajectory, VirtualWorldEngine
from fsoc_tracker.perception_ekf import BeaconDetector, PriorityTracker, TargetTrackerEKF
from fsoc_tracker.pid_fsm import GainScheduledPID, TrackingFSM, TrackingState
from fsoc_tracker.pipeline import RunConfig, TrackingPipeline, VideoSource


def test_slew_is_per_second_and_sensor_never_wraps():
    world = VirtualWorldEngine(seed=1)
    start = world.camera.copy()
    velocity = world.slew_camera(np.array([999.0, -999.0]), 0.1, 5.0)
    np.testing.assert_allclose(velocity, [50, -50])
    np.testing.assert_allclose(world.camera - start, [5, -5])
    for _ in range(100):
        world.slew_camera(np.array([-1e6, 1e6]), 1.0, 10.0)
    assert world.render_viewport().shape == (480, 640)
    assert np.all(world.camera >= [320, 240])
    assert np.all(world.camera <= [1680, 1760])
    np.testing.assert_allclose(world.slew_camera(np.zeros(2), 0.1, 5), 0)


@pytest.mark.parametrize("trajectory", list(Trajectory))
def test_all_trajectories_are_bounded_and_speed_limited(trajectory):
    world = VirtualWorldEngine(seed=4)
    world.set_trajectory(trajectory)
    last = world.target.copy()
    for _ in range(600):
        world.step(1 / 30, 40)
        assert np.linalg.norm(world.target - last) <= 40 / 30 + 1e-5
        assert np.all((world.target >= 20) & (world.target <= 1980))
        last = world.target.copy()


def test_disturbances_repeat_and_include_both_impulse_polarities():
    frame = np.full((480, 640), 128, np.uint8)
    a, b = DisturbanceInjector(10), DisturbanceInjector(10)
    cfg = DisturbanceConfig(salt_pepper=0.1)
    out, _ = a.apply(frame, cfg)
    np.testing.assert_array_equal(out, b.apply(frame, cfg)[0])
    assert (out == 0).sum() == (out == 255).sum() == int(frame.size * 0.1) // 2
    assert np.all(frame == 128)


def test_detector_separates_spots_in_impulse_noise():
    frame = np.zeros((480, 640), np.uint8)
    frame[95:105, 195:205] = 255
    frame[295:305, 495:505] = 200
    frame, _ = DisturbanceInjector(9).apply(frame, DisturbanceConfig(salt_pepper=0.1))
    detections, _ = BeaconDetector().detect(frame)
    assert len(detections) == 2
    assert min(np.linalg.norm(d.center - [199.5, 99.5]) for d in detections) < 1


def test_ekf_predicts_through_occlusion_and_covariance_stays_positive():
    ekf = TargetTrackerEKF()
    ekf.initialize(np.array([0.0, 0.0]))
    for i in range(1, 150):
        ekf.predict(1 / 30)
        ekf.update(np.array([i * 2.0, i * 0.5]))
    last = ekf.position.copy()
    for _ in range(15):
        ekf.predict(1 / 30)
    assert np.linalg.norm(ekf.position - last - [30, 7.5]) < 3
    assert np.linalg.eigvalsh(ekf.P).min() > 0


def test_crossing_decoy_does_not_reverse_priority_track():
    tracker = PriorityTracker()
    detector = BeaconDetector()
    tracker.designate(np.array([220.0, 240.0]))
    last = None
    for i in range(90):
        frame = np.zeros((480, 640), np.uint8)
        x, other = 220 + i * 2, 420 - i * 2
        cv2.rectangle(frame, (x - 5, 235), (x + 4, 244), 255, -1)
        cv2.rectangle(frame, (other - 5, 235), (other + 4, 244), 210, -1)
        ds, _ = detector.detect(frame)
        last = tracker.step(ds, np.zeros(2), 1 / 30)
    assert last is not None
    assert abs(tracker.ekf.position[0] - 397.5) < 4
    assert tracker.ekf.velocity[0] > 40


def test_fsm_visits_all_seven_states():
    fsm = TrackingFSM()
    seen = {fsm.state}
    for detected in [True] * 8 + [False] * 4 + [True] * 8:
        seen.add(fsm.step(detected, True, 0.033))
    assert seen == set(TrackingState)


def test_pid_saturation_and_zero_command():
    pid = GainScheduledPID()
    assert np.max(np.abs(pid.compute(np.array([1e4, -1e4]), np.zeros(2), 0.033, 50))) <= 50
    pid.reset()
    np.testing.assert_allclose(pid.compute(np.zeros(2), np.zeros(2), 0.033, 50), 0)


def test_closed_loop_converges_without_ground_truth_control():
    pipe = TrackingPipeline(RunConfig(speed=25, decoys=0))
    results = [pipe.step() for _ in range(300)]
    assert results[-1].row["pointing_error_px"] < 10
    assert np.mean([r.row["detected"] for r in results]) > 0.95


def test_audit_rows_record_actual_detector_backend():
    result = TrackingPipeline(RunConfig()).step()
    assert result.row["detector_backend"] == "OpenCV moments"
    assert result.row["detector_status"] == result.detector_status


def test_video_preserves_source_coordinates_and_has_no_world(tmp_path: Path):
    path = tmp_path / "beacon.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (800, 400))
    assert writer.isOpened()
    for _ in range(8):
        frame = np.zeros((400, 800, 3), np.uint8)
        frame[195:205, 595:605] = 255
        writer.write(frame)
    writer.release()
    path.with_suffix(".csv").write_text("frame,x,y\n" + "".join(f"{i},599.5,199.5\n" for i in range(8)))
    source = VideoSource(path)
    pipe = TrackingPipeline(RunConfig(mode="video"), source)
    assert pipe.world is None
    rows = [pipe.step().row for _ in range(8)]
    with pytest.raises(EOFError):
        pipe.step()
    pipe.close()
    assert abs(rows[-1]["centroid_x"] - 599.5) < 2
    assert rows[-1]["centroid_error_px"] < 2
    assert rows[-1]["pan_px_s"] == rows[-1]["tilt_px_s"] == 0


def test_audit_missing_truth_is_not_zero_rmse(tmp_path: Path):
    log = AuditLogger(tmp_path, {"mode": "video"})
    log.append(
        {
            "time_s": 0.033,
            "dt_s": 0.033,
            "detected": False,
            "locked": False,
            "centroid_error_px": None,
            "pointing_error_px": None,
            "processing_ms": 2.0,
            "frame": 0,
            "state": "SEARCHING",
            "wall_dt_s": 0.033,
        }
    )
    summary = log.summary()
    assert summary["centroid_rmse_px"] is None
    paths = log.export()
    assert all(p.exists() for p in paths)
    assert any(p.suffix == ".pdf" for p in paths)
    log.close()


def test_default_priority_is_bright_beacon_even_when_decoy_is_closer():
    pipe = TrackingPipeline(RunConfig(trajectory=Trajectory.CIRCULAR, speed=25, decoys=3))
    result = pipe.step()
    assert result.row["centroid_error_px"] < 1


def test_jitter_uncertainty_is_used_without_revealing_injected_translation():
    pipe = TrackingPipeline(RunConfig(disturbances=DisturbanceConfig(jitter=20)))
    rows = [pipe.step().row for _ in range(180)]
    assert np.mean([row["detected"] for row in rows]) > 0.95


def test_occlusion_coasts_and_reacquires_without_identity_reset():
    from dataclasses import replace

    pipe = TrackingPipeline(RunConfig(speed=25))
    for _ in range(90):
        pipe.step()
    pipe.config = replace(pipe.config, occluded=True)
    lost = [pipe.step().row for _ in range(15)]
    assert all(not row["detected"] for row in lost)
    assert any(row["state"] == "PREDICTING" for row in lost)
    assert abs(lost[-1]["pan_px_s"]) > 1
    pipe.config = replace(pipe.config, occluded=False)
    regained = [pipe.step().row for _ in range(10)]
    assert regained[-1]["locked"]
    assert regained[-1]["reacquisition_s"] < 1


def test_fsm_abandons_coast_for_spiral_search():
    fsm = TrackingFSM()
    for _ in range(5):
        fsm.step(True, True, 0.1)
    for _ in range(30):
        fsm.step(False, True, 0.1)
    assert fsm.state is TrackingState.SEARCHING


def test_invalid_model_falls_back(tmp_path: Path):
    invalid = tmp_path / "broken.onnx"
    invalid.write_bytes(b"not an ONNX graph")
    detector = BeaconDetector(invalid)
    frame = np.zeros((480, 640), np.uint8)
    frame[235:245, 315:325] = 255
    detections, _ = detector.detect(frame)
    assert len(detections) == 1
    assert detector.backend == "OpenCV moments"


@pytest.mark.parametrize("labels", ["frame,x,y\n0,nan,1\n", "frame,x,y\n0,1,1\n0,2,2\n"])
def test_video_rejects_nonfinite_or_duplicate_ground_truth(tmp_path: Path, labels: str):
    video = tmp_path / "invalid_labels.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    assert writer.isOpened()
    writer.write(np.zeros((480, 640, 3), np.uint8))
    writer.release()
    video.with_suffix(".csv").write_text(labels)
    with pytest.raises(ValueError, match="invalid or duplicate"):
        VideoSource(video)
```

## tests/test_dashboard.py

```python
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
```

## tests/test_entrypoint.py

```python
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
```

## tests/test_onnx.py

```python
"""Real ONNX Runtime contract check with a synthetic graph, not trained AI weights."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fsoc_tracker.perception_ekf import BeaconDetector


def test_yolov8_raw_output_letterbox_nms_and_moment_refinement(tmp_path: Path) -> None:
    predictions = np.zeros((1, 5, 8400), np.float32)
    # Input is 640x640: the 640x480 sensor has an 80-pixel top pad.
    predictions[0, :, 0] = [320, 320, 20, 20, 0.95]
    predictions[0, :, 1] = [321, 320, 20, 20, 0.8]
    node = helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(predictions))
    graph = helper.make_graph(
        [node],
        "test_raw_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5, 8400])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "synthetic_contract.onnx"
    onnx.save(model, path)
    detector = BeaconDetector(path)
    frame = np.zeros((480, 640), np.uint8)
    frame[235:245, 315:325] = 255
    detections, _ = detector.detect(frame)
    assert detector.backend == "YOLOv8 ONNX"
    assert len(detections) == 1
    np.testing.assert_allclose(detections[0].center, [319.5, 239.5], atol=0.1)
```

## tools/package_source.py

```python
"""Build a complete readable codebook, source ZIP and SHA-256 delivery manifest."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    output = root / "output" / "delivery"
    output.mkdir(parents=True, exist_ok=True)
    folders = ("fsoc_tracker", "tests", "tools", "docs", "models", "examples")
    files = sorted(
        path
        for folder in folders
        for path in (root / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".onnx"}
    )
    files += [
        root / name
        for name in (
            "README.md",
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-lock-windows-py313.txt",
            "pyproject.toml",
            "fsoc_tracker.spec",
            "run_tracker.py",
            "Start-FSOC.ps1",
            ".gitignore",
        )
    ]
    priority = [
        root / "fsoc_tracker" / name
        for name in (
            "math3d_engine.py",
            "disturbances.py",
            "perception_ekf.py",
            "pid_fsm.py",
            "main_dashboard.py",
            "pipeline.py",
            "audit_exporter.py",
            "scene_view.py",
            "__main__.py",
            "__init__.py",
        )
    ]
    code = priority + [
        path
        for path in files
        if path not in priority and path.suffix in {".py", ".spec", ".ps1", ".toml", ".txt"}
    ]
    tree = "\n".join(path.relative_to(root).as_posix() for path in sorted(files))
    content = ["# Complete FSOC Tracker source\n\n## Directory structure\n\n```text\n" + tree + "\n```\n"]
    for path in code:
        language = {".py": "python", ".spec": "python", ".ps1": "powershell", ".toml": "toml"}.get(
            path.suffix, "text"
        )
        content.append(
            f"\n## {path.relative_to(root).as_posix()}\n\n```{language}\n"
            + path.read_text(encoding="utf-8").rstrip()
            + "\n```\n"
        )
    book = output / "COMPLETE_SOURCE.md"
    book.write_text("".join(content), encoding="utf-8")
    archive = output / "FSOCTracker-source.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, "FSOCTracker/" + path.relative_to(root).as_posix())
        bundle.write(book, "FSOCTracker/COMPLETE_SOURCE.md")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed.")
        for path in files:
            if bundle.read("FSOCTracker/" + path.relative_to(root).as_posix()) != path.read_bytes():
                raise RuntimeError(f"Archive differs from source: {path}")
    artifacts = [archive, book]
    executable = root / "dist" / "FSOCTracker.exe"
    if executable.is_file():
        artifacts.append(executable)
    manifest = {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in artifacts
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Packaged {len(files)} files and {len(code)} complete source/configuration listings.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
```

## tools/smoke_executable.py

```python
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
```

## tools/validate_scenarios.py

```python
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
```

## requirements.txt

```text
numpy>=1.26,<3
opencv-python>=4.9,<5
PyQt6>=6.6,<7
pyqtgraph>=0.13.7,<0.15
pandas>=2.2,<4
reportlab>=4.1,<5
onnxruntime>=1.18,<2
```

## requirements-dev.txt

```text
-r requirements.txt
pytest>=8,<10
pyinstaller>=6.10,<7
ruff>=0.11
mypy>=1.15
onnx>=1.16,<2
```

## requirements-lock-windows-py313.txt

```text
# Exact local validation/build environment; includes development tools.
# For other Python versions use requirements.txt.
altgraph==0.17.5
ast_serialize==0.11.2
charset-normalizer==3.5.1
colorama==0.4.6
flatbuffers==25.12.19
iniconfig==2.3.0
librt==0.15.0
ml_dtypes==0.6.0
mypy==2.3.1
mypy_extensions==1.1.0
numpy==2.5.3
onnx==1.23.0
onnxruntime==1.30.0
opencv-python==4.14.0.94
packaging==26.3
pandas==3.0.6
pathspec==1.1.1
pefile==2024.8.26
pillow==12.3.0
pluggy==1.6.0
protobuf==7.36.2
Pygments==2.21.0
pyinstaller==6.22.3
pyinstaller-hooks-contrib==2026.7
pymupdf==1.28.2
pypdf==6.19.0
PyQt6==6.11.0
PyQt6-Qt6==6.11.2
PyQt6_sip==13.12.0
pyqtgraph==0.14.0
pytest==9.1.1
python-dateutil==2.9.0.post0
pywin32-ctypes==0.2.3
reportlab==4.5.1
ruff==0.16.8
setuptools==84.0.0
six==1.17.0
typing_extensions==4.16.0
tzdata==2026.4
```

## pyproject.toml

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.ruff]
line-length = 110
target-version = "py310"

[tool.mypy]
python_version = "3.10"
check_untyped_defs = true
disallow_untyped_defs = true
ignore_missing_imports = true
files = ["fsoc_tracker"]
```

## fsoc_tracker.spec

```python
# -*- mode: python ; coding: utf-8 -*-
"""One-file Windows build; optional models are included only when present."""
from pathlib import Path
import os
import sys
import PyQt6
from PyInstaller.utils.hooks import collect_dynamic_libs

root = Path(SPECPATH)
qt_bin = Path(PyQt6.__file__).resolve().parent / 'Qt6' / 'bin'
# Isolate DLL discovery from unrelated native tools injected into the host PATH.
# In particular, Poppler's version-suffixed ICU is incompatible with Qt's use of
# the Windows system ICU API, despite having the same icuuc.dll filename.
if sys.platform == 'win32':
    windows = Path(os.environ['SystemRoot'])
    os.environ['PATH'] = os.pathsep.join(map(str, [qt_bin, Path(sys.base_prefix),
        Path(sys.base_prefix) / 'DLLs', windows / 'System32', windows]))
datas = [(str(root / 'models' / 'README.md'), 'models')]
datas += [(str(path), 'models') for path in (root / 'models').glob('*.onnx')]
a = Analysis(
    [str(root / 'run_tracker.py')], pathex=[str(root)],
    binaries=collect_dynamic_libs('onnxruntime'), datas=datas,
    hiddenimports=['pyqtgraph', 'onnxruntime', 'reportlab.pdfbase._fontdata'],
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'PyQt5', 'PySide2', 'PySide6', 'IPython', 'pytest'],
    noarchive=False,
)
# A build host's PATH can contain mismatched MSVC runtime DLLs.
# Qt requires matching CRT exports. Normalize all collected copies and
# the extraction-root copies to the runtime shipped with this PyQt6 installation.
crt = {p.name.lower(): p for pattern in ('msvcp140*.dll', 'vcruntime140*.dll')
       for p in qt_bin.glob(pattern)}
a.binaries = [(dest, str(crt[Path(dest).name.lower()]), kind)
              if Path(dest).name.lower() in crt else (dest, source, kind)
              for dest, source, kind in a.binaries]
root_names = {dest.lower() for dest, _, _ in a.binaries}
a.binaries += [(path.name, str(path), 'BINARY') for name, path in crt.items()
               if name not in root_names]
# Qt on modern Windows uses the OS ICU unless its wheel supplies its own build.
# Filter cached analysis records too, so incremental builds cannot retain a
# third-party ICU selected by a previous, contaminated PATH.
system_icu = {'icuuc.dll', 'icuin.dll', 'icudt.dll'}
a.binaries = [entry for entry in a.binaries
              if Path(entry[0]).name.lower() not in system_icu
              or (qt_bin / Path(entry[0]).name).is_file()]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [], name='FSOCTracker',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
)
```

## run_tracker.py

```python
from fsoc_tracker.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
```

## Start-FSOC.ps1

```powershell
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    py -3 -m venv (Join-Path $taskRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.10+ and retry.' }
    & $taskPython -m pip install -r (Join-Path $taskRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
& $taskPython (Join-Path $taskRoot 'run_tracker.py') @args
```
