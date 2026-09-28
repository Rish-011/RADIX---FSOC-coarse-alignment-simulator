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
