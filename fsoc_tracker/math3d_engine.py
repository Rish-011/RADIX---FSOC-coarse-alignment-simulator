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
