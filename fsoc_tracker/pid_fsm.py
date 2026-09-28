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
