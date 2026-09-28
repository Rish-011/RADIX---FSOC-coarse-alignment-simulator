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
