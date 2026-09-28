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
