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
