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
