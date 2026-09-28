# Optional beacon detector

Place beacon-trained YOLOv8-Nano weights here as `yolov8n_fsoc.onnx`, or use
`--model C:\path\to\beacon.onnx`. No weights were provided with this request;
the application runs using the OpenCV spatial-moments detector by default.

Supported export: one float32 NCHW RGB input, raw `[1,4+classes,N]` output,
absolute input-pixel center-x, center-y, width, height, class scores; export with
`nms=False`, `half=False`, batch 1. Standard letterboxing and class-agnostic NMS
are applied. All exported classes should be beacon classes. Train on representative
beacon/noise data and validate before claiming AI-model performance. Generic
COCO YOLO weights are not trained for laser spots. Segmentation, pose and
end-to-end NMS exports are not supported.
