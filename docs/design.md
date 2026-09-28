# FSOC mission control design

The user's explicit requirements authorize this new application. The PDF and pasted
text are reference data, not instructions to execute. AstraTrack supplies visual
inspiration (ground grid, line of sight, camera footprint); no source is copied.

Pure NumPy owns a 2000 x 2000 monochrome world and 640 x 480 sensor window. Camera
rate limits are configurable maximums in [5,10] degrees/second mapped to [50,100]
pixels/second. Velocity may be zero. This requested display mapping is not a
physical calibration for the PDF's suggested 4 x 3 degree optical FOV.

Five trajectories include an arc-length sampled cubic B-spline implemented in
NumPy. World and disturbance PRNGs have separate seeds. Perception receives only
pixels and known camera pose, never target truth or the injected jitter vector.
A nonlinear coordinated-turn EKF estimates position, velocity and turn rate.
Temporal association gates all detections, rejects ambiguous crossings, and
coasts through occlusion. Identical unresolved beacons cannot be guaranteed
distinguishable; an operator can designate a beacon by clicking the sensor.

Benchmark mode has no world, gimbal, PID actuation, or injected disturbances. It
reads every video frame at a fixed 30 Hz source timebase, letterboxes the complete
frame, and restores source coordinates in logs. Optional sidecar CSV columns
frame,x,y are zero-based source-frame ground truth. Missing labels yield N/A
centroid RMSE, never a fabricated metric. Boresight displacement is separate.

A QThread owns the pipeline and capture; commands cross a thread-safe queue.
Only one unacknowledged display packet may be queued; telemetry still records
every processed frame. Weather and report exports run in separate workers.
Shutdown requests interruption and waits asynchronously for workers to finish.

Audit files include per-frame CSV, summary CSV, JSON configuration provenance,
and a ReportLab PDF. Tracking, acquisition, reacquisition and FPS thresholds are
measured, not promised for physically unreachable or noisy scenarios.
