# Operator guide

Launch `FSOCTracker.exe` or `python run_tracker.py`. Mode A starts automatically.

1. Watch the sensor image and tracking state. Cyan marks the fixed boresight and
   10 px boundary; green brackets mark the associated beacon; gray brackets show
   other detections. Amber shows the EKF trail and predicted position.
2. Choose Straight Line, Circular, Figure of 8, Random Walk or Custom B-Spline.
   Changing trajectory starts a new session and resets telemetry. The custom editor
   accepts clicks or a dragged stroke; four control points are required. Its preview
   is the actual cubic B-spline, not a straight line joining clicks.
3. Set target speed in px/s and the per-axis maximum gimbal speed in deg/s.
   Five deg/s maps to 50 px/s; ten maps to 100 px/s. Targets faster than the camera
   and targets near world boundaries may not remain centered.
4. Add decoys, then click the desired beacon to designate it. Automatic initial
   selection uses proximity to the boresight. Association subsequently uses history.
   When merged detections are ambiguous the estimator coasts instead of deliberately
   switching identity. A new designation resets the tracker but remains in the same
   audit session and is recorded as an event.
5. Select Occlude priority beacon to remove its image while retaining its motion.
   Observe TARGET_LOST, PREDICTING and, on recovery, REACQUIRING. After two seconds
   without detections the camera searches an expanding Archimedean spiral.
6. Try the three disturbance presets or set noise independently. Salt/pepper is
   capped at 10% with equal distinct black and white pixels; jitter is uniform
   within +/-20 px per frame; Gaussian noise is in digital intensity units (DN).
   Scintillation and Poisson noise are separately selectable. The fixed seed makes
   repeat runs reproducible when settings and their frame-change times are identical.
7. Enter a city and an OpenWeatherMap API key, then Fetch weather. Wind maps to
   `clip(1.5 * wind_m_s, 0, 20)` px jitter; visibility maps to
   `clip(1 - visibility_m/10000, 0, 1)` fog. This is a demonstration heuristic,
   not a calibrated atmosphere or platform-dynamics model.
8. Drop a single local MP4 or browse to it to enter Benchmark-2. A same-name CSV
   supplies optional original-resolution centroid labels (`frame,x,y`). Full-frame
   resizing preserves aspect ratio and measurements are mapped back to the source.
   Synthetic disturbances are disabled in video mode. The error KPI now displays
   labeled centroid error; without labels it displays N/A. Source FPS metadata is
   preserved, while evaluation timing uses the specified fixed 30 Hz clock.
9. Pause/Resume retains the session without adding paused time to source duration
   or FPS. Reset simulation switches back to Mode A and starts a fresh audit log.
10. Generate ISRO Audit at any point. Per-frame CSV automatically streams to disk;
    the export button adds a consistent CSV snapshot, summary CSV, JSON provenance,
    and PDF. The status bar displays the PDF path. Reports are stored in
    `%USERPROFILE%\Documents\FSOCTracker\audits` unless `--output` is specified.

## Reading the telemetry

Green PASS means the current metric meets the displayed threshold. Amber FAIL is
an actual measured failure, not a software exception. N/A means no valid measurement
exists yet. Acquisition measures confirmed visual detection, while pointing error
measures centering. The report includes the initial transient; it does not remove
startup error to improve reported results. Centroid RMSE only uses labeled detections;
read it alongside lock retention and detection coverage.

The perspective view displays world context for demonstration. The camera marker is
illustratively elevated above the planar field; it is not a satellite orbital model.
The actual tracking sensor is monochrome and the control loop uses image detections.

## Troubleshooting

- **No model:** expected until trained beacon ONNX weights are supplied. Classical
  detection remains functional and the active backend is always displayed.
- **Cannot open video:** verify the local MP4 plays, is not empty and uses an OpenCV
  supported codec. Decoding failure before the advertised final frame reports an error.
- **Ground-truth error:** use UTF-8 CSV, zero-based integer frame indices, unique
  rows, finite x/y and original-video bounds. Do not provide boresight offsets as labels.
- **Weather failure:** check key activation, city name, allowance and connectivity.
  The request has an eight-second socket timeout and runs outside the UI thread.
- **Low FPS:** close competing CPU work, compare headless throughput and check the
  model. Playback is not allowed to skip evaluation frames to inflate speed.
- **Large pointing error:** reduce target speed, jitter, or fog; increase maximum
  slew within 5-10 deg/s; reacquire or designate the correct target.
- **Shutdown:** the window waits for capture, export and weather workers to finish;
  it never force-terminates a worker while it owns files or Qt resources.

Logs rotate at 2 MB (three backups). API keys are not recorded in audit metadata.
Per-frame CSV remains available even if a later export fails.
