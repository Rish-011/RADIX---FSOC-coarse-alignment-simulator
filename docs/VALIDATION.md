# Validation record

Validated locally on Windows 11 x64 with Python 3.13.12. Application source also
passes Python 3.10 grammar parsing; other Python/OS combinations were not runtime-tested.
Exact build dependencies are recorded in `requirements-lock-windows-py313.txt`.

## Automated checks

- `python -m pytest -q`: **29 passed**, including GUI worker lifecycle, video-mode switching,
  desktop CLI arguments, explicit video ground truth, detector provenance and reset during weather fetching.
- `python -m ruff check fsoc_tracker tests tools run_tracker.py`: **passed**.
- `python -m mypy --python-version 3.13`: **passed**, 10 application modules.
- All package modules parse under `ast.parse(..., feature_version=(3,10))`.
- A real ONNX Runtime inference test uses a synthetic Constant graph with YOLOv8's
  `[1,5,8400]` output layout to exercise letterboxing, NMS and centroid refinement.
  This checks the wrapper contract; it does **not** measure trained-model accuracy.
- Missing/corrupt model fallback, poisoned label rejection paths, source-pixel mapping,
  video EOF, rate clamps, bounded trajectories, all seven FSM states, EKF covariance,
  crossing decoys, 10% impulse noise, maximum jitter and occlusion are covered by tests.

## Deterministic scenario measurements

Each simulation row is a 360-frame / 12-second run, seed 42, target speed 25 px/s,
maximum slew 10 deg/s. Trajectory runs use two decoys. Headless FPS is unpaced
throughput on this machine and includes competing system load; it is not a minimum
hardware guarantee. The dashboard separately measures its paced throughput.

| Scenario | Lock retention | Centroid RMSE, px | Pointing RMSE after 3 s, px | Headless FPS |
|---|---:|---:|---:|---:|
| Straight line | 100.00% | 0.0000 | 2.677 | 76.1 |
| Circular | 100.00% | 0.2833 | 2.380 | 109.6 |
| Figure of 8 | 100.00% | 0.0000 | 1.153 | 116.5 |
| Random walk | 100.00% | 0.0000 | 3.430 | 113.3 |
| Custom B-spline | 100.00% | 0.0000 | 3.098 | 97.2 |
| 10% salt-and-pepper | 100.00% | 0.1281 | 2.666 | 94.6 |
| Maximum jitter (+/-20 px) | 99.16% | 0.1729 | **16.724** | 85.3 |
| Fog 85%, Gaussian sigma 8, impulses 5% | 100.00% | 0.2256 | 2.757 | 24.3 |
| 0.5-second occlusion | 95.25% | 0.0000 | 2.663 | 104.4 |
| Annotated 1280 x 720 MP4, 180 frames | 100.00% | 0.0036 | N/A | 85.2 |

Maximum jitter fails the 10 px pointing threshold despite accurate centroiding;
independent random per-frame displacement cannot be removed by the rate-limited
gimbal. The occlusion run reacquires confirmed lock in 0.60 s from loss onset.
The full audit includes initial alignment error, while this table explicitly
separates the post-3-second region for convergence analysis. No startup frames are
removed from the exported report metrics.

The default 900-frame run recorded 0.10 s detection acquisition, 100% retention,
8.714 px average pointing error, 89.669 px peak pointing error and 19.948 px full-run
pointing RMSE. The initially off-center target therefore does not meet the 10 px
criterion for every frame. Detection acquisition and physical alignment are distinct.

Run `python tools/validate_scenarios.py` to reproduce the matrix and create the
annotated MP4/CSV fixture. Raw logs and `results.json` go under `output/scenarios/`.

## Visual and packaging checks

The dashboard screenshot was inspected for sensor, boresight, target brackets,
EKF trail, KPI readability, charts, controls and perspective context. The right
deck scrolls vertically on smaller screens. Sensor data remain 640 x 480 regardless
of desktop DPI scaling. The packaged application's generated PDF was rendered and visually checked as a single page, with the
metric table and definitions fitting without clipping.

The Windows PyInstaller one-file build includes Qt runtime/plugins, OpenCV,
NumPy, Pandas, ReportLab and ONNX Runtime. Optional pyqtgraph OpenGL and Pandas Jinja2
hook warnings are unrelated to used features (no OpenGL view or styled HTML tables).

## Remaining validation scope

- Trained beacon weights, evaluator-provided videos and independently labeled
  real sensor data were unavailable. No AI accuracy or evaluator acceptance claim
  is made beyond the tested wrapper and synthetic data.
- Live OpenWeatherMap success was not exercised because no API key was supplied.
  Network access runs in a separate worker and errors omit credential-bearing URLs.
- Identical unresolved beacons can be ambiguous. History-based association cannot
  guarantee identity where the images contain no distinguishing information.
- The perspective display is an illustration of the requested 2D world, not a
  physical orbit, optical propagation or calibrated gimbal model.

The first packaged GUI smoke test exposed a build-host DLL collision: PyInstaller
collected Poppler's version-suffixed ICU library instead of relying on the Windows
ICU API expected by Qt. Binary import/export inspection confirmed the missing
`ucnv_open` and related symbols. The final spec isolates PATH discovery, normalizes
Qt CRT copies, and removes incompatible cached ICU entries. Package startup must
be checked with `python tools/smoke_executable.py` after every dependency change.

Final packaged executable smoke: **PASS**. The standalone executable completed a
90-frame headless run with PDF/CSV export, rendered the GUI and sensor frames,
saved its screenshot, opened the annotated MP4 directly with explicit ground-truth
labels, and exited cleanly. All three packaged run logs contained no errors.
The final checks are stored under `output/resume-package/`.

Run `python tools/package_source.py` after final verification to refresh the complete
code listing, source ZIP and SHA-256 manifest under `output/delivery/`.
