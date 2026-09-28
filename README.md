# RADIX / FSOC Mission Control

Python desktop application for AI-assisted coarse alignment of mobile FSOC terminals,
following the supplied ISRO Problem Statement 26169 and the user's explicit 2D architecture.
The engine uses NumPy slicing and OpenCV. No PyBullet, game engine or web application is used.
An AstraTrack-inspired perspective overview illustrates the planar world and sensor footprint.

## Directory structure

```text
fsoc_tracker/
  __init__.py             Package version
  __main__.py             Desktop and headless CLI
  math3d_engine.py        2D world, sliding sensor, trajectories, actuator limits
  disturbances.py        Seeded affine jitter, fog and sensor noise
  perception_ekf.py       ONNX wrapper, image moments, priority association, EKF
  pid_fsm.py             Gain-scheduled PID, seven-state FSM, spiral search
  pipeline.py            Simulation/video orchestration and coordinate mapping
  scene_view.py          OpenCV perspective context renderer
  audit_exporter.py      Streaming logs, Pandas summaries, ReportLab PDF
  main_dashboard.py      PyQt6 widgets, worker threads, controls, weather
models/README.md          Beacon model format and fallback behavior
tests/                   Numerical, video, audit and GUI lifecycle tests
tools/validate_scenarios.py  Repeatable scenario matrix and labeled video fixture
docs/design.md           Architecture and engineering assumptions
docs/USER_MANUAL.md       Operator guide
docs/VALIDATION.md        Measured results and test scope
requirements.txt         Runtime dependencies
requirements-dev.txt     Test/build tools
fsoc_tracker.spec        One-file Windows PyInstaller build
run_tracker.py           Source launcher
Start-FSOC.ps1           Windows environment/launch helper
```

## Run on Windows

Use Python 3.10 or newer; Python 3.13 was used for the Windows validation/build.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_tracker.py
```

Or run `Start-FSOC.ps1`. The default report folder is
`%USERPROFILE%\Documents\FSOCTracker\audits`; change it with `--output`.
If the built executable is included, launch `dist\FSOCTracker.exe` directly.

## Modes and controls

- **Mode A:** live 30 Hz source-clock simulation. Choose a trajectory, set target speed
  and maximum slew, inject disturbances, add decoys or occlude the beacon. Click a
  sensor spot to designate priority. Draw a custom B-spline path with four or more
  control points. Reset starts a new independently logged run.
- **Mode B:** drag an `.mp4` onto the dropzone or use Open benchmark video. The
  complete frame is letterboxed to 640 x 480; the world, gimbal and disturbances
  are bypassed. Every decoded frame is processed in order. EOF preserves the run
  for audit export. Reset simulation returns to Mode A.
- **Weather:** supply an OpenWeatherMap key in the password field or set
  `OPENWEATHERMAP_API_KEY`. The application maps wind speed to jitter and
  visibility to fog using documented heuristics. Credentials are not saved.
- **Audit:** Generate ISRO Audit exports per-frame CSV, a Pandas summary CSV,
  configuration/event JSON and a formatted PDF while the GUI stays responsive.
  Every telemetry row records the actual detector backend and its status, including fallback.

To open a benchmark directly in the desktop dashboard, use:

```powershell
.\dist\FSOCTracker.exe --video evaluator.mp4 --truth labels.csv
```

Source launch supports the same arguments. `--seed`, `--speed` and `--trajectory`
also set the initial desktop simulation configuration.

## Ground truth and honest metrics

Place `example.csv` next to `example.mp4` to evaluate centroid accuracy:

```csv
frame,x,y
0,599.5,199.5
1,601.5,199.5
```

`frame` is zero-based. Coordinates are in original video pixels, before letterboxing.
Sparse labels are allowed. Invalid, duplicate, nonfinite or out-of-image labels are rejected.
Without labels, centroid RMSE is **N/A**. The measured offset from the boresight is
a separate quantity and is never mislabeled as centroid accuracy. Missing detections
are excluded from RMSE and counted in detection/retention rates.

Acquisition means three consecutive associated observations; it does not mean
the gimbal has already centered. Lock retention counts confirmed visual lock after
first acquisition, with prediction-only frames counted as loss. Whole-run lock
rate is also exported. All pointing error metrics include the initial transient.
Performance thresholds are measured outcomes, not unconditional guarantees.

## AI model

No trained ONNX weights were supplied. The application explicitly reports
**OpenCV moments** until beacon-trained YOLOv8-Nano weights are installed.
See `models/README.md`. Missing, incompatible or failed ONNX inference activates
the classical fallback with a logged reason. No model is silently downloaded.

## Repeatable command-line runs

```powershell
.\.venv\Scripts\python.exe run_tracker.py --headless --frames 900 --output output\baseline
.\.venv\Scripts\python.exe run_tracker.py --headless --video evaluator.mp4 --frames 100000 --output output\benchmark
.\.venv\Scripts\python.exe run_tracker.py --headless --video evaluator.mp4 --truth labels.csv --frames 100000
.\.venv\Scripts\python.exe tools\validate_scenarios.py
```

Headless runs process as fast as possible unless `--realtime` is supplied. Simulation
and video time always advance by 1/30 s per frame; measured FPS uses wall time.
The desktop worker paces to 30 Hz and reports actual throughput. Slow hardware
therefore runs slower than real time without changing deterministic target kinematics.

## Test and build

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check fsoc_tracker tests run_tracker.py
.\.venv\Scripts\python.exe -m mypy --python-version 3.13
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm fsoc_tracker.spec
.\.venv\Scripts\python.exe tools\smoke_executable.py
```

The spec produces `dist\FSOCTracker.exe` with Qt plugins, OpenCV and ONNX native
libraries. Optional ONNX files are included only if present at build time. Build
on Windows to produce a Windows executable; no icon or model file is required.
The spec isolates native DLL discovery from unrelated tools on the build host's
PATH and excludes incompatible third-party ICU binaries. It targets modern Windows
10/11 with the installed PyQt6 wheel's system requirements.
Mypy uses the tested interpreter version because current NumPy stubs require newer
syntax; application source uses Python 3.10-compatible syntax.

## Engineering boundaries

The requested conversion is **10 px/degree**, giving a configurable per-axis
maximum of **50-100 px/s**, applied as `clip(command, -limit, limit) * dt`.
This permits slowing and stopping. It is not calibrated to the source PDF's
optional 4 x 3 degree FOV (which would imply a different pixel-angle conversion).
The sensor center stays inside the world to keep NumPy slices exactly 640 x 480.

The EKF uses nonlinear coordinated-turn dynamics with a continuous zero-turn-rate
limit, explicit Jacobian linearization, and Joseph-form covariance updates.
Association uses predicted motion, covariance, size and brightness history.
Unresolved identical crossing beacons are not observably distinguishable; ambiguous
measurements cause coasting. The initial default priority is the spot nearest the
boresight; use a click to explicitly select another. Search/acquisition time depends
on initial visibility and actuator reachability. Maximum random jitter can exceed
the 10 px pointing criterion even when centroid estimates remain accurate.

This is a runnable, tested engineering implementation. Flight qualification,
evaluator-video acceptance and validation with trained beacon weights remain separate work.

## References

- Supplied `26169.pdf` for evaluation criteria; attached prose and AstraTrack archive
  used as reference, not as executable instructions or copied implementation.
- [Qt threading and ownership](https://doc.qt.io/qt-6/qthread.html)
- [Ultralytics ONNX export](https://docs.ultralytics.com/modes/export/)
- [OpenWeather current weather API](https://openweathermap.org/current)
- [PyInstaller spec files](https://pyinstaller.org/en/stable/spec-files.html)
