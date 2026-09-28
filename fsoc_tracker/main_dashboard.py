"""PyQt6 mission control. Widgets stay on the GUI thread; workers own their data."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from typing import Any

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QCloseEvent,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .audit_exporter import AuditLogger
from .disturbances import DisturbanceConfig
from .math3d_engine import Frame, Trajectory, cubic_bspline
from .pipeline import FrameResult, RunConfig, TrackingPipeline, VideoSource
from .scene_view import render_scene

LOG = logging.getLogger(__name__)
STYLE = """
QMainWindow, QDialog { background: #090e13; }
QWidget { color: #c9d6df; font-family: 'Segoe UI'; font-size: 12px; }
QGroupBox { background: #101820; border: 1px solid #24303c; border-radius: 6px;
 margin-top: 16px; padding: 12px 8px 8px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; color: #758b9b; }
QPushButton { background: #182733; border: 1px solid #304653; border-radius: 4px; padding: 7px 12px; }
QPushButton:hover { border-color: #59d5e3; background: #203742; }
QPushButton:disabled { color: #536471; border-color: #26323c; }
QPushButton#primary { background: #173e45; color: #8eeaf0; border-color: #327c88; font-weight: 600; }
QLineEdit, QComboBox, QSpinBox { background: #0b1219; border: 1px solid #30414d; padding: 5px; border-radius: 3px; }
QComboBox QAbstractItemView { background: #14212b; color: #c9d6df; selection-background-color: #245568; }
QSlider::groove:horizontal { height: 4px; background: #273642; border-radius: 2px; }
QSlider::handle:horizontal { background: #58cad7; width: 12px; margin: -5px 0; border-radius: 6px; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #47606d; background: #111b24; }
QCheckBox::indicator:checked { background: #56cbd7; }
QScrollArea { border: none; background: transparent; }
QStatusBar { background: #101820; color: #8ca0ae; }
"""


class MissionWorker(QThread):
    packet_ready = pyqtSignal(object)
    status = pyqtSignal(str)
    failed = pyqtSignal(str)
    audit_ready = pyqtSignal(object)

    def __init__(
        self,
        config: RunConfig,
        output: Path,
        video_path: Path | None = None,
        custom_points: np.ndarray | None = None,
        truth_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.output = output
        self.video_path = video_path
        self.custom_points = custom_points
        self.truth_path = truth_path
        self.commands: Queue[tuple[str, Any]] = Queue()
        self.display_pending = Event()
        self.wake = Event()
        self.logger: AuditLogger | None = None

    def submit(self, name: str, value: Any = None) -> None:
        self.commands.put((name, value))
        self.wake.set()

    def stop(self) -> None:
        self.requestInterruption()
        self.wake.set()

    def run(self) -> None:
        pipeline: TrackingPipeline | None = None
        source: VideoSource | None = None
        paused = False
        try:
            cv2.setNumThreads(2)
            source = VideoSource(self.video_path, self.truth_path) if self.video_path is not None else None
            pipeline = TrackingPipeline(self.config, source)
            if pipeline.world is not None and self.custom_points is not None:
                pipeline.world.set_trajectory(self.config.trajectory, self.custom_points)
            metadata = asdict(self.config)
            metadata["detector"] = pipeline.detector.status
            metadata["video"] = str(self.video_path) if self.video_path else None
            if source:
                metadata["reported_video_fps"] = source.reported_fps
                metadata["ground_truth"] = str(source.truth_path) if source.truth_path else None
            self.logger = AuditLogger(self.output, metadata)
            self.audit_ready.emit(self.logger)
            self.status.emit(pipeline.detector.status)
            while not self.isInterruptionRequested():
                self.wake.clear()
                while True:
                    try:
                        name, value = self.commands.get_nowait()
                    except Empty:
                        break
                    if name == "pause":
                        paused = bool(value)
                    elif name == "config":
                        pipeline.config = value
                        self.logger.event(pipeline.frame_index / 30, asdict(value))
                    elif name == "designate":
                        pipeline.designate(np.asarray(value, dtype=float))
                        self.logger.event(pipeline.frame_index / 30, {"designation_sensor": list(value)})
                if paused:
                    self.wake.wait(0.1)
                    continue
                started = time.perf_counter()
                try:
                    result = pipeline.step()
                except EOFError:
                    self.status.emit(
                        "BENCHMARK COMPLETE - all decoded frames logged. Generate ISRO Audit to export."
                    )
                    break
                scene = None
                if not self.display_pending.is_set() and pipeline.world is not None:
                    scene = render_scene(pipeline.world, list(pipeline.tracker.history))
                remaining = 1 / 30 - (time.perf_counter() - started)
                if remaining > 0:
                    # A stop/config command can wake this wait without touching worker-owned state.
                    self.wake.wait(remaining)
                result.row["wall_dt_s"] = time.perf_counter() - started
                self.logger.append(result.row)
                if not self.display_pending.is_set():
                    self.display_pending.set()
                    self.packet_ready.emit(
                        {"result": result, "scene": scene, "summary": self.logger.summary()}
                    )
        except Exception as exc:
            LOG.exception("Mission worker stopped")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if pipeline is not None:
                pipeline.close()
            elif source is not None:
                source.close()
            if self.logger is not None:
                self.logger.close()


class ExportWorker(QThread):
    exported = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, logger: AuditLogger) -> None:
        super().__init__()
        self.logger = logger

    def run(self) -> None:
        try:
            self.exported.emit(self.logger.export())
        except Exception as exc:
            LOG.exception("Audit export failed")
            self.failed.emit(f"Export failed: {exc}")


class WeatherWorker(QThread):
    weather_ready = pyqtSignal(float, float, str)
    failed = pyqtSignal(str)

    def __init__(self, city: str, api_key: str) -> None:
        super().__init__()
        self.city, self.api_key = city, api_key

    def run(self) -> None:
        query = urllib.parse.urlencode({"q": self.city, "appid": self.api_key, "units": "metric"})
        try:
            request = urllib.request.Request(
                "https://api.openweathermap.org/data/2.5/weather?" + query,
                headers={"User-Agent": "FSOCTracker/1.0"},
            )
            with urllib.request.urlopen(request, timeout=8) as response:
                data = json.load(response)
            wind = float(data.get("wind", {}).get("speed", 0))
            visibility = float(data.get("visibility", 10000))
            jitter = float(np.clip(wind * 1.5, 0, 20))
            fog = float(np.clip(1 - visibility / 10000, 0, 1))
            self.weather_ready.emit(
                jitter,
                fog,
                f"{self.city}: wind {wind:.1f} m/s, visibility {visibility:.0f} m (heuristic mapping)",
            )
        except urllib.error.HTTPError as exc:
            self.failed.emit(f"Weather HTTP {exc.code}. Check city, key activation and API allowance.")
        except (OSError, ValueError, KeyError, TypeError):
            # urllib exceptions can contain credential-bearing URLs; never surface them.
            self.failed.emit("Weather unavailable. Check network connectivity and try again.")


class ImagePanel(QLabel):
    clicked = pyqtSignal(float, float)

    def __init__(self, text: str, width: int, height: int) -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(width, height)
        self.setStyleSheet("background:#060b10; border:1px solid #263541; color:#627888;")
        self.sensor_clicks = False
        self.image_size = (width, height)
        self.display_rect = (0.0, 0.0, float(width), float(height))

    def show_frame(self, array: Frame) -> None:
        height, width = array.shape[:2]
        fmt = QImage.Format.Format_Grayscale8 if array.ndim == 2 else QImage.Format.Format_BGR888
        contiguous = np.ascontiguousarray(array)
        image = QImage(contiguous.tobytes(), width, height, contiguous.strides[0], fmt).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.image_size = (width, height)
        self.display_rect = (
            (self.width() - pixmap.width()) / 2,
            (self.height() - pixmap.height()) / 2,
            float(pixmap.width()),
            float(pixmap.height()),
        )
        self.setPixmap(pixmap)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and self.sensor_clicks and event.button() == Qt.MouseButton.LeftButton:
            x, y, width, height = self.display_rect
            p = event.position()
            if x <= p.x() < x + width and y <= p.y() < y + height:
                self.clicked.emit(
                    (p.x() - x) * self.image_size[0] / width, (p.y() - y) * self.image_size[1] / height
                )
        super().mousePressEvent(event)


class DropZone(QLabel):
    video_dropped = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__(
            "DROP EVALUATOR .MP4 HERE\nFull frame input | 30 Hz source clock | optional matching .csv"
        )
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(66)
        self.setStyleSheet("border:1px dashed #3b7885; background:#112129; color:#82c9d4; padding:8px;")

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        mime = event.mimeData() if event is not None else None
        if event is not None and mime is not None:
            urls = mime.urls()
            if (
                len(urls) == 1
                and urls[0].isLocalFile()
                and Path(urls[0].toLocalFile()).suffix.lower() == ".mp4"
            ):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent | None) -> None:
        mime = event.mimeData() if event is not None else None
        if event is not None and mime is not None:
            urls = mime.urls()
            if (
                len(urls) == 1
                and urls[0].isLocalFile()
                and Path(urls[0].toLocalFile()).suffix.lower() == ".mp4"
            ):
                self.video_dropped.emit(urls[0].toLocalFile())
                event.acceptProposedAction()


class PathCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(520, 520)
        self.points: list[tuple[float, float]] = []

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.points.append((event.position().x(), event.position().y()))
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.buttons() & Qt.MouseButton.LeftButton:
            p = (event.position().x(), event.position().y())
            if not self.points or np.linalg.norm(np.array(p) - self.points[-1]) > 18:
                self.points.append(p)
                self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b131a"))
        painter.setPen(QPen(QColor("#213340"), 1))
        for value in range(0, 521, 52):
            painter.drawLine(value, 0, value, 520)
            painter.drawLine(0, value, 520, value)
        painter.setPen(QPen(QColor("#ecbd65"), 2))
        for x, y in self.points:
            painter.drawEllipse(QPointF(x, y), 3, 3)
        if len(self.points) >= 4:
            curve = cubic_bspline(np.asarray(self.points))
            painter.setPen(QPen(QColor("#60d5df"), 2))
            for a, b in pairwise(curve):
                painter.drawLine(QPointF(*a), QPointF(*b))
        painter.end()


class PathDialog(QDialog):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Draw a custom B-spline trajectory")
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Click at least four control points, or drag. The beacon traverses the curve and returns.")
        )
        self.canvas = PathCanvas()
        layout.addWidget(self.canvas)
        clear = QPushButton("Clear path")
        clear.clicked.connect(self.clear)
        layout.addWidget(clear)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def clear(self) -> None:
        self.canvas.points.clear()
        self.canvas.update()

    def validate(self) -> None:
        if len(self.canvas.points) < 4:
            QMessageBox.information(self, "Path needs control points", "Draw or click at least four points.")
        else:
            self.accept()


class MissionControl(QMainWindow):
    def __init__(
        self,
        output_directory: Path,
        model_path: Path | None = None,
        *,
        initial_config: RunConfig | None = None,
        video_path: Path | None = None,
        truth_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.output_directory = output_directory
        self.initial_config = initial_config or RunConfig(model_path=model_path)
        self.model_path = model_path or self.initial_config.model_path
        self.worker: MissionWorker
        self.logger: AuditLogger | None = None
        self.export_worker: ExportWorker | None = None
        self.weather_worker: WeatherWorker | None = None
        self.last_packet: dict[str, Any] | None = None
        self.pending_start: tuple[RunConfig, Path | None] | None = None
        self.custom_points: np.ndarray | None = None
        self.closing = False
        self.mode = "simulation"
        self.sliders: dict[str, QSlider] = {}
        self.kpis: dict[str, QLabel] = {}
        self.history: deque[tuple[float, float, float, float]] = deque(maxlen=450)
        self.setWindowTitle("RADIX FSOC Simulator - ISRO 26169")
        self.setStyleSheet(STYLE)
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.resize(1460, 960)
        self._build_ui()
        config = self.initial_config
        self.trajectory.setCurrentText(config.trajectory.value)
        self.sliders["speed"].setValue(round(config.speed))
        self.slew.setValue(round(config.slew_deg_s))
        self.decoys.setValue(config.decoys)
        self.occlude.setChecked(config.occluded)
        disturbance = config.disturbances
        for key, value in {
            "jitter": disturbance.jitter,
            "fog": disturbance.fog * 100,
            "salt": disturbance.salt_pepper * 100,
            "gaussian": disturbance.gaussian_sigma,
            "scintillation": disturbance.scintillation * 100,
        }.items():
            self.sliders[key].setValue(round(value))
        self.poisson.setChecked(disturbance.poisson)
        self._start(
            replace(config, mode="video" if video_path else "simulation", model_path=self.model_path),
            video_path,
            truth_path,
        )

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 16, 20, 10)
        header = QHBoxLayout()
        title = QLabel("RADIX FSOC SIMULATOR")
        title.setStyleSheet("font-size:21px; font-weight:600; color:#e8f3f8; letter-spacing:2px;")
        header.addWidget(title)
        header.addStretch()
        self.mode_label = QLabel("MODE A  /  LIVE CLOSED LOOP")
        self.mode_label.setStyleSheet("color:#68d3df; font-weight:600;")
        header.addWidget(self.mode_label)
        outer.addLayout(header)
        subtitle = QLabel(
            "COARSE ALIGNMENT LABORATORY     /     2000 x 2000 WORLD     /     640 x 480 FOCAL PLANE     /     PS 26169"
        )
        subtitle.setStyleSheet("color:#657e8f; font-size:10px; padding-bottom:8px;")
        outer.addWidget(subtitle)
        metrics = QHBoxLayout()
        for key, heading in [
            ("error", "TRACKING ERROR / <= 10 px"),
            ("retention", "LOCK RETENTION / > 95%"),
            ("acquisition", "ACQUISITION / <= 2.0 s"),
            ("fps", "THROUGHPUT / >= 20 FPS"),
        ]:
            box = QGroupBox(heading)
            layout = QVBoxLayout(box)
            value = QLabel("--")
            value.setStyleSheet("font-size:26px; font-weight:600; color:#8195a4;")
            layout.addWidget(value)
            self.kpis[key] = value
            metrics.addWidget(box)
        outer.addLayout(metrics)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 6, 0)
        sensor_box = QGroupBox("01 / SENSOR FEED - CLICK A BEACON TO DESIGNATE PRIORITY")
        sensor_layout = QVBoxLayout(sensor_box)
        self.sensor = ImagePanel("Initializing sensor pipeline...", 640, 480)
        self.sensor.setFixedSize(640, 480)
        self.sensor.sensor_clicks = True
        self.sensor.clicked.connect(lambda x, y: self.worker.submit("designate", (x, y)))
        sensor_layout.addWidget(self.sensor, alignment=Qt.AlignmentFlag.AlignCenter)
        self.state_label = QLabel("SEARCHING   |   waiting for observations")
        self.state_label.setStyleSheet("color:#e6b963; padding:4px;")
        sensor_layout.addWidget(self.state_label)
        ll.addWidget(sensor_box)
        self.error_plot = self._plot("TRACKING ERROR CONVERGENCE", "px")
        self.error_curve = self.error_plot.plot(pen=pg.mkPen("#64d5e0", width=1.5))
        self.error_plot.addItem(
            pg.InfiniteLine(pos=10, angle=0, pen=pg.mkPen("#b78c4b", style=Qt.PenStyle.DashLine))
        )
        ll.addWidget(self.error_plot)
        self.velocity_plot = self._plot("MOTOR VELOCITY / PAN - CYAN / TILT - AMBER", "px/s")
        self.pan_curve = self.velocity_plot.plot(pen=pg.mkPen("#64d5e0", width=1.5))
        self.tilt_curve = self.velocity_plot.plot(pen=pg.mkPen("#e6b963", width=1.5))
        ll.addWidget(self.velocity_plot)
        splitter.addWidget(left)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        top = QHBoxLayout()
        debug_box = QGroupBox("02 / VISION DEBUGGER")
        dl = QVBoxLayout(debug_box)
        self.debug = ImagePanel("Binary / Otsu feed", 240, 180)
        self.debug.setFixedSize(240, 180)
        dl.addWidget(self.debug, alignment=Qt.AlignmentFlag.AlignCenter)
        top.addWidget(debug_box)
        scenario_box = QGroupBox("03 / SCENARIO")
        sl = QVBoxLayout(scenario_box)
        self.trajectory = QComboBox()
        self.trajectory.addItems([t.value for t in Trajectory])
        self.trajectory.currentTextChanged.connect(self._trajectory_changed)
        sl.addWidget(self.trajectory)
        draw = QPushButton("Draw B-spline path")
        draw.clicked.connect(self._draw_path)
        sl.addWidget(draw)
        self.decoys = QSpinBox()
        self.decoys.setRange(0, 5)
        self.decoys.setPrefix("Decoys: ")
        self.decoys.valueChanged.connect(self._controls_changed)
        sl.addWidget(self.decoys)
        self.slew = QSpinBox()
        self.slew.setRange(5, 10)
        self.slew.setValue(10)
        self.slew.setSuffix(" deg/s max slew")
        self.slew.valueChanged.connect(self._controls_changed)
        sl.addWidget(self.slew)
        self.occlude = QCheckBox("Occlude priority beacon")
        self.occlude.toggled.connect(self._controls_changed)
        sl.addWidget(self.occlude)
        top.addWidget(scenario_box)
        rl.addLayout(top)
        controls = QGroupBox("04 / DISTURBANCE CONTROL DECK")
        cl = QGridLayout(controls)
        for i, (name, label, maximum, initial, unit) in enumerate(
            [
                ("jitter", "Platform jitter", 20, 0, " px"),
                ("fog", "Fog / contrast loss", 100, 0, "%"),
                ("salt", "Salt & pepper", 10, 0, "%"),
                ("speed", "Target speed", 150, 25, " px/s"),
                ("gaussian", "Gaussian sigma", 20, 0, " DN"),
                ("scintillation", "Scintillation", 100, 0, "%"),
            ]
        ):
            label_widget = QLabel(label)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, maximum)
            slider.setValue(initial)
            readout = QLabel(f"{initial}{unit}")
            readout.setMinimumWidth(62)
            slider.valueChanged.connect(lambda v, out=readout, suffix=unit: out.setText(f"{v}{suffix}"))
            slider.valueChanged.connect(self._controls_changed)
            cl.addWidget(label_widget, i, 0)
            cl.addWidget(slider, i, 1)
            cl.addWidget(readout, i, 2)
            self.sliders[name] = slider
        self.poisson = QCheckBox("Poisson shot noise")
        self.poisson.toggled.connect(self._controls_changed)
        cl.addWidget(self.poisson, 6, 0, 1, 3)
        presets = QHBoxLayout()
        for name in ["Clear Sky", "High Jitter", "Dense Fog"]:
            button = QPushButton(name)
            button.clicked.connect(lambda checked=False, preset=name: self._preset(preset))
            presets.addWidget(button)
        cl.addLayout(presets, 7, 0, 1, 3)
        rl.addWidget(controls)
        weather_box = QGroupBox("05 / LIVE WEATHER - OPENWEATHERMAP")
        wl = QHBoxLayout(weather_box)
        self.city = QLineEdit("Bengaluru,IN")
        self.city.setPlaceholderText("City,country")
        self.api_key = QLineEdit(os.environ.get("OPENWEATHERMAP_API_KEY", ""))
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("API key (not saved)")
        self.weather_button = QPushButton("Fetch weather")
        self.weather_button.clicked.connect(self._fetch_weather)
        wl.addWidget(self.city)
        wl.addWidget(self.api_key)
        wl.addWidget(self.weather_button)
        rl.addWidget(weather_box)
        self.dropzone = DropZone()
        self.dropzone.video_dropped.connect(self.load_video)
        rl.addWidget(self.dropzone)
        scene_box = QGroupBox("06 / WORLD CONTEXT - OPENCV PERSPECTIVE PROJECTION")
        scene_layout = QVBoxLayout(scene_box)
        self.scene = ImagePanel("Perspective scene", 420, 210)
        self.scene.setMaximumHeight(260)
        scene_layout.addWidget(self.scene)
        rl.addWidget(scene_box, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(right)
        scroll.setMinimumWidth(500)
        splitter.addWidget(scroll)
        splitter.setSizes([690, 660])
        outer.addWidget(splitter, 1)
        actions = QHBoxLayout()
        self.pause_button = QPushButton("Pause")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._pause)
        self.reset_button = QPushButton("Reset simulation")
        self.reset_button.clicked.connect(self.reset_simulation)
        browse = QPushButton("Open benchmark video")
        browse.clicked.connect(self._browse)
        self.export_button = QPushButton("Generate ISRO Audit")
        self.export_button.setObjectName("primary")
        self.export_button.clicked.connect(self._export)
        self.export_button.setEnabled(False)
        for button in [self.pause_button, self.reset_button, browse]:
            actions.addWidget(button)
        actions.addStretch()
        actions.addWidget(QLabel("30 Hz SOURCE CLOCK  /  SEE AUDIT FOR MEASURED PERFORMANCE"))
        actions.addWidget(self.export_button)
        outer.addLayout(actions)
        self.setCentralWidget(root)
        self.status_bar.showMessage("Initializing mission...")

    @staticmethod
    def _plot(title: str, units: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(background="#0d151d")
        plot.setTitle(title, color="#859aa9", size="9pt")
        plot.setLabel("left", units, color="#728693")
        plot.setLabel("bottom", "Mission time", units="s", color="#728693")
        plot.showGrid(x=True, y=True, alpha=0.13)
        plot.setMinimumHeight(115)
        plot.setMaximumHeight(155)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        return plot

    def _config(self) -> RunConfig:
        values = {key: slider.value() for key, slider in self.sliders.items()}
        return RunConfig(
            trajectory=Trajectory(self.trajectory.currentText()),
            speed=float(values["speed"]),
            seed=self.initial_config.seed,
            slew_deg_s=float(self.slew.value()),
            decoys=self.decoys.value(),
            occluded=self.occlude.isChecked(),
            model_path=self.model_path,
            disturbances=DisturbanceConfig(
                jitter=float(values["jitter"]),
                fog=values["fog"] / 100,
                salt_pepper=values["salt"] / 100,
                gaussian_sigma=float(values["gaussian"]),
                poisson=self.poisson.isChecked(),
                scintillation=values["scintillation"] / 100,
            ),
        )

    def _controls_changed(self, *_args: Any) -> None:
        if hasattr(self, "worker") and self.mode == "simulation":
            self.worker.submit("config", self._config())

    def _start(self, config: RunConfig, video: Path | None, truth_path: Path | None = None) -> None:
        self.mode = config.mode
        self.history.clear()
        self.last_packet = None
        self.logger = None
        self.export_button.setEnabled(False)
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(False)
        self.pause_button.setText("Pause")
        self.pause_button.blockSignals(False)
        self.mode_label.setText(
            "MODE A  /  LIVE CLOSED LOOP" if video is None else "MODE B  /  VIDEO BENCHMARK"
        )
        for widget in [
            *self.sliders.values(),
            self.trajectory,
            self.decoys,
            self.slew,
            self.occlude,
            self.poisson,
        ]:
            widget.setEnabled(video is None)
        self.weather_button.setEnabled(
            video is None and not (self.weather_worker is not None and self.weather_worker.isRunning())
        )
        self.worker = MissionWorker(config, self.output_directory, video, self.custom_points, truth_path)
        self.worker.packet_ready.connect(self._packet)
        self.worker.status.connect(self.status_bar.showMessage)
        self.worker.failed.connect(self._error)
        self.worker.audit_ready.connect(self._audit_ready)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _request_start(self, config: RunConfig, video: Path | None = None) -> None:
        if self.worker.isRunning():
            self.pending_start = (config, video)
            self.worker.stop()
            self.status_bar.showMessage("Finishing current run and releasing its resources...")
        else:
            self._start(config, video)

    def _worker_finished(self) -> None:
        if self.pending_start is not None and not self.closing:
            config, video = self.pending_start
            self.pending_start = None
            self._start(config, video)

    def _audit_ready(self, logger: AuditLogger) -> None:
        self.logger = logger
        self.export_button.setEnabled(True)

    def _packet(self, packet: dict[str, Any]) -> None:
        self.last_packet = packet
        result: FrameResult = packet["result"]
        row, summary = result.row, packet["summary"]
        self.sensor.show_frame(result.image)
        self.debug.show_frame(result.binary)
        if packet["scene"] is not None:
            self.scene.show_frame(packet["scene"])
        elif self.mode == "video":
            self.scene.clear()
            self.scene.setText(
                "BENCHMARK BYPASS\nWorld and gimbal disabled\nCentroid RMSE uses original-pixel ground truth"
            )
        error = row["pointing_error_px"] if self.mode == "simulation" else row["centroid_error_px"]
        self._kpi("error", error, "px", error is not None and error <= 10)
        retention = summary["lock_retention_pct"]
        self._kpi("retention", retention, "%", retention is not None and retention > 95)
        acquisition = summary["acquisition_s"]
        self._kpi("acquisition", acquisition, "s", acquisition is not None and acquisition <= 2)
        fps = summary["average_fps"]
        self._kpi("fps", fps, "FPS", fps is not None and fps >= 20)
        rmse = summary["centroid_rmse_px"]
        rmse_text = f"{rmse:.3f} px" if rmse is not None else "N/A (no labeled observations)"
        reacq = summary["max_reacquisition_s"]
        reacq_text = f"{reacq:.2f}s" if reacq is not None else "--"
        self.state_label.setText(
            f"{row['state']}  |  {result.backend}  |  Centroid RMSE: {rmse_text}  |  Reacq: {reacq_text}"
        )
        self.history.append(
            (row["time_s"], error if error is not None else float("nan"), row["pan_px_s"], row["tilt_px_s"])
        )
        data = np.array(self.history)
        self.error_curve.setData(data[:, 0], data[:, 1], connect="finite")
        self.pan_curve.setData(data[:, 0], data[:, 2])
        self.tilt_curve.setData(data[:, 0], data[:, 3])
        self.worker.display_pending.clear()

    def _kpi(self, key: str, value: float | None, unit: str, passed: bool) -> None:
        color = "#79d6a0" if passed else "#e8b46a" if value is not None else "#7a8e9c"
        text = f"{value:.2f} {unit}  {'PASS' if passed else 'FAIL'}" if value is not None else "N/A"
        self.kpis[key].setText(text)
        self.kpis[key].setStyleSheet(f"font-size:23px; font-weight:600; color:{color};")

    def _pause(self, checked: bool) -> None:
        self.pause_button.setText("Resume" if checked else "Pause")
        self.worker.submit("pause", checked)

    def reset_simulation(self) -> None:
        self._request_start(self._config())

    def _trajectory_changed(self, _text: str) -> None:
        if hasattr(self, "worker"):
            self.reset_simulation()

    def _draw_path(self) -> None:
        dialog = PathDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.custom_points = np.asarray(dialog.canvas.points) / 520 * 1960 + 20
            self.trajectory.blockSignals(True)
            self.trajectory.setCurrentText(Trajectory.CUSTOM.value)
            self.trajectory.blockSignals(False)
            self.reset_simulation()

    def _preset(self, name: str) -> None:
        presets = {
            "Clear Sky": (0, 0, 0, 0, 0),
            "High Jitter": (20, 5, 2, 3, 10),
            "Dense Fog": (3, 85, 5, 8, 30),
        }
        for key, value in zip(["jitter", "fog", "salt", "gaussian", "scintillation"], presets[name]):
            self.sliders[key].setValue(value)

    def _browse(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Open Benchmark-2 video", "", "MP4 video (*.mp4)")
        if filename:
            self.load_video(filename)

    def load_video(self, filename: str) -> None:
        path = Path(filename)
        if not path.is_file() or path.suffix.lower() != ".mp4":
            self._error("Select an existing .mp4 video file.")
            return
        self._request_start(replace(self._config(), mode="video"), path)

    def _fetch_weather(self) -> None:
        if (
            self.closing
            or self.mode != "simulation"
            or (self.weather_worker is not None and self.weather_worker.isRunning())
        ):
            return
        if not self.api_key.text().strip() or not self.city.text().strip():
            self._error("Enter your OpenWeatherMap API key and city, or set OPENWEATHERMAP_API_KEY.")
            return
        self.weather_button.setEnabled(False)
        self.weather_worker = WeatherWorker(self.city.text().strip(), self.api_key.text().strip())
        self.weather_worker.weather_ready.connect(self._weather_received)
        self.weather_worker.failed.connect(self._error)
        self.weather_worker.finished.connect(
            lambda: self.weather_button.setEnabled(self.mode == "simulation" and not self.closing)
        )
        self.weather_worker.start()

    def _weather_received(self, jitter: float, fog: float, message: str) -> None:
        if self.mode == "simulation" and not self.closing:
            self.sliders["jitter"].setValue(round(jitter))
            self.sliders["fog"].setValue(round(fog * 100))
            self.status_bar.showMessage(message)

    def _export(self) -> None:
        if self.logger is None or (self.export_worker is not None and self.export_worker.isRunning()):
            return
        self.export_button.setEnabled(False)
        self.export_worker = ExportWorker(self.logger)
        self.export_worker.exported.connect(self._exported)
        self.export_worker.failed.connect(self._error)
        self.export_worker.finished.connect(lambda: self.export_button.setEnabled(self.logger is not None))
        self.export_worker.start()

    def _exported(self, paths: list[Path]) -> None:
        self.status_bar.showMessage(f"Audit saved: {next(p for p in paths if p.suffix == '.pdf')}")

    def _error(self, message: str) -> None:
        self.status_bar.showMessage(message)
        if not self.closing:
            QMessageBox.warning(self, "Mission control", message)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        if event is None:
            return
        self.closing = True
        self.pending_start = None
        self.worker.stop()
        busy = self.worker.isRunning() or any(
            w is not None and w.isRunning() for w in [self.export_worker, self.weather_worker]
        )
        if busy:
            event.ignore()
            self.status_bar.showMessage("Closing: waiting for workers to release capture and files...")
            QTimer.singleShot(100, self.close)
        else:
            event.accept()


def launch(
    output: Path,
    model_path: Path | None = None,
    smoke_seconds: float = 0,
    screenshot: Path | None = None,
    *,
    initial_config: RunConfig | None = None,
    video_path: Path | None = None,
    truth_path: Path | None = None,
) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("FSOCTracker")
    window = MissionControl(
        output, model_path, initial_config=initial_config, video_path=video_path, truth_path=truth_path
    )
    window.show()
    if smoke_seconds > 0:

        def finish_smoke() -> None:
            if screenshot is not None:
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                window.grab().save(str(screenshot))
            window.close()

        QTimer.singleShot(round(smoke_seconds * 1000), finish_smoke)
    return app.exec()
