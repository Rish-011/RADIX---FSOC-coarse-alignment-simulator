"""Incremental telemetry with snapshot-safe, independently named audit exports."""

from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class AuditLogger:
    def __init__(self, directory: Path, metadata: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.directory = directory
        self.session = f"FSOC_{stamp}"
        self.metadata = metadata
        self.path = directory / f"{self.session}_live.csv"
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer: csv.DictWriter[str] | None = None
        self.lock = RLock()
        self.count = 0
        self.duration = self.wall_duration = self.processing_total = 0.0
        self.detected = self.locked = self.after_acquisition = self.retained = 0
        self.point_sum = self.point_sq = self.point_peak = 0.0
        self.point_count = self.centroid_count = 0
        self.centroid_sq = self.centroid_peak = 0.0
        self.acquisition: float | None = None
        self.reacquisition: float | None = None
        self.closed = False

    def append(self, row: dict[str, Any]) -> None:
        with self.lock:
            if self.closed:
                raise RuntimeError("Audit log is closed.")
            if self.writer is None:
                self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
            self.count += 1
            self.duration += float(row["dt_s"])
            self.wall_duration += float(row["wall_dt_s"])
            self.processing_total += float(row["processing_ms"])
            self.detected += bool(row["detected"])
            self.locked += bool(row["locked"])
            if self.acquisition is None and row["locked"]:
                self.acquisition = float(row["time_s"])
            if self.acquisition is not None:
                self.after_acquisition += 1
                self.retained += bool(row["locked"])
            if row.get("reacquisition_s") is not None:
                value = float(row["reacquisition_s"])
                self.reacquisition = max(self.reacquisition or 0.0, value)
            point = row.get("pointing_error_px")
            if point is not None and math.isfinite(point):
                self.point_count += 1
                self.point_sum += point
                self.point_sq += point**2
                self.point_peak = max(self.point_peak, point)
            centroid = row.get("centroid_error_px")
            if centroid is not None and math.isfinite(centroid):
                self.centroid_count += 1
                self.centroid_sq += centroid**2
                self.centroid_peak = max(self.centroid_peak, centroid)
            if self.count % 30 == 0:
                self.handle.flush()

    def event(self, time_s: float, config: dict[str, Any]) -> None:
        with self.lock:
            self.metadata.setdefault("events", []).append({"time_s": time_s, "config": config})

    def summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "mode": self.metadata.get("mode", "unknown"),
                "frames": self.count,
                "duration_s": self.duration,
                "average_fps": self.count / self.wall_duration if self.wall_duration else None,
                "processing_fps": 1000 * self.count / self.processing_total
                if self.processing_total
                else None,
                "mean_processing_ms": self.processing_total / self.count if self.count else None,
                "acquisition_s": self.acquisition,
                "max_reacquisition_s": self.reacquisition,
                "average_pointing_error_px": self.point_sum / self.point_count if self.point_count else None,
                "peak_pointing_error_px": self.point_peak if self.point_count else None,
                "pointing_rmse_px": math.sqrt(self.point_sq / self.point_count) if self.point_count else None,
                "centroid_rmse_px": math.sqrt(self.centroid_sq / self.centroid_count)
                if self.centroid_count
                else None,
                "peak_centroid_error_px": self.centroid_peak if self.centroid_count else None,
                "ground_truth_matched_frames": self.centroid_count,
                "detection_rate_pct": 100 * self.detected / self.count if self.count else None,
                "lock_retention_pct": 100 * self.retained / self.after_acquisition
                if self.after_acquisition
                else None,
                "whole_run_lock_pct": 100 * self.locked / self.count if self.count else None,
            }

    def export(self) -> list[Path]:
        stamp = datetime.now(timezone.utc).strftime("%H%M%S_%f")
        stem = self.directory / f"{self.session}_audit_{stamp}"
        raw_path = Path(f"{stem}_frames.csv")
        summary_path = Path(f"{stem}_summary.csv")
        metadata_path = Path(f"{stem}_config.json")
        pdf_path = Path(f"{stem}.pdf")
        with self.lock:
            if not self.closed:
                self.handle.flush()
            summary = self.summary()
            metadata = json.loads(json.dumps(self.metadata, default=str))
            shutil.copyfile(self.path, raw_path)
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._pdf(pdf_path, summary, metadata)
        return [raw_path, summary_path, metadata_path, pdf_path]

    @staticmethod
    def _pdf(path: Path, summary: dict[str, Any], metadata: dict[str, Any]) -> None:
        styles = getSampleStyleSheet()
        document = SimpleDocTemplate(
            str(path),
            pagesize=(8.27 * inch, 11.69 * inch),
            rightMargin=42,
            leftMargin=42,
            topMargin=40,
            bottomMargin=40,
        )
        story: list[Any] = [
            Paragraph("FSOC | Coarse Alignment Audit", styles["Title"]),
            Paragraph("ISRO Problem Statement 26169 - measured software run", styles["Normal"]),
            Spacer(1, 14),
        ]
        rows: list[list[Any]] = [["Metric", "Measured value"]]
        for name, value in summary.items():
            formatted = "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
            rows.append([name.replace("_", " "), formatted])
        table = Table(rows, colWidths=[330, 175], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#13232d")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f5f7"), colors.white]),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.extend(
            [table, Spacer(1, 16), Paragraph("Metric definitions and interpretation", styles["Heading2"])]
        )
        notes = [
            "Acquisition: first three consecutive associated observations. Reacquisition: loss onset to confirmed lock.",
            "Lock retention: confirmed visual lock frames after first acquisition / all frames after acquisition. Prediction-only frames count as lost.",
            "Centroid RMSE: sqrt(mean(dx^2 + dy^2)) over frames with both accepted detection and ground truth. Missing observations are reported by detection rate, not assigned zero error.",
            "Simulation centroid and pointing errors use sensor pixels, including injected jitter. Benchmark centroid error uses original video pixels; video pointing error is observed displacement from the 640 x 480 boresight, not centroid accuracy.",
            "Average FPS includes processing and pacing but excludes operator pauses. Processing FPS excludes pacing. Source time advances by 1/30 s per frame.",
            "Targets: pointing error <= 10 px; retention > 95%; acquisition <= 2 s; reacquisition <= 1 s; throughput >= 20 FPS. These are acceptance criteria, not guaranteed outcomes.",
            "No ground truth means N/A centroid RMSE. This report is a software measurement, not an ISRO certification.",
            f"Configuration seed: {metadata.get('seed', 'N/A')}. Mode: {metadata.get('mode', 'N/A')}. See companion JSON for configuration changes and detector provenance.",
        ]
        for note in notes:
            story.extend([Paragraph(escape(note), styles["BodyText"]), Spacer(1, 5)])
        document.build(story)

    def close(self) -> None:
        with self.lock:
            if not self.closed:
                self.handle.flush()
                self.handle.close()
                self.closed = True
