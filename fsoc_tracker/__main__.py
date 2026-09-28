"""Desktop entrypoint and repeatable headless benchmark CLI."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .audit_exporter import AuditLogger
from .math3d_engine import Trajectory
from .pipeline import RunConfig, TrackingPipeline, VideoSource


def main() -> int:
    parser = argparse.ArgumentParser(description="FSOC coarse alignment mission control")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, default=Path.home() / "Documents" / "FSOCTracker" / "audits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=25.0)
    parser.add_argument(
        "--trajectory", choices=[t.value for t in Trajectory], default=Trajectory.STRAIGHT.value
    )
    parser.add_argument("--realtime", action="store_true", help="Pace headless runs to 30 Hz.")
    parser.add_argument("--smoke-seconds", type=float, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--screenshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.frames <= 0 or not 0 <= args.speed <= 150:
        parser.error("frames must be positive; speed must be between 0 and 150 px/s.")
    args.output.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        args.output / "fsoc.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO, handlers=[handler], format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    model = args.model or bundle / "models" / "yolov8n_fsoc.onnx"
    config = RunConfig(
        mode="video" if args.video else "simulation",
        trajectory=Trajectory(args.trajectory),
        speed=args.speed,
        seed=args.seed,
        model_path=model,
    )
    if args.truth and not args.video:
        parser.error("--truth requires --video.")
    if not args.headless:
        from .main_dashboard import launch

        return launch(
            args.output,
            model,
            args.smoke_seconds,
            args.screenshot,
            initial_config=config,
            video_path=args.video,
            truth_path=args.truth,
        )
    pipeline: TrackingPipeline | None = None
    logger: AuditLogger | None = None
    try:
        source = VideoSource(args.video, args.truth) if args.video else None
        pipeline = TrackingPipeline(config, source)
        metadata = asdict(config)
        metadata["video"] = str(args.video) if args.video else None
        metadata["detector"] = pipeline.detector.status
        if source:
            metadata["ground_truth"] = str(source.truth_path) if source.truth_path else None
            metadata["reported_video_fps"] = source.reported_fps
        logger = AuditLogger(args.output, metadata)
        for _ in range(args.frames):
            started = time.perf_counter()
            try:
                result = pipeline.step()
            except EOFError:
                break
            if args.realtime:
                time.sleep(max(0, 1 / 30 - (time.perf_counter() - started)))
            result.row["wall_dt_s"] = time.perf_counter() - started
            logger.append(result.row)
        paths = logger.export()
        output = {"summary": logger.summary(), "artifacts": [str(path) for path in paths]}
        if sys.stdout is not None:
            print(json.dumps(output, indent=2))
        return 0
    except Exception as exc:
        logging.getLogger(__name__).exception("Headless run failed")
        if sys.stderr is not None:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
