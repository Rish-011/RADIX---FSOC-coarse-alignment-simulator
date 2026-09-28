"""Build a complete readable codebook, source ZIP and SHA-256 delivery manifest."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    output = root / "output" / "delivery"
    output.mkdir(parents=True, exist_ok=True)
    folders = ("fsoc_tracker", "tests", "tools", "docs", "models", "examples")
    files = sorted(
        path
        for folder in folders
        for path in (root / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".onnx"}
    )
    files += [
        root / name
        for name in (
            "README.md",
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-lock-windows-py313.txt",
            "pyproject.toml",
            "fsoc_tracker.spec",
            "run_tracker.py",
            "Start-FSOC.ps1",
            ".gitignore",
        )
    ]
    priority = [
        root / "fsoc_tracker" / name
        for name in (
            "math3d_engine.py",
            "disturbances.py",
            "perception_ekf.py",
            "pid_fsm.py",
            "main_dashboard.py",
            "pipeline.py",
            "audit_exporter.py",
            "scene_view.py",
            "__main__.py",
            "__init__.py",
        )
    ]
    code = priority + [
        path
        for path in files
        if path not in priority and path.suffix in {".py", ".spec", ".ps1", ".toml", ".txt"}
    ]
    tree = "\n".join(path.relative_to(root).as_posix() for path in sorted(files))
    content = ["# Complete FSOC Tracker source\n\n## Directory structure\n\n```text\n" + tree + "\n```\n"]
    for path in code:
        language = {".py": "python", ".spec": "python", ".ps1": "powershell", ".toml": "toml"}.get(
            path.suffix, "text"
        )
        content.append(
            f"\n## {path.relative_to(root).as_posix()}\n\n```{language}\n"
            + path.read_text(encoding="utf-8").rstrip()
            + "\n```\n"
        )
    book = output / "COMPLETE_SOURCE.md"
    book.write_text("".join(content), encoding="utf-8")
    archive = output / "FSOCTracker-source.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, "FSOCTracker/" + path.relative_to(root).as_posix())
        bundle.write(book, "FSOCTracker/COMPLETE_SOURCE.md")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed.")
        for path in files:
            if bundle.read("FSOCTracker/" + path.relative_to(root).as_posix()) != path.read_bytes():
                raise RuntimeError(f"Archive differs from source: {path}")
    artifacts = [archive, book]
    executable = root / "dist" / "FSOCTracker.exe"
    if executable.is_file():
        artifacts.append(executable)
    manifest = {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in artifacts
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Packaged {len(files)} files and {len(code)} complete source/configuration listings.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
