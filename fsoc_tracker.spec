# -*- mode: python ; coding: utf-8 -*-
"""One-file Windows build; optional models are included only when present."""
from pathlib import Path
import os
import sys
import PyQt6
from PyInstaller.utils.hooks import collect_dynamic_libs

root = Path(SPECPATH)
qt_bin = Path(PyQt6.__file__).resolve().parent / 'Qt6' / 'bin'
# Isolate DLL discovery from unrelated native tools injected into the host PATH.
# In particular, Poppler's version-suffixed ICU is incompatible with Qt's use of
# the Windows system ICU API, despite having the same icuuc.dll filename.
if sys.platform == 'win32':
    windows = Path(os.environ['SystemRoot'])
    os.environ['PATH'] = os.pathsep.join(map(str, [qt_bin, Path(sys.base_prefix),
        Path(sys.base_prefix) / 'DLLs', windows / 'System32', windows]))
datas = [(str(root / 'models' / 'README.md'), 'models')]
datas += [(str(path), 'models') for path in (root / 'models').glob('*.onnx')]
a = Analysis(
    [str(root / 'run_tracker.py')], pathex=[str(root)],
    binaries=collect_dynamic_libs('onnxruntime'), datas=datas,
    hiddenimports=['pyqtgraph', 'onnxruntime', 'reportlab.pdfbase._fontdata'],
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'PyQt5', 'PySide2', 'PySide6', 'IPython', 'pytest'],
    noarchive=False,
)
# A build host's PATH can contain mismatched MSVC runtime DLLs.
# Qt requires matching CRT exports. Normalize all collected copies and
# the extraction-root copies to the runtime shipped with this PyQt6 installation.
crt = {p.name.lower(): p for pattern in ('msvcp140*.dll', 'vcruntime140*.dll')
       for p in qt_bin.glob(pattern)}
a.binaries = [(dest, str(crt[Path(dest).name.lower()]), kind)
              if Path(dest).name.lower() in crt else (dest, source, kind)
              for dest, source, kind in a.binaries]
root_names = {dest.lower() for dest, _, _ in a.binaries}
a.binaries += [(path.name, str(path), 'BINARY') for name, path in crt.items()
               if name not in root_names]
# Qt on modern Windows uses the OS ICU unless its wheel supplies its own build.
# Filter cached analysis records too, so incremental builds cannot retain a
# third-party ICU selected by a previous, contaminated PATH.
system_icu = {'icuuc.dll', 'icuin.dll', 'icudt.dll'}
a.binaries = [entry for entry in a.binaries
              if Path(entry[0]).name.lower() not in system_icu
              or (qt_bin / Path(entry[0]).name).is_file()]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [], name='FSOCTracker',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
    version=str(root / 'version_info.txt'),
)
