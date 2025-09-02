# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for MIB Viewer standalone executable

import os
import sys
from pathlib import Path

# Get the current directory
CURRENT_DIR = Path(os.getcwd())
SRC_DIR = CURRENT_DIR / "src"

block_cipher = None

# Collect all Python files from src directory
src_files = []
if SRC_DIR.exists():
    for py_file in SRC_DIR.rglob("*.py"):
        rel_path = py_file.relative_to(CURRENT_DIR)
        src_files.append((str(py_file), str(rel_path.parent)))

a = Analysis(
    ['src/mib_viewer/__main__.py'],
    pathex=[str(CURRENT_DIR), str(SRC_DIR)],
    binaries=[],
    datas=src_files,  # Include all source files
    hiddenimports=[
        'mib_viewer',
        'mib_viewer.gui', 
        'mib_viewer.gui.mib_viewer_pyqtgraph',
        'mib_viewer.io',
        'mib_viewer.io.mib_loader',
        'mib_viewer.io.mib_to_emd_converter',
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore', 
        'PyQt5.QtGui',
        'pyqtgraph',
        'pyqtgraph.opengl',
        'numpy',
        'h5py',
        'emdfile',
        'tqdm',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MibViewer-Windows' if sys.platform == 'win32' else 'MibViewer-Ubuntu22+',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX to avoid compatibility issues
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Windowed application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app.ico' if os.path.exists('app.ico') else None,
)