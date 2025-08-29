@echo off
echo MIB Viewer - Windows Setup Test
echo ================================
echo.

REM Test Python installation
echo [1/5] Testing Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo   FAIL: Python not found
    goto :error
) else (
    python --version
    echo   PASS: Python installed
)
echo.

REM Test package imports
echo [2/5] Testing core packages...
python -c "import PyQt5; print('   PASS: PyQt5 available')" 2>nul || echo   FAIL: PyQt5 missing
python -c "import pyqtgraph; print('   PASS: pyqtgraph available')" 2>nul || echo   FAIL: pyqtgraph missing  
python -c "import numpy; print('   PASS: numpy available')" 2>nul || echo   FAIL: numpy missing
python -c "import h5py; print('   PASS: h5py available')" 2>nul || echo   FAIL: h5py missing
echo.

REM Test file structure
echo [3/5] Testing file structure...
if exist "src\mib_viewer\main_pyqtgraph.py" (
    echo   PASS: PyQtGraph main found
) else (
    echo   FAIL: src\mib_viewer\main_pyqtgraph.py missing
)
if exist "pyproject.toml" (
    echo   PASS: pyproject.toml found
) else (
    echo   FAIL: pyproject.toml missing  
)
echo.

REM Test GUI startup (dry run)
echo [4/5] Testing GUI imports...
python -c "from src.mib_viewer.gui.mib_viewer_pyqtgraph import MibViewerPyQtGraph; print('   PASS: GUI classes importable')" 2>nul || echo   FAIL: GUI import issues
echo.

REM Test data loading capability
echo [5/5] Testing data loading...
python -c "from src.mib_viewer.gui.mib_viewer_qt import load_mib; print('   PASS: MIB loading functions available')" 2>nul || echo   FAIL: MIB loading issues
echo.

echo ================================
echo Windows setup test COMPLETE
echo.
echo If any tests failed, try:
echo   1. pip install -r requirements.txt
echo   2. Check WINDOWS_INSTALL.md for details
echo.
pause
goto :end

:error
echo.
echo Setup test FAILED - check Python installation
echo See WINDOWS_INSTALL.md for detailed instructions
pause

:end