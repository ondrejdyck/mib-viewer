@echo off
echo MIB EELS Viewer (PyQtGraph) - Starting...
echo Version: Alpha with 4D STEM support
echo.

REM Change to the directory where this batch file is located
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check if uv is installed
uv --version >nul 2>&1
if errorlevel 1 (
    echo Installing uv package manager...
    pip install uv
    if errorlevel 1 (
        echo ERROR: Failed to install uv
        pause
        exit /b 1
    )
)

REM Install dependencies with fallback strategies
if not exist ".venv" (
    echo Installing dependencies (this may take a few minutes)...
    echo Trying uv sync...
    uv sync
    if errorlevel 1 (
        echo WARNING: uv sync failed, trying fallback methods...
        echo.
        echo Fallback 1: Trying uv pip install...
        uv pip install -e .
        if errorlevel 1 (
            echo Fallback 2: Trying direct pip install...
            pip install -r requirements.txt
            if errorlevel 1 (
                echo ERROR: All dependency installation methods failed
                echo Try manually: pip install PyQt5 pyqtgraph numpy matplotlib h5py
                pause
                exit /b 1
            )
        )
    )
)

REM Run the application with fallback
echo Launching MIB Viewer (PyQtGraph version)...
echo.

REM Try PyQtGraph version first
if exist "src\mib_viewer\main_pyqtgraph.py" (
    echo Running PyQtGraph version with 4D STEM support...
    uv run python src/mib_viewer/main_pyqtgraph.py
    if errorlevel 1 (
        echo uv run failed, trying direct python...
        python src/mib_viewer/main_pyqtgraph.py
    )
) else if exist "src\mib_viewer\main.py" (
    echo Fallback: Running original Qt version...
    uv run python src/mib_viewer/main.py
    if errorlevel 1 (
        echo uv run failed, trying direct python...
        python src/mib_viewer/main.py
    )
) else (
    echo ERROR: No main.py found in src/mib_viewer/
    echo Expected: src/mib_viewer/main_pyqtgraph.py or src/mib_viewer/main.py
    pause
    exit /b 1
)

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Application exited with an error
    pause
)