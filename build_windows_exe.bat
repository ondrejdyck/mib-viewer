@echo off
echo Building MIB Viewer Windows Executable
echo =======================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python from python.org
    pause
    exit /b 1
)

REM Install dependencies directly with pip (avoids uv PyQt5 wheel issue)
echo Installing dependencies with pip (bypassing uv PyQt5 issue)...
pip install PyQt5 pyqtgraph numpy matplotlib h5py emdfile tqdm py4dstem pyinstaller
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Build the executable
echo.
echo Building standalone executable...
python build_exe.py
if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

REM Check if executable was created
if exist "dist\MibViewer.exe" (
    echo.
    echo SUCCESS: MibViewer.exe created!
    echo Location: dist\MibViewer.exe
    echo.
    for %%A in ("dist\MibViewer.exe") do echo Size: %%~zA bytes
    echo.
    echo You can now copy MibViewer.exe to any Windows machine
    echo No installation required - just run the .exe file
) else (
    echo ERROR: MibViewer.exe not found in dist folder
    echo Check the build output above for errors
)

echo.
pause