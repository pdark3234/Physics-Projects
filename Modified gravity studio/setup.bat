@echo off
setlocal enableextensions
title Modified Gravity Studio Setup

cd /d "%~dp0"

set "PY_CMD="
where python >nul 2>nul && set "PY_CMD=python"
if not defined PY_CMD (
    where py >nul 2>nul && set "PY_CMD=py -3"
)

if not defined PY_CMD (
    echo [ERROR] Python 3 was not found in PATH.
    echo Install Python 3.11+ and rerun this script.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo.
        pause
        exit /b 1
    )
)

echo [INFO] Activating virtual environment...
call ".venv\Scripts\activate.bat"

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    echo.
    pause
    exit /b 1
)

echo.
echo [INFO] Setup complete.
echo [INFO] Start the app with: run_new_user.bat
echo.
pause
exit /b 0
