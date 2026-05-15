@echo off
setlocal enableextensions
title Modified Gravity Studio

cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "MGS_VERBOSE=false"
set "MGS_SYMBOLIC_LOGS=false"

echo.
echo =========================================
echo   Modified Gravity Studio
echo   http://localhost:5000
echo =========================================
echo.

set "PY_CMD="
where python >nul 2>nul && set "PY_CMD=python"
if not defined PY_CMD (
    where py >nul 2>nul && set "PY_CMD=py -3"
)

if not defined PY_CMD (
    echo [ERROR] Python 3 was not found in PATH.
    echo Install Python or activate your existing environment and try again.
    echo.
    pause
    exit /b 1
)

echo [INFO] Using %PY_CMD%
echo [INFO] Starting server...
echo.

%PY_CMD% run.py
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Modified Gravity Studio exited with code %EXIT_CODE%.
) else (
    echo [INFO] Modified Gravity Studio stopped.
)
echo.
pause
exit /b %EXIT_CODE%
