@echo off
REM ---------------------------------------------------------------
REM  Environment setup launcher.
REM
REM  *** KEEP THIS FILE PURE ASCII AND MINIMAL. ***  See the note in
REM  run_probe.bat for why mixing "chcp" with non-ASCII text breaks
REM  cmd.exe batch parsing.
REM
REM  All setup logic lives in setup_env.py.
REM ---------------------------------------------------------------

chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

REM Locate a usable interpreter: prefer "python", fall back to "py -3".
set "PY=python"
python -c "import sys" >nul 2>nul
if errorlevel 1 set "PY=py -3"

%PY% -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No usable Python found on PATH.
    echo         Install Python 3.9 - 3.12 and tick "Add python.exe to PATH".
    echo         https://www.python.org/downloads/
    pause
    exit /b 1
)

%PY% setup_env.py
if errorlevel 1 pause
