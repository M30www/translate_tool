@echo off
REM ---------------------------------------------------------------
REM  Semi-realtime OCR probe launcher (probe_live.py).
REM
REM  *** KEEP THIS FILE PURE ASCII AND MINIMAL. ***
REM  Reason: cmd.exe re-reads a .bat by byte offset; non-ASCII text
REM  plus "chcp" shifts line boundaries and comments get executed.
REM  All real logic lives in Python.
REM ---------------------------------------------------------------

chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found: .venv\Scripts\python.exe
    echo         Please run setup_ocr.bat first.
    pause
    exit /b 1
)

if "%~1"=="" (
    ".venv\Scripts\python.exe" probe_live.py --select
) else (
    ".venv\Scripts\python.exe" probe_live.py %*
)

REM Keep the window open on failure so a double-click shows the error.
if errorlevel 1 pause
