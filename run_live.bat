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

REM Default arguments used when launched with no arguments (double-click).
REM   --select      interactively pick the chat panel region
REM   --min-score   drop low-confidence garbage lines (misreads score 0.53-0.72,
REM                 real chat lines score >=0.88). See docs/experiments.md E7.
REM Change DEFAULT below if you want different defaults; note that passing ANY
REM argument bypasses DEFAULT entirely (then you must pass everything yourself).
set "DEFAULT=--select --min-score 0.75"

if "%~1"=="" (
    ".venv\Scripts\python.exe" probe_live.py %DEFAULT%
) else (
    ".venv\Scripts\python.exe" probe_live.py %*
)

REM Keep the window open on failure so a double-click shows the error.
if errorlevel 1 pause
