@echo off
REM ---------------------------------------------------------------
REM  Realtime OCR + translation (probe_live.py --translate).
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

REM Defaults for double-click:
REM   --select           pick the chat panel region
REM   --min-score 0.75   drop low-confidence garbage (real chat lines score >=0.88)
REM   --translate        translate new lines in a background thread
REM   --threshold 0.004  a bit less sensitive than the 0.002 default: in a real match
REM                      the game world keeps moving and only ~17% of OCR runs produced
REM                      new lines (E11). Watch the recall in the summary: a new chat
REM                      line only changes ~0.3-1% of the region, so do NOT go much higher.
REM Engine defaults to "free" (no API key). Use --engine llm after setting DS_LLM_API_KEY.
set "DEFAULT=--select --min-score 0.75 --translate --threshold 0.004"

if "%~1"=="" (
    ".venv\Scripts\python.exe" probe_live.py %DEFAULT%
) else (
    ".venv\Scripts\python.exe" probe_live.py %*
)

REM Keep the window open so you can read the summary after Ctrl+C or an error.
echo.
pause
