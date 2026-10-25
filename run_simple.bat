@echo off
REM ---------------------------------------------------------------
REM  Launcher for explore\ocr\ocr_simple.py
REM
REM  *** KEEP THIS FILE PURE ASCII AND MINIMAL. ***
REM
REM  Why: cmd.exe re-reads a .bat file by byte offset. If the file
REM  contains non-ASCII text AND changes the code page mid-file with
REM  "chcp", the next read resumes in the middle of a multibyte
REM  character. Line boundaries then shift, the "REM" prefix gets
REM  swallowed, and fragments of comments get executed as commands.
REM  That is a real bug we already hit once.
REM
REM  So: launcher only. All real content lives in the Python script.
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
    echo Usage: run_simple.bat [image-file or folder]
    echo    e.g. run_simple.bat shots\
    echo         run_simple.bat shots\chat1.png
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "explore\ocr\ocr_simple.py" %*

REM Always pause: this script exists to print results, so a window
REM opened by a double-click must not close before they can be read.
pause
