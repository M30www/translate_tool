@echo off
REM ---------------------------------------------------------------
REM  OCR probe launcher.
REM
REM  *** KEEP THIS FILE PURE ASCII AND MINIMAL. ***
REM
REM  Why: cmd.exe re-reads a .bat file by byte offset. If the file
REM  contains non-ASCII text AND changes the code page with "chcp",
REM  the next read resumes in the middle of a multibyte character.
REM  Line boundaries then shift, the "REM" prefix gets swallowed, and
REM  fragments of comments are executed as commands. That produced
REM  errors like:  'preprocess' is not recognized as a command.
REM
REM  So: launcher only, pure ASCII. All real content lives in Python.
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

".venv\Scripts\python.exe" ocr_probe.py %*

REM Keep the window open on failure so a double-click shows the error.
if errorlevel 1 pause
