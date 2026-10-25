@echo off
REM ---------------------------------------------------------------
REM  One-click offline translation test (translate_log.py).
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

REM Default: newest log, free engine (no API key needed), first 20 lines only
REM so a double-click finishes in well under a minute.
REM Passing any argument bypasses DEFAULT entirely.
set "DEFAULT=--limit 20"

echo ===============================================================
echo   Offline translation test   (no API key needed)
echo   Result will be written to:  logs\translated_^<timestamp^>.md
echo ===============================================================
echo.

if "%~1"=="" (
    ".venv\Scripts\python.exe" translate_log.py %DEFAULT%
) else (
    ".venv\Scripts\python.exe" translate_log.py %*
)
if errorlevel 1 goto :failed

REM Open the newest result file so a double-click shows something useful.
set "LATEST="
for /f "delims=" %%f in ('dir /b /o-d "logs\translated_*.md" 2^>nul') do (
    if not defined LATEST set "LATEST=logs\%%f"
)
if defined LATEST (
    echo.
    echo [OK] opening %LATEST%
    start "" "%LATEST%"
)
echo.
echo Done. Press any key to close.
pause >nul
exit /b 0

:failed
echo.
echo [FAILED] see the error above.
pause
exit /b 1
