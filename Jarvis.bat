@echo off
rem ===========================================================================
rem  Jarvis.bat — fastest way to start JARVIS by hand (double-click me).
rem  JARVIS starts asleep: clap twice, then say "wake up".
rem ===========================================================================
setlocal
cd /d "%~dp0"
call "%~dp0_env.bat"

if not defined PY (
    echo.
    echo  [X] Python was not found on this PC.
    echo      Install Python 3.11+ from https://www.python.org/downloads/
    echo      and tick "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)

echo.
echo  Starting JARVIS...
echo  Clap twice, then say "wake up".
echo.
start "JARVIS" "%PYW%" "%~dp0main.py"
timeout /t 2 /nobreak >nul
exit /b 0
