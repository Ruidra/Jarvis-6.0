@echo off
rem ===========================================================================
rem  Sentinel.bat — run the clap listener in the foreground (with live logs).
rem  Clap twice + say "wake up" and it will start JARVIS for you.
rem  Press Ctrl+C to stop.  Use install_autostart.bat for the silent version.
rem ===========================================================================
setlocal
cd /d "%~dp0"
call "%~dp0_env.bat"

if not defined PY (
    echo  [X] Python not found. Install Python 3.11+ and re-run.
    pause
    exit /b 1
)

echo.
echo  ================================================
echo    JARVIS SENTINEL - always-on clap listener
echo  ================================================
echo    Clap twice, then say "wake up".
echo    JARVIS will start on its own.
echo    Ctrl+C to stop.
echo.
"%PY%" "%~dp0sentinel.py"
echo.
pause
exit /b 0
