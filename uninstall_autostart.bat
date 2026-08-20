@echo off
rem ===========================================================================
rem  uninstall_autostart.bat — stop the Sentinel and remove it from startup.
rem  JARVIS itself is untouched; you can still launch it with Jarvis.bat.
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0_env.bat"

set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "LNK=%STARTUP%\JARVIS Sentinel.lnk"

echo.
echo  Removing JARVIS Sentinel auto-start...

if defined PY (
    "%PY%" "%~dp0sentinel.py" --stop
)

if exist "%LNK%" (
    del /f /q "%LNK%"
    echo  [OK] Startup shortcut removed.
) else (
    echo  [i] No startup shortcut was installed.
)

echo.
echo  Done. JARVIS will no longer start itself.
echo  Re-enable any time with install_autostart.bat
echo.
pause
exit /b 0
