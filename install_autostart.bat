@echo off
rem ===========================================================================
rem  install_autostart.bat — ONE CLICK SETUP.
rem  Registers the JARVIS Sentinel to start silently every time you log in.
rem  After this you never launch JARVIS again:  clap clap + "wake up".
rem  Remove it any time with uninstall_autostart.bat.
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0_env.bat"

set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "LNK=%STARTUP%\JARVIS Sentinel.lnk"
set "VBS=%~dp0Sentinel-Silent.vbs"

echo.
echo  ==========================================================
echo    JARVIS  -  AUTO START INSTALLER
echo  ==========================================================
echo.

if not defined PY (
    echo  [X] Python was not found on this PC.
    echo      Install Python 3.11+ from https://www.python.org/downloads/
    echo      then run this installer again.
    echo.
    pause
    exit /b 1
)
echo  Python      : %PY%
echo  Sentinel    : %VBS%
echo  Startup dir : %STARTUP%
echo.

if not exist "%VBS%" (
    echo  [X] Sentinel-Silent.vbs is missing next to this installer.
    pause
    exit /b 1
)

rem ── 1. Check the audio dependency is present ──────────────────────────────
"%PY%" -c "import sounddevice" >nul 2>&1
if errorlevel 1 (
    echo  Installing the microphone library ^(sounddevice^)...
    "%PY%" -m pip install --quiet sounddevice numpy
)

rem ── 2. Offer exact wake-phrase recognition ────────────────────────────────
"%PY%" -c "import vosk" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Optional: Vosk gives EXACT "wake up" recognition instead of
    echo  "any speech after the clap". It is a ~40 MB one-time download.
    choice /c YN /m "  Install Vosk now (recommended)"
    if errorlevel 2 goto :skipvosk
    "%PY%" -m pip install vosk
)
"%PY%" "%~dp0sentinel.py" --install-vosk
:skipvosk

rem ── 3. Create the Startup shortcut ────────────────────────────────────────
echo.
echo  Registering the Sentinel for auto-start...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$w = New-Object -ComObject WScript.Shell; $s = $w.CreateShortcut('%LNK%'); $s.TargetPath = 'wscript.exe'; $s.Arguments = '\"%VBS%\"'; $s.WorkingDirectory = '%~dp0'; $s.Description = 'JARVIS Sentinel - clap + wake up listener'; $s.WindowStyle = 7; $s.Save()"

if not exist "%LNK%" (
    echo  [X] Could not create the startup shortcut.
    echo      You can still run Sentinel.bat manually.
    pause
    exit /b 1
)
echo  [OK] Auto-start installed.

rem ── 4. Start it right now so it works without a reboot ────────────────────
echo.
echo  Starting the Sentinel now...
"%PY%" "%~dp0sentinel.py" --stop >nul 2>&1
start "" wscript.exe "%VBS%"
timeout /t 3 /nobreak >nul
"%PY%" "%~dp0sentinel.py" --status

echo.
echo  ==========================================================
echo    DONE.  From now on, just:
echo.
echo        1. CLAP TWICE
echo        2. SAY  "WAKE UP"
echo.
echo    JARVIS starts and answers by itself - at login, forever.
echo    Test your claps with Clap-Test.bat
echo    Remove auto-start with uninstall_autostart.bat
echo  ==========================================================
echo.
pause
exit /b 0
