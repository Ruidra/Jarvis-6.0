@echo off
rem ===========================================================================
rem  Clap-Test.bat — tune your clap detection.
rem  Prints a line every time a clap pattern is recognised. JARVIS is NOT
rem  launched. If your claps are missed, raise "clap_sensitivity" in
rem  config\api_keys.json (e.g. 1.4). If random noise triggers it, lower it
rem  (e.g. 0.7) or raise "clap_count" to 3.
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
echo  CLAP TEST - clap now. Ctrl+C to finish.
echo.
"%PY%" "%~dp0sentinel.py" --test
echo.
pause
exit /b 0
