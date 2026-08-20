@echo off
rem ===========================================================================
rem  _env.bat — shared Python locator for every JARVIS launcher.
rem  Sets:  PY  (console python)   PYW (windowless python)   JARVIS_DIR
rem  Do NOT add "setlocal" here: the variables must reach the calling script.
rem ===========================================================================
set "JARVIS_DIR=%~dp0"
set "PY="
set "PYW="

if exist "%JARVIS_DIR%.venv\Scripts\python.exe" (
    set "PY=%JARVIS_DIR%.venv\Scripts\python.exe"
    if exist "%JARVIS_DIR%.venv\Scripts\pythonw.exe" set "PYW=%JARVIS_DIR%.venv\Scripts\pythonw.exe"
    goto :done
)
if exist "%JARVIS_DIR%venv\Scripts\python.exe" (
    set "PY=%JARVIS_DIR%venv\Scripts\python.exe"
    if exist "%JARVIS_DIR%venv\Scripts\pythonw.exe" set "PYW=%JARVIS_DIR%venv\Scripts\pythonw.exe"
    goto :done
)
for /f "delims=" %%i in ('where python 2^>nul') do (
    if not defined PY set "PY=%%i"
)
if not defined PY (
    for /f "delims=" %%i in ('where py 2^>nul') do (
        if not defined PY set "PY=%%i"
    )
)
if defined PY (
    for %%d in ("%PY%") do set "PYDIR=%%~dpi"
    if exist "%PYDIR%pythonw.exe" set "PYW=%PYDIR%pythonw.exe"
)

:done
if not defined PYW set "PYW=%PY%"
exit /b 0
