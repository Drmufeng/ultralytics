@echo off
setlocal

set ROOT=%~dp0..\..
set PY=
set SCRIPT=%ROOT%\scripts\env\check_env.py

if defined PYTHON_EXE (
  set PY=%PYTHON_EXE%
)

if not defined PY (
  if exist "%~dp0..\..\..\.venv310\Scripts\python.exe" set PY=%~dp0..\..\..\.venv310\Scripts\python.exe
)

if not defined PY (
  if exist "%ROOT%\.venv310\Scripts\python.exe" set PY=%ROOT%\.venv310\Scripts\python.exe
)

if not defined PY (
  set PY=python
)

if not exist "%PY%" (
  if /I "%PY%"=="python" (
    where python >nul 2>nul
    if errorlevel 1 (
      echo [ERROR] Python executable not found in PATH.
      goto :end
    )
  ) else (
    echo [ERROR] Python not found: %PY%
    goto :end
  )
)

if not exist "%SCRIPT%" (
  echo [ERROR] Script not found: %SCRIPT%
  goto :end
)

echo [INFO] Checking and fixing environment...
"%PY%" "%SCRIPT%" --fix --prefer-gpu
echo [INFO] Exit code: %ERRORLEVEL%

:end
pause
endlocal
