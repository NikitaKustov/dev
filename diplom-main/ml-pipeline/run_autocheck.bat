@echo off
setlocal
cd /d "%~dp0\.."

python "ml-pipeline\scripts\autocheck.py" --check-server --check-scripts --check-data %*
if errorlevel 1 (
  echo.
  echo Auto-check failed.
  exit /b 1
)

echo.
echo Auto-check passed.
exit /b 0
