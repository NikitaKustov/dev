@echo off
setlocal
cd /d "%~dp0\.."

python "ml-pipeline\scripts\bootstrap_all.py" %*
if errorlevel 1 (
  echo.
  echo Bootstrap failed.
  exit /b 1
)

echo.
echo Bootstrap completed successfully.
exit /b 0
