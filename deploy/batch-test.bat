@echo off
cd /d "%~dp0\.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" training\batch_test.py %*
) else (
  python training\batch_test.py %*
)
pause
