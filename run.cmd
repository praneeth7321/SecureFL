@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
if errorlevel 1 (
    echo.
    echo Run failed. Review the message above.
    pause
    exit /b 1
)
echo.
echo Press any key to close this window.
pause >nul
