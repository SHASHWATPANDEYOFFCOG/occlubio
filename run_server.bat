@echo off
REM Double-click launcher for the occlubio console. Args pass through, e.g.
REM   run_server.bat -Port 8080 -Reload
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_server.ps1" %*
pause
