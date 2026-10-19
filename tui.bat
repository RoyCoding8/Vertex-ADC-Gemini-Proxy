@echo off
cd /d "%~dp0"
uv run python tui.py %*
if errorlevel 1 pause
