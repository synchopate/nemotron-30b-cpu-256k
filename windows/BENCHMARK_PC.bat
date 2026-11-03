@echo off
chcp 65001 >nul
title llama-modchip Windows Benchmark Scorecard
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0modchip_win.ps1" -Mode Bench %*
echo.
echo Premi un tasto per chiudere...
pause >nul
