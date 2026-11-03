@echo off
chcp 65001 >nul
title llama-modchip Windows Hybrid Edition (GPU + CPU MoE)
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0modchip_win.ps1" %*
if errorlevel 1 (
    echo.
    echo [!] Il programma si e' fermato con un errore. Premi un tasto per chiudere...
    pause >nul
)
