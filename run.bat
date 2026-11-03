@echo off
rem Model Lab launcher: creates the virtual environment on first run, installs
rem dependencies when they are missing, starts the server and opens the browser.
rem   run.bat            live providers (needs an API key)
rem   run.bat --mock     built-in mock provider, no key needed
rem   run.bat --port 9000
setlocal EnableExtensions
title Model Lab
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

if exist "%PY%" goto :deps

echo [Model Lab] Creating virtual environment in .venv ...
where py >nul 2>nul
if errorlevel 1 (
    python -m venv .venv
) else (
    py -3 -m venv .venv
)
if not exist "%PY%" (
    echo.
    echo [Model Lab] Could not create the virtual environment.
    echo             Install Python 3.10 or newer from https://www.python.org/downloads/
    echo             and tick "Add python.exe to PATH", then run this file again.
    pause
    exit /b 1
)

:deps
"%PY%" -c "import fastapi, uvicorn, httpx, pytest" >nul 2>nul
if not errorlevel 1 goto :start

echo [Model Lab] Installing dependencies ...
"%PY%" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
    echo.
    echo [Model Lab] Installing dependencies failed. Check your internet connection and try again.
    pause
    exit /b 1
)

:start
"%PY%" -m app --open %*
if errorlevel 1 (
    echo.
    echo [Model Lab] The server stopped with an error. If the port is busy, try: run.bat --port 8010
    pause
)
endlocal
