@echo off
title PILSS Custom Patch Dataset Labeller
setlocal enabledelayedexpansion

:: 1. Check Python executable in PATH
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in your system PATH!
    echo Please install Python 3.9+ from https://python.org and ensure "Add to PATH" is checked.
    pause
    exit /b 1
)

:: 2. Run Boot-Cached Requirement Check
python check_environment.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Dependency check or installation failed.
    pause
    exit /b 1
)

:: 3. Launch the Labeller Application
echo Starting PILSS Patch Labeller...
python patch_dataset_labeller.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [NOTICE] Application closed with exit code %ERRORLEVEL%.
    pause
)
