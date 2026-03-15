@echo off
REM Build script for MMC fNIRS simulation (Windows)

echo ========================================
echo Building MMC fNIRS Simulation
echo ========================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
echo.
echo Configuring with CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release

if errorlevel 1 (
    echo CMake configuration failed!
    exit /b 1
)

REM Build
echo.
echo Building...
cmake --build . --config Release --parallel

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo.
echo ========================================
echo Build complete: build\Release\mmc_fnirs.exe
echo ========================================
