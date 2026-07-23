@echo off
chcp 65001 >nul 2>nul
title Build Collatz GPU Validator

echo ============================================================
echo   Building Collatz GPU Validator
echo   Target: AMD Radeon 9060XT
echo ============================================================
echo.

cd /d "%~dp0"

REM Load Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if %errorlevel% neq 0 (
    call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 2>nul
)

REM Set ROCm path
set PATH=C:\Program Files\AMD\ROCm\7.1\bin;%PATH%

echo [OK] Environment ready
echo.

REM Compile
hipcc -O3 -std=c++17 ^
      --offload-arch=native ^
      -ffast-math ^
      -funroll-loops ^
      -fomit-frame-pointer ^
      -march=native ^
      -D__HIP_PLATFORM_AMD__ ^
      -DNDEBUG ^
      src/collatz.hip src/collatz.cpp ^
      -o collatz_amd.exe

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Compilation failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Build complete!
echo [INFO] collatz_amd.exe is ready
echo.
pause