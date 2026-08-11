@echo off
chcp 65001 >nul 2>nul
title Build AMD HIP Collatz Verifier - EXTREME

echo ============================================================
echo   Building AMD HIP Collatz Verifier
echo   EXTREME OPTIMIZATION - 9060XT 2048 Core Edition
echo ============================================================
echo.

cd /d "%~dp0"

REM Visual Studio
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if %errorlevel% neq 0 (
    call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 2>nul
)

REM HIP
set PATH=C:\Program Files\AMD\ROCm\7.1\bin;%PATH%

REM ============================================================
REM Force GPU 0, disable fallback to system RAM
REM ============================================================
set HIP_VISIBLE_DEVICES=0
set HIP_FORCE_DEVICE=0
set HIP_DEVICE=0

echo [OK] Environment ready - GPU 0 forced
echo.

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
echo [SUCCESS] Compilation complete!
echo [INFO] collatz_amd.exe is ready - VRAM forced
echo.
pause