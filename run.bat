@echo off
cd /d "%~dp0"
echo ========================================
echo   AMD HIP Collatz Verifier
echo   Radeon 9060XT Optimized
echo ========================================
echo.
echo [1] Start (continue from checkpoint)
echo [2] Reset and start from beginning
echo [3] Exit
echo.
set /p choice="Select (1-3): "

if "%choice%"=="1" goto RUN
if "%choice%"=="2" goto RESET
if "%choice%"=="3" goto EXIT

echo Invalid choice
pause
goto :EOF

:RUN
echo.
echo Starting verifier...
collatz_amd.exe
pause
goto :EOF

:RESET
echo.
set /p confirm="Delete checkpoint? (Y/N): "
if /i "%confirm%"=="Y" (
    if exist "checkpoint.bin" del checkpoint.bin
    echo Checkpoint deleted.
) else (
    echo Cancelled.
)
pause
goto :EOF

:EXIT
echo Goodbye!
exit /b 0