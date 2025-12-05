@echo off
echo ========================================
echo Brain-Tumor-Detection-Hybrid Setup
echo Step 1: Creating Repository Directory
echo ========================================
echo.

set DEST=c:\Users\abhij\Desktop\GitHub\Brain-Tumor-Detection-Hybrid

echo Creating main directory...
if not exist "%DEST%" mkdir "%DEST%"

echo Creating subdirectories...
cd /d "%DEST%"
mkdir src 2>nul
mkdir interface 2>nul
mkdir interface\templates 2>nul
mkdir interface\uploads 2>nul
mkdir docs 2>nul
mkdir scripts 2>nul
mkdir checkpoints 2>nul

echo.
echo ========================================
echo Directory structure created!
echo ========================================
echo.
echo Now run: COPY_FILES_TO_HYBRID.bat
echo.
pause
