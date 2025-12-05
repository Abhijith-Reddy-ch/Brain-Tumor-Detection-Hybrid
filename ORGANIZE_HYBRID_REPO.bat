@echo off
echo ========================================
echo Brain-Tumor-Detection-Hybrid Setup
echo Organizing Files Automatically
echo ========================================
echo.

set SOURCE=c:\Users\abhij\Desktop\GitHub\Brain-Tumor-Detection
set DEST=c:\Users\abhij\Desktop\GitHub\Brain-Tumor-Detection-Hybrid

echo Creating directory structure...
cd %DEST%
if not exist "src" mkdir src
if not exist "interface" mkdir interface
if not exist "interface\templates" mkdir interface\templates
if not exist "interface\uploads" mkdir interface\uploads
if not exist "docs" mkdir docs
if not exist "scripts" mkdir scripts
if not exist "checkpoints" mkdir checkpoints

echo.
echo Copying root files...
copy "%SOURCE%\HYBRID_README.md" "%DEST%\README.md"
copy "%SOURCE%\HYBRID_.gitignore" "%DEST%\.gitignore"
copy "%SOURCE%\HYBRID_requirements.txt" "%DEST%\requirements.txt"
copy "%SOURCE%\HYBRID_LICENSE" "%DEST%\LICENSE"

echo.
echo Copying source files to src\...
copy "%SOURCE%\train.py" "%DEST%\src\"
copy "%SOURCE%\model.py" "%DEST%\src\"
copy "%SOURCE%\qnn_utils.py" "%DEST%\src\"
copy "%SOURCE%\dataset.py" "%DEST%\src\"
copy "%SOURCE%\export_onnx.py" "%DEST%\src\"

echo.
echo Copying interface files to interface\...
copy "%SOURCE%\Interface\app.py" "%DEST%\interface\"
copy "%SOURCE%\Interface\model_utils.py" "%DEST%\interface\"
copy "%SOURCE%\Interface\requirements.txt" "%DEST%\interface\"
copy "%SOURCE%\Interface\templates\index.html" "%DEST%\interface\templates\"

echo.
echo Copying setup scripts to scripts\...
copy "%SOURCE%\HYBRID_setup.bat" "%DEST%\scripts\setup.bat"
copy "%SOURCE%\HYBRID_setup.sh" "%DEST%\scripts\setup.sh"

echo.
echo Creating placeholder files...
echo. > "%DEST%\checkpoints\.gitkeep"
echo. > "%DEST%\interface\uploads\.gitkeep"

echo.
echo Copying model checkpoint if it exists...
if exist "%SOURCE%\checkpoints\best.pth" (
    copy "%SOURCE%\checkpoints\best.pth" "%DEST%\checkpoints\"
    echo Model checkpoint copied successfully!
) else (
    echo Model checkpoint not found - you'll need to download it separately
)

echo.
echo ========================================
echo Files organized successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Navigate to: %DEST%
echo 2. Initialize Git: git init
echo 3. Add files: git add .
echo 4. Commit: git commit -m "Initial commit: Hybrid QNN-CNN Brain Tumor Detection"
echo 5. Create GitHub repo and push
echo.
echo See HYBRID_ORGANIZATION_GUIDE.md for detailed instructions
echo.
pause
