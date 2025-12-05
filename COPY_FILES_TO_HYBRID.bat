@echo off
echo ========================================
echo Brain-Tumor-Detection-Hybrid Setup
echo Step 2: Copying Files
echo ========================================
echo.

set SOURCE=c:\Users\abhij\Desktop\GitHub\Brain-Tumor-Detection
set DEST=c:\Users\abhij\Desktop\GitHub\Brain-Tumor-Detection-Hybrid

echo Copying root files...
copy "%SOURCE%\HYBRID_README.md" "%DEST%\README.md" >nul 2>&1 && echo [OK] README.md || echo [FAIL] README.md
copy "%SOURCE%\HYBRID_.gitignore" "%DEST%\.gitignore" >nul 2>&1 && echo [OK] .gitignore || echo [FAIL] .gitignore
copy "%SOURCE%\HYBRID_requirements.txt" "%DEST%\requirements.txt" >nul 2>&1 && echo [OK] requirements.txt || echo [FAIL] requirements.txt
copy "%SOURCE%\HYBRID_LICENSE" "%DEST%\LICENSE" >nul 2>&1 && echo [OK] LICENSE || echo [FAIL] LICENSE

echo.
echo Copying source files to src\...
copy "%SOURCE%\train.py" "%DEST%\src\" >nul 2>&1 && echo [OK] train.py || echo [FAIL] train.py
copy "%SOURCE%\model.py" "%DEST%\src\" >nul 2>&1 && echo [OK] model.py || echo [FAIL] model.py
copy "%SOURCE%\qnn_utils.py" "%DEST%\src\" >nul 2>&1 && echo [OK] qnn_utils.py || echo [FAIL] qnn_utils.py
copy "%SOURCE%\dataset.py" "%DEST%\src\" >nul 2>&1 && echo [OK] dataset.py || echo [FAIL] dataset.py
copy "%SOURCE%\export_onnx.py" "%DEST%\src\" >nul 2>&1 && echo [OK] export_onnx.py || echo [FAIL] export_onnx.py

echo.
echo Copying interface files to interface\...
copy "%SOURCE%\Interface\app.py" "%DEST%\interface\" >nul 2>&1 && echo [OK] app.py || echo [FAIL] app.py
copy "%SOURCE%\Interface\model_utils.py" "%DEST%\interface\" >nul 2>&1 && echo [OK] model_utils.py || echo [FAIL] model_utils.py
copy "%SOURCE%\Interface\requirements.txt" "%DEST%\interface\" >nul 2>&1 && echo [OK] interface requirements.txt || echo [FAIL] interface requirements.txt
copy "%SOURCE%\Interface\templates\index.html" "%DEST%\interface\templates\" >nul 2>&1 && echo [OK] index.html || echo [FAIL] index.html

echo.
echo Copying setup scripts to scripts\...
copy "%SOURCE%\HYBRID_setup.bat" "%DEST%\scripts\setup.bat" >nul 2>&1 && echo [OK] setup.bat || echo [FAIL] setup.bat
copy "%SOURCE%\HYBRID_setup.sh" "%DEST%\scripts\setup.sh" >nul 2>&1 && echo [OK] setup.sh || echo [FAIL] setup.sh

echo.
echo Creating placeholder files...
echo. > "%DEST%\checkpoints\.gitkeep" 2>nul && echo [OK] checkpoints\.gitkeep || echo [FAIL] checkpoints\.gitkeep
echo. > "%DEST%\interface\uploads\.gitkeep" 2>nul && echo [OK] interface\uploads\.gitkeep || echo [FAIL] interface\uploads\.gitkeep

echo.
echo Copying model checkpoint if it exists...
if exist "%SOURCE%\checkpoints\best.pth" (
    copy "%SOURCE%\checkpoints\best.pth" "%DEST%\checkpoints\" >nul 2>&1 && echo [OK] Model checkpoint copied! || echo [FAIL] Model checkpoint
) else (
    echo [SKIP] Model checkpoint not found - download separately from releases
)

echo.
echo ========================================
echo File organization complete!
echo ========================================
echo.
echo Repository location: %DEST%
echo.
echo Next steps:
echo 1. cd %DEST%
echo 2. git init
echo 3. git add .
echo 4. git commit -m "Initial commit: Hybrid QNN-CNN Brain Tumor Detection"
echo 5. Create GitHub repo and push
echo.
pause
