@echo off
echo ==========================================
echo Pushing changes to Brain-Tumor-Detection-Hybrid
echo ==========================================

echo.
echo 1. Adding 'hybrid' remote...
git remote add hybrid https://github.com/Abhijith-Reddy-ch/Brain-Tumor-Detection-Hybrid.git
if %errorlevel% neq 0 (
    echo Remote 'hybrid' might already exist. Continuing...
)

echo.
echo 2. Staging changes...
git add .

echo.
echo 3. Committing changes...
git commit -m "Fix Grad-CAM and enable Hybrid model in Interface"

echo.
echo 4. Pushing to 'hybrid' remote (main branch)...
git push -u hybrid main

echo.
echo ==========================================
echo Done.
pause
