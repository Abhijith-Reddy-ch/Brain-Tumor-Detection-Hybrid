@echo off
echo ==========================================
echo Resolving Conflicts and Pushing
echo ==========================================

echo.
echo 1. Accepting REMOTE version for conflicting files...
git checkout --theirs .gitignore LICENSE README.md requirements.txt scripts/setup.bat scripts/setup.sh

echo.
echo 2. Removing temporary/redundant files...
if exist Interface\model_utils_hybrid.py del Interface\model_utils_hybrid.py
if exist Interface\run_verify.bat del Interface\run_verify.bat
if exist Interface\test_python.py del Interface\test_python.py
if exist Interface\verify_gradcam.py del Interface\verify_gradcam.py
if exist check_git.py del check_git.py
if exist test_write.txt del test_write.txt
if exist push_to_hybrid.bat del push_to_hybrid.bat

echo.
echo 3. Staging resolved files...
git add .

echo.
echo 4. Committing merge...
git commit -m "Merge remote-tracking branch 'hybrid/main' into main"

echo.
echo 5. Pushing to 'hybrid' remote...
git push hybrid main

echo.
echo ==========================================
echo Done.
pause
