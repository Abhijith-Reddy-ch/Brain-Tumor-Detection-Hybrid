@echo off
echo ========================================
echo Brain Tumor Detection - Hybrid Setup
echo ========================================
echo.

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing core dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Installing interface dependencies...
pip install -r interface\requirements.txt

echo.
echo Creating necessary directories...
if not exist "checkpoints" mkdir checkpoints
if not exist "data" mkdir data
if not exist "interface\uploads" mkdir interface\uploads

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To train the model, run:
echo   python src\train.py --train_dir "data\Training" --test_dir "data\Testing" --use_qnn
echo.
echo To start the web interface, run:
echo   cd interface
echo   python app.py
echo.
pause
