@echo off
python verify_gradcam.py > verification_log_bat.txt 2>&1
if %errorlevel% neq 0 (
    echo Python script failed with error level %errorlevel% >> verification_log_bat.txt
) else (
    echo Python script completed successfully >> verification_log_bat.txt
)
