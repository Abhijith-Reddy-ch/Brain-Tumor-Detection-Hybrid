@echo off
echo Starting training script wrapper...
python -u train.py --train_dir "C:\Users\abhij\Desktop\brain_tumor_qnn_only\brain_tumor_qnn_only\Data\Training" --test_dir "C:\Users\abhij\Desktop\brain_tumor_qnn_only\brain_tumor_qnn_only\Data\Testing" --epochs 20
pause
