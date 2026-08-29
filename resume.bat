@echo off
cd /d "%~dp0"
.\.venv\Scripts\python.exe .\scripts\train_bhojpuri_whisper.py --dataset .\data\merged_bhojpuri --output-dir .\models\bhojpuri-whisper-small-full --num-train-epochs 3 --per-device-train-batch-size 1 --gradient-accumulation-steps 8 --dataloader-num-workers 0 --wer-eval-steps 100 --save-steps 100 --resume-from-checkpoint .\models\bhojpuri-whisper-small-full\checkpoint-14000
