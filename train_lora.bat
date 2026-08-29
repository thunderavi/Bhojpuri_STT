@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HOME=F:\bhojpuri-AI\.hf_cache
set HF_HUB_CACHE=F:\bhojpuri-AI\.hf_cache\hub
set HF_DATASETS_CACHE=F:\bhojpuri-AI\.hf_cache\datasets

echo ============================================================
echo   Bhojpuri Whisper-Small -- LoRA Fine-Tuning Runner (FAST)
echo   Starting Base: models/LORAmodel (Checkpoint-14900)
echo   Target: 3,500 Steps (~2h remaining) ^| Eval Every 500 Steps
echo ============================================================

.\.venv\Scripts\python.exe scripts\train_lora_whisper.py --resume %*

pause
