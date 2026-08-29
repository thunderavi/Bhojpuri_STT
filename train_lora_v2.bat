@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HOME=F:\bhojpuri-AI\.hf_cache
set HF_HUB_CACHE=F:\bhojpuri-AI\.hf_cache\hub
set HF_DATASETS_CACHE=F:\bhojpuri-AI\.hf_cache\datasets

echo ============================================================
echo   Bhojpuri Whisper-Small -- LoRA v2 Fine-Tuning (IMPROVED)
echo   Base: models/LORAmodel (Checkpoint-14900)
echo   LoRA r=16, alpha=32, modules: q/k/v/out_proj
echo   Target: 2000 Steps (~3h)  ^|  Eval Every 500 Steps
echo ============================================================

.\.venv\Scripts\python.exe scripts\train_lora_whisper.py ^
    --lora-rank 16 ^
    --lora-alpha 32 ^
    --lora-dropout 0.05 ^
    --lora-target-modules q_proj,k_proj,v_proj,out_proj ^
    --learning-rate 5e-5 ^
    --warmup-steps 200 ^
    --max-steps 2000 ^
    --eval-steps 500 ^
    --save-steps 500 ^
    --logging-steps 25 ^
    --per-device-train-batch-size 4 ^
    --per-device-eval-batch-size 8 ^
    --gradient-accumulation-steps 4 ^
    --output-dir models/LORAmodel/lora-v2-checkpoints ^
    --report-path models/LORAmodel/lora_v2_wer_report.txt ^
    %*

pause
