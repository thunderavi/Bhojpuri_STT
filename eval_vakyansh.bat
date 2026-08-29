@echo off
title Bhojpuri ASR - Vakyansh Evaluation Benchmark
cd /d "f:\bhojpuri-AI"
echo ================================================================
echo    Starting Bhojpuri Vakyansh ASR Evaluation (Auto-Resume On)
echo ================================================================
echo.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
.\.venv\Scripts\python.exe -u .\scripts\eval_vakyansh_bhojpuri.py
echo.
echo Evaluation process finished. Check report folder for results.
pause
