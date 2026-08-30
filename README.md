# 🎙️ Bhojpuri ASR — Whisper Fine-Tuning Project

> Automatic Speech Recognition (ASR) for the Bhojpuri language using OpenAI Whisper-Small, custom full fine-tuning, and LoRA parameter-efficient fine-tuning (PEFT).

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Hardware & Environment](#-hardware--environment)
- [Project Structure](#-project-structure)
- [Training History & Results](#-training-history--results)
  - [Phase 1 — Full Fine-Tuning](#phase-1--full-fine-tuning-whisper-small)
  - [Phase 2 — LoRA v1](#phase-2--lora-v1-parameter-efficient-fine-tuning)
  - [Phase 3 — LoRA v2 (In Progress)](#phase-3--lora-v2-expanded-attention-modules)
- [Model Comparison](#-model-comparison)
- [Quick Start](#-quick-start)
- [Scripts Reference](#-scripts-reference)
- [Dataset](#-dataset)
- [Reports](#-reports)

---

## 🧭 Project Overview

This project fine-tunes OpenAI's **Whisper-Small** model (~244M parameters) for **Bhojpuri language ASR**, a low-resource Indo-Aryan language spoken by ~50 million people. The project also benchmarks against the **Vakyansh Wav2Vec2** model pre-trained on Bhojpuri data.

**Goal:** Minimize Word Error Rate (WER) on a custom merged Bhojpuri evaluation set.

---

## 💻 Hardware & Environment

| Item | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU |
| VRAM | ~8 GB |
| CUDA | 12.4 |
| Python | 3.11 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.15.0 |
| PEFT | Latest |
| OS | Windows 11 |

---

## 📁 Project Structure

```
bhojpuri-AI/
├── scripts/
│   ├── train_bhojpuri_whisper.py      # Phase 1: Full Whisper fine-tuning
│   ├── train_lora_whisper.py          # Phase 2 & 3: LoRA fine-tuning (v1 + v2)
│   ├── eval_vakyansh_bhojpuri.py      # Vakyansh Wav2Vec2 GPU evaluation
│   ├── merge_bhojpuri_datasets.py     # Merges multiple Bhojpuri datasets
│   ├── analyze_model.py               # Reads trainer_state.json and prints WER table
│   ├── smoke_test_whisper.py          # Quick sanity test of model output
│   ├── transcribe_wav_folder.py       # Batch transcribe a folder of .wav files
│   ├── generate_report.py             # Generates Whisper training PDF report
│   ├── generate_vakyansh_report.py    # Generates Vakyansh evaluation PDF report
│   └── generate_comparison_pdf_report.py  # Side-by-side model comparison PDF
│
├── models/
│   ├── bhojpuri-whisper-small-full/   # Full fine-tuned model (checkpoint-14900)
│   │   └── wer_report.txt             # WER log for all training checkpoints
│   ├── LORAmodel/                     # Base for LoRA training (copy of ckpt-14900)
│   │   ├── lora_wer_report.txt        # LoRA v1 WER log
│   │   ├── lora_v2_wer_report.txt     # LoRA v2 WER log
│   │   ├── lora-checkpoints/          # LoRA v1 intermediate checkpoints
│   │   ├── lora-v2-checkpoints/       # LoRA v2 intermediate checkpoints
│   │   ├── lora-adapters-best/        # Best LoRA v1 adapter weights
│   │   └── lora-merged-final/         # LoRA v1 merged standalone model (~922 MB)
│   └── vakyansh-bhojpuri/             # Vakyansh Wav2Vec2 model files
│
├── data/
│   └── merged_bhojpuri/               # HuggingFace dataset (83,853 train / 1,054 eval)
│
├── report/                            # PDF & text evaluation reports
│   ├── Bhojpuri_AI_Whisper_Training_Report.pdf
│   ├── Bhojpuri_AI_Vakyansh_Evaluation_Report.pdf
│   ├── Bhojpuri_ASR_Model_Comparison_Report.pdf
│   ├── Bhojpuri_ASR_Model_Comparison_Report.md  # Master comparison table (Markdown)
│   ├── vakyansh_wer_report.txt        # Per-sample Vakyansh predictions
│   ├── vakyansh_wer_summary.txt       # Vakyansh final summary
│   └── vakyansh_checkpoint.json       # Auto-resume state for Vakyansh eval
│
├── wav/                               # Raw .wav test files for transcription
├── outputs/                           # Misc output files
│
├── train_lora.bat                     # 🚀 Run LoRA v1 training
├── train_lora_v2.bat                  # 🚀 Run LoRA v2 training (improved)
├── resume.bat                         # Resume full fine-tuning from checkpoint
├── eval_vakyansh.bat                  # Run Vakyansh GPU evaluation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

---

## 📈 Training History & Results

### Phase 1 — Full Fine-Tuning (Whisper-Small)

**Script:** `scripts/train_bhojpuri_whisper.py`  
**Dataset:** 83,853 training samples | 1,054 eval samples  
**Effective batch size:** 16 (batch=4 × grad_accum=4)

The base `openai/whisper-small` model was fully fine-tuned for ~1.42 epochs over 14,900 steps.

#### WER Progression (Key Checkpoints)

| Step | Epoch | Eval Loss | WER |
|------|-------|-----------|-----|
| 900 | 0.086 | 0.3724 | 53.00% |
| 1,700 | 0.162 | 0.3420 | 49.19% |
| 3,000 | 0.286 | 0.3301 | 49.46% |
| 5,200 | 0.496 | 0.2985 | 45.28% |
| 7,700 | 0.735 | 0.2813 | 43.48% |
| 8,700 | 0.830 | 0.2776 | 42.74% |
| 9,600 | 0.916 | 0.2681 | 43.12% |
| 10,500 | 1.002 | 0.2668 | 41.66% |
| 10,800 | 1.030 | 0.2712 | 41.56% |
| 13,600 | 1.298 | 0.2633 | 41.65% |
| 14,500 | 1.383 | 0.2604 | 41.06% |
| **14,900** | **1.422** | **0.2655** | **40.58% ✅ Best** |

**Result:** WER **40.58%** at checkpoint-14900.  
**Model size:** 922 MB

---

### Phase 2 — LoRA v1 (Parameter-Efficient Fine-Tuning)

**Script:** `scripts/train_lora_whisper.py`  
**Run command:** `.\train_lora.bat`  
**Starting from:** `models/LORAmodel` (copy of checkpoint-14900)

#### LoRA v1 Configuration

| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| Alpha (α) | 16 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Learning rate | 1e-4 |
| Max steps | 3,500 |
| Warmup steps | 100 |
| Trainable params | 884,736 / 242,619,648 (**0.36%**) |

#### Issue Encountered & Fixed

The initial run **stalled at preprocessing step [4/6]** due to a Windows multiprocessing bug.  
`datasets.map()` with `num_proc=8` uses Python's `spawn`-based multiprocessing on Windows, which cannot pickle lambda functions with `nonlocal` variables.  
**Fix:** Removed the `num_proc` argument entirely (runs synchronously — safe and correct on Windows).

#### LoRA v1 WER Results

| Step | WER |
|---|---|
| 100 (start) | 40.59% |
| 700 | **38.99%** ← First time sub-39% |
| 1,400 | 38.90% |
| 1,800 | **38.86%** ← Best checkpoint |
| 2,700 | 39.13% |
| 3,500 | 40.17% (scheduler fully decayed) |
| **Final eval** | **38.91%** ✅ |

**Result:** WER improved from **40.58% → 38.91%** (+1.67% improvement).  
**Trainable parameters:** only **884K out of 242M** (0.36%) were updated.  
**Merged model saved to:** `models/LORAmodel/lora-merged-final/`

#### Why WER Plateaued

After step ~1,800, WER stopped improving. The cosine learning rate scheduler decayed to near-zero by step 3,500 (`lr ≈ 5.88e-08`). Continuing would not help — a fresh run with better hyperparameters is needed.

---

### Phase 3 — LoRA v2 (Expanded Attention Modules)

**Script:** `scripts/train_lora_whisper.py` (same script, new CLI args)  
**Run command:** `.\train_lora_v2.bat`  
**Starting from:** `models/LORAmodel` (same base checkpoint-14900)  
**Status:** ✅ Complete

#### LoRA v2 Configuration

| Parameter | v1 | v2 | Reason for Change |
|---|---|---|---|
| Rank (r) | 8 | **16** | 2x capacity for richer adaptation |
| Alpha (α) | 16 | **32** | Maintains α/r = 2 ratio |
| Target modules | q, v | **q, k, v, out** | All 4 attention projections → more coverage |
| Learning rate | 1e-4 | **5e-5** | Less aggressive, avoids overshoot |
| Warmup steps | 100 | **200** | Smoother ramp-up |
| Max steps | 3,500 | **2,000** | — |
| Output dir | lora-checkpoints | **lora-v2-checkpoints** | v1 preserved |
| WER log | lora_wer_report.txt | **lora_v2_wer_report.txt** | v1 log preserved |

#### LoRA v2 WER Results

| Step | Epoch | WER |
|------|-------|-----|
| 600 | 0.286 | **38.88%** ← Best checkpoint |
| 2,000 | 0.382 | 39.45% (LR fully decayed) |
| **Final eval** | — | **38.91% ✅** |

**Actual runtime:** ~8h 49min (eval took ~96 min/checkpoint vs ~12 min in v1 due to larger rank)  
**Merged model saved to:** `models/LORAmodel/lora-v2-merged-final/`

#### Why v2 Matched But Didn't Beat v1

The larger rank (r=16) and expanded modules helped early (step 600 WER: 38.88%), but the cosine LR scheduler decayed to near-zero by step 2,000. The model essentially tied v1 at **38.91%**. The eval speed was also 8x slower with the larger configuration, making each eval ~96 minutes.

---

## 📊 Model Comparison

| Model | Type | WER |
|---|---|---|
| Vakyansh Wav2Vec2 (60h Bhojpuri) | Pre-trained CTC | **108.22%** |
| Whisper-Small (zero-shot) | No Bhojpuri training | ~75%+ |
| Whisper-Small Full Fine-Tuned | Full fine-tune, 14,900 steps | **40.58%** |
| Whisper-Small + LoRA v1 | PEFT, r=8, q+v, 3,500 steps | **38.91%** |
| Whisper-Small + LoRA v2 | PEFT, r=16, q+k+v+out, 2,000 steps | **38.91%** (tied v1) |

> **Lower WER = Better.** The Vakyansh model scores >100% WER on our test set because it was trained on a different data distribution and hallucinates extra tokens.

---

## 🚀 Quick Start

### 1. Set up environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare dataset

```powershell
.\.venv\Scripts\python.exe scripts\merge_bhojpuri_datasets.py
```

### 3. Run LoRA v2 training (recommended)

```powershell
.\train_lora_v2.bat
```

### 4. Transcribe a WAV file

```powershell
.\.venv\Scripts\python.exe scripts\transcribe_wav_folder.py --model models/LORAmodel/lora-merged-final --input wav/
```

### 5. Evaluate Vakyansh model

```powershell
.\eval_vakyansh.bat
```

---

## 📜 Scripts Reference

| Script | Purpose |
|---|---|
| `train_bhojpuri_whisper.py` | Full Whisper-Small fine-tuning from scratch |
| `train_lora_whisper.py` | LoRA fine-tuning (v1 + v2 via CLI args) |
| `eval_vakyansh_bhojpuri.py` | GPU-batched Vakyansh Wav2Vec2 evaluation |
| `merge_bhojpuri_datasets.py` | Merges and deduplicates Bhojpuri datasets |
| `analyze_model.py` | Prints WER table from `trainer_state.json` |
| `smoke_test_whisper.py` | Quick inference test on a single audio |
| `transcribe_wav_folder.py` | Batch transcription of a folder of WAV files |
| `generate_report.py` | PDF report for Whisper training |
| `generate_vakyansh_report.py` | PDF report for Vakyansh evaluation |
| `generate_comparison_pdf_report.py` | Side-by-side PDF comparison of all models |

---

## 📦 Dataset

- **Source:** Multiple Bhojpuri audio corpora merged using `merge_bhojpuri_datasets.py`
- **Train split:** 83,853 samples
- **Eval split:** 1,054 samples
- **Format:** HuggingFace `datasets` saved to disk at `data/merged_bhojpuri/`
- **Sample rate:** 16,000 Hz (mono)
- **Text column:** `text`
- **Audio column:** `audio`

---

## 📄 Reports

All evaluation reports are stored in the `report/` folder:

| File | Description |
|---|---|
| `Bhojpuri_ASR_Model_Comparison_Report.md` | Master comparison table (Markdown) |
| `Bhojpuri_ASR_Model_Comparison_Report.pdf` | Same as above in PDF |
| `Bhojpuri_AI_Whisper_Training_Report.pdf` | Full fine-tuning detailed report |
| `Bhojpuri_AI_Vakyansh_Evaluation_Report.pdf` | Vakyansh model evaluation report |
| `vakyansh_wer_report.txt` | Per-sample reference vs hypothesis |
| `vakyansh_checkpoint.json` | Auto-resume checkpoint (1,054 samples) |

---

## 🔑 Key Findings

1. **Full fine-tuning is essential for low-resource languages** — zero-shot Whisper performs very poorly on Bhojpuri.
2. **LoRA is highly effective** — training only 0.36% of parameters gave a meaningful WER improvement.
3. **The Vakyansh Wav2Vec2 model** (despite being trained specifically on 60h of Bhojpuri audio) performs much worse (108% WER) on this eval set, likely due to train/test distribution mismatch.
4. **Windows multiprocessing gotcha** — HuggingFace `datasets.map()` with `num_proc > 0` will hang silently on Windows when the map function uses closures/lambdas. Always omit `num_proc` to force single-process execution.
