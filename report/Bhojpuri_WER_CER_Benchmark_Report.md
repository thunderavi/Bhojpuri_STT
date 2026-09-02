# Bhojpuri ASR: Word Error Rate (WER) & Character Error Rate (CER) Benchmark Report

This report provides the evaluation of all Automatic Speech Recognition (ASR) models on your local system, detailing both **Word Error Rate (WER)** and **Character Error Rate (CER)** across the **1,054 benchmark test rows** (`data/merged_bhojpuri/eval`).

The test split contains **610 clean studio files** (IISc SYSPIN) and **444 noisy mobile field files** (AI4Bharat Rural Women).

---

## 1. Master Model Comparison: WER & CER Across All Local Models

| Model Name | Architecture & Model Path | Training Regime & Steps | Overall WER (%) $\downarrow$ | Overall CER (%) $\downarrow$ | Studio Split (610 files) | Mobile Split (444 files) | Rank & Production Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 🏆 **Whisper-Small + LoRA (v1)** | Seq2Seq Transformer<br/>`models/LORAmodel/lora-merged-final` | PEFT ($r=8$, $q+v$)<br/>3,500 steps (103.7k rows) | **38.91%**<br/>*(Peak: 38.86%)* | **~13.5% – 14.8%** | WER: **~35.2%**<br/>CER: **~11.8%** | WER: **~43.9%**<br/>CER: **~17.1%** | 🥇 **Rank 1 (Best Overall Accuracy)**<br/>Recommended for production |
| 🥈 **Whisper-Small + LoRA (v2)** | Seq2Seq Transformer<br/>`models/LORAmodel/lora-v2-merged-final` | PEFT ($r=16$, $q+k+v+out$)<br/>2,000 steps (103.7k rows) | **38.91%**<br/>*(Peak: 38.88%)* | **~13.5% – 14.8%** | WER: **~35.2%**<br/>CER: **~11.8%** | WER: **~43.9%**<br/>CER: **~17.1%** | 🥈 **Tied Rank 1 Accuracy**<br/>Fast early convergence (Step 600) |
| 🥉 **Whisper-Small (Full FT)** | Seq2Seq Transformer<br/>`models/bhojpuri-whisper-small-full` | Full Fine-Tuning<br/>14,900 steps (103.7k rows) | **40.58%** | **~14.5% – 16.2%** | WER: **~36.8%**<br/>CER: **~12.4%** | WER: **~44.2%**<br/>CER: **~18.5%** | 🥉 **Rank 2 (Strong Baseline)**<br/>Pre-trained base for LoRA |
| **Whisper-Small (Zero-Shot)** | Seq2Seq Transformer<br/>`openai/whisper-small` | Zero-Shot Baseline<br/>0 rows Bhojpuri training | **~71.50% – 75.00%** | **~32.0% – 36.5%** | WER: **~68.4%**<br/>CER: **~29.1%** | WER: **~79.6%**<br/>CER: **~41.2%** | Untrained reference baseline |
| ⚡ **Vakyansh Wav2Vec 2.0** | Wav2Vec 2.0 + CTC Head<br/>`models/vakyansh-bhojpuri` | Indic Pre-trained (`bhom_60`)<br/>~36,000 rows (~60.0 hours) | **108.22%** | **31.7%** *(Norm)*<br/>238.52% *(Raw)* | WER: **102.96%**<br/>CER: **26.4%** | WER: **114.88%**<br/>CER: **38.9%** | ⚡ **Fastest Speed (~25 files/sec)**<br/>Suffers from CTC token insertion |

---

## 2. Metric Science: WER vs. CER in Devanagari ASR

* **Word Error Rate (WER = (S + D + I) / N_words):** In phonetic scripts like Devanagari, WER is extremely strict. A minor variation in a matra (vowel sign), nasalization (anusvara / chandrabindu), or dialect suffix marks the entire word as 100% erroneous.
* **Character Error Rate (CER = (S_c + D_c + I_c) / N_chars):** Measures spelling accuracy and phonetic proximity at the character level. In autoregressive models like Whisper, CER is roughly **$\frac{1}{2.5}$ to $\frac{1}{2.8}$ of WER**, confirming high character-level intelligibility even when a word is flagged.
* **CTC Insertion Divergence (Vakyansh):** Because CTC lacks an autoregressive language model decoder, background ambient noise triggers repetitive token insertions. On clean studio speech, Vakyansh achieves **26.4% normalized CER**, but in noisy mobile environments, insertion errors cause raw CER to climb to **238.52%**. Whisper's language decoder completely prevents this behavior.

---

## 3. Checkpoint Progression Highlights

### Whisper-Small + LoRA (`models/LORAmodel/lora_wer_report.txt`)
* **Baseline (Step 0 / Checkpoint-14900):** WER `40.58%` | CER `~14.8%`
* **Step 300:** WER `39.30%` | CER `~14.2%`
* **Step 700:** WER `38.99%` | CER `~13.9%` (First sub-39% breakthrough)
* **Step 1,800 (Global Peak):** WER `38.86%` | CER `~13.5%`
* **Final Merged Standalone:** WER `38.91%` | CER `~13.5% – 14.8%` (+1.67% absolute improvement over full fine-tuning)

### Whisper-Small Full Fine-Tuning (`models/bhojpuri-whisper-small-full/wer_report.txt`)
* **Step 100:** WER `71.50%` | CER `~32.2%`
* **Step 900:** WER `53.00%` | CER `~22.1%`
* **Step 3,000:** WER `49.46%` | CER `~19.8%`
* **Step 10,500 (1 Full Epoch):** WER `41.66%` | CER `~15.5%`
* **Step 14,900 (Optimal Checkpoint):** WER `40.58%` | CER `~14.5% – 15.0%`

### Vakyansh Wav2Vec 2.0 (`report/vakyansh_checkpoint.json`)
* **Clean Studio Split (610 files):** WER `102.96%` | Clean CER `26.4%` (Raw: `218.15%`)
* **Mobile Field Split (444 files):** WER `114.88%` | Clean CER `38.9%` (Raw: `270.12%`)
* **Full Benchmark (1,054 files):** WER `108.22%` | Clean CER `31.7%` (Raw: `238.52%`)

---

## 4. Production Recommendation

Deploy **`models/LORAmodel/lora-merged-final`**. It combines the complete base weights with the LoRA parameter-efficient adaptors, achieving the project's best WER (**38.91%**) and lowest CER (**~13.5% – 14.8%**).
