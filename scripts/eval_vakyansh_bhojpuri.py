"""
eval_vakyansh_bhojpuri.py
=========================
High-speed GPU-accelerated evaluation of Vakyansh Bhojpuri ASR (Wav2Vec2)
against the evaluation split of merged_bhojpuri dataset.

Features:
  - 🚀 Full GPU (CUDA) Acceleration via Hugging Face Wav2Vec2ForCTC
  - ⚡ Batched Inference (processes multiple audio samples concurrently on CUDA)
  - 🔄 Auto-Resume: Automatically resumes from the last completed checkpoint
  - 💾 Comprehensive Reports:
      1. report/vakyansh_wer_summary.txt  (Comparison with Whisper-Small)
      2. report/vakyansh_wer_report.txt   (Per-sample Reference vs Hypothesis log)
      3. report/vakyansh_checkpoint.json  (Internal state for auto-resume)

Usage:
  python scripts/eval_vakyansh_bhojpuri.py
  (or run eval_vakyansh.bat)
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

# Safe UTF-8 environment setup for Windows
if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr.encoding != "utf-8" and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Paths & Cache ─────────────────────────────────────────────────────────────
HF_CACHE_DIR = Path("F:/bhojpuri-AI/.hf_cache")
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")

MODEL_ID         = "Harveenchadha/vakyansh-wav2vec2-bhojpuri-bhom-60"
DATASET_PATH     = Path("data/merged_bhojpuri")
REPORT_DIR       = Path("report")
REPORT_PATH      = REPORT_DIR / "vakyansh_wer_report.txt"
SUMMARY_PATH     = REPORT_DIR / "vakyansh_wer_summary.txt"
CHECKPOINT_PATH  = REPORT_DIR / "vakyansh_checkpoint.json"

TARGET_SR        = 16_000
BATCH_SIZE       = 16      # Batched inference on GPU
WHISPER_BEST_WER = 40.58   # Whisper-Small checkpoint-14900

import numpy as np
import torch
import librosa
import soundfile as sf
from jiwer import wer
from datasets import Dataset, Audio
from transformers import AutoModelForCTC, AutoProcessor

def main():
    print("=" * 65)
    print("      🎙️ VAKYANSH BHOJPURI ASR (GPU-ACCELERATED EVALUATION)")
    print("=" * 65)
    print(f" PyTorch version : {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f" CUDA Available  : {cuda_available}")
    if cuda_available:
        print(f" GPU Device Name : {torch.cuda.get_device_name(0)}")
        print(f" VRAM Available  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f" Active Device   : {device}")
    print(f" Batch Size      : {BATCH_SIZE}")
    print("=" * 65)

    # 1. Load Processor & Model on GPU
    print(f"\n[1/3] Loading HuggingFace model '{MODEL_ID}' on {device.type.upper()}...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE_DIR / "hub"))
        model = AutoModelForCTC.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE_DIR / "hub")).to(device)
        model.eval()
        print(f"      ✅ Model and Processor successfully loaded onto {device.type.upper()}.")
    except Exception as exc:
        print(f"      ❌ Failed to load model on {device}: {exc}")
        return

    # 2. Load Evaluation Dataset
    eval_path = DATASET_PATH / "eval"
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation dataset split not found at: {eval_path}")
    print(f"\n[2/3] Loading evaluation split: {eval_path.name}")
    eval_ds = Dataset.load_from_disk(str(eval_path))
    eval_ds = eval_ds.cast_column("audio", Audio(decode=False))
    total_samples = len(eval_ds)
    print(f"      ✅ Total evaluation samples: {total_samples}")

    # 3. Check for Checkpoint to Resume
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict[str, str]] = {}
    skipped_count = 0
    errors: list[str] = []

    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
            for item in ckpt_data.get("results", []):
                results[item["idx"]] = {"ref": item["ref"], "hyp": item["hyp"]}
            skipped_count = ckpt_data.get("skipped_count", 0)
            errors = ckpt_data.get("errors", [])
            print(f"\n[3/3] 🔄 RESUME DETECTED:")
            print(f"      Found previous checkpoint with {len(results)} completed samples.")
            if results:
                curr_refs = [v["ref"] for v in results.values()]
                curr_hyps = [v["hyp"] for v in results.values()]
                init_wer = wer(curr_refs, curr_hyps) * 100
                print(f"      Current checkpoint Running WER: {init_wer:.2f}%")
        except Exception as e:
            print(f"\n[3/3] Could not read checkpoint file ({e}), starting clean.")
            results = {}
    else:
        print("\n[3/3] Starting evaluation from sample 0.")

    # Audio helper
    def load_audio_array(audio_dict: dict) -> np.ndarray | None:
        raw_bytes = audio_dict.get("bytes")
        if raw_bytes:
            try:
                with io.BytesIO(raw_bytes) as buf:
                    arr, sr = sf.read(buf, dtype="float32", always_2d=False)
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=1)
                if sr != TARGET_SR:
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
                return arr
            except Exception:
                pass
        path_str = audio_dict.get("path", "")
        if path_str and Path(path_str).exists():
            try:
                arr, _ = librosa.load(path_str, sr=TARGET_SR, mono=True)
                return arr
            except Exception:
                pass
        return None

    def save_checkpoint_and_report():
        ckpt_payload = {
            "completed_samples": len(results),
            "total_samples": total_samples,
            "skipped_count": skipped_count,
            "errors": errors[-50:],
            "results": [{"idx": k, "ref": v["ref"], "hyp": v["hyp"]} for k, v in sorted(results.items())]
        }
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(ckpt_payload, f, ensure_ascii=False, indent=2)

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("=" * 150 + "\n")
            f.write("  VAKYANSH BHOJPURI (WAV2VEC2) GPU EVALUATION SAMPLES LOG\n")
            f.write(f"  Model   : {MODEL_ID}\n")
            f.write(f"  Dataset : {DATASET_PATH}\n")
            f.write(f"  Progress: {len(results)} / {total_samples} samples\n")
            f.write("=" * 150 + "\n")
            f.write(f"{'#':<6} | {'REFERENCE (GROUND TRUTH)':<70} | {'HYPOTHESIS (PREDICTION)':<70}\n")
            f.write("-" * 150 + "\n")
            for idx, item in sorted(results.items()):
                ref_p = (item["ref"][:66] + "..") if len(item["ref"]) > 68 else item["ref"]
                hyp_p = (item["hyp"][:66] + "..") if len(item["hyp"]) > 68 else item["hyp"]
                f.write(f"{idx:<6} | {ref_p:<70} | {hyp_p:<70}\n")

    # 4. Batched GPU Inference Loop
    pending_indices: list[int] = [i for i in range(total_samples) if i not in results]
    print(f"\n🚀 Running GPU batch inference on {len(pending_indices)} pending samples...")
    print("-" * 65)
    t_start = time.time()

    for batch_start in range(0, len(pending_indices), BATCH_SIZE):
        batch_indices = pending_indices[batch_start : batch_start + BATCH_SIZE]
        
        valid_indices: list[int] = []
        audio_arrays: list[np.ndarray] = []
        batch_refs: list[str] = []

        for idx in batch_indices:
            sample = eval_ds[idx]
            ref = str(sample.get("text", "") or "").strip()
            if not ref:
                skipped_count += 1
                continue

            audio_dict = sample.get("audio")
            if not audio_dict:
                skipped_count += 1
                continue

            arr = load_audio_array(audio_dict)
            if arr is None:
                skipped_count += 1
                errors.append(f"Sample {idx}: audio decode failed")
                continue

            valid_indices.append(idx)
            audio_arrays.append(arr)
            batch_refs.append(ref)

        if not audio_arrays:
            continue

        t0 = time.time()
        try:
            # Vectorized GPU inference
            inputs = processor(
                audio_arrays,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)

            for idx, ref, hyp in zip(valid_indices, batch_refs, transcriptions):
                results[idx] = {"ref": ref, "hyp": hyp.strip()}

        except Exception as exc:
            # Fallback to single item if batch causes OOM or error
            for idx, arr, ref in zip(valid_indices, audio_arrays, batch_refs):
                try:
                    inp = processor(arr, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(device)
                    with torch.no_grad():
                        lgt = model(inp).logits
                    pid = torch.argmax(lgt, dim=-1)
                    hyp = processor.batch_decode(pid)[0].strip()
                    results[idx] = {"ref": ref, "hyp": hyp}
                except Exception as e:
                    skipped_count += 1
                    errors.append(f"Sample {idx}: GPU inference error ({e})")

        batch_time = time.time() - t0
        save_checkpoint_and_report()

        # Display running stats
        curr_refs = [v["ref"] for v in results.values()]
        curr_hyps = [v["hyp"] for v in results.values()]
        running_wer = wer(curr_refs, curr_hyps) * 100
        speed = len(valid_indices) / max(batch_time, 0.001)
        print(f"[{len(results):>5}/{total_samples}] Running WER: {running_wer:6.2f}% | Batch: {len(valid_indices)} audios in {batch_time:.2f}s ({speed:.1f} audios/sec)")
        sys.stdout.flush()

    # Final Save
    save_checkpoint_and_report()

    # Final Metrics
    final_refs = [v["ref"] for v in results.values()]
    final_hyps = [v["hyp"] for v in results.values()]
    final_wer_pct = wer(final_refs, final_hyps) * 100 if final_refs else 0.0
    total_elapsed = time.time() - t_start

    winner = "Whisper-Small (checkpoint-14900)" if WHISPER_BEST_WER < final_wer_pct else "Vakyansh Bhojpuri (Wav2Vec2)"
    diff_wer = abs(final_wer_pct - WHISPER_BEST_WER)

    summary_content = f"""=================================================================
       BHOJPURI ASR BENCHMARK - FINAL COMPARISON REPORT
=================================================================
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
Device Used            : {device.type.upper()} ({torch.cuda.get_device_name(0) if cuda_available else 'CPU'})
Evaluation Split Samples: {total_samples}
Successfully Evaluated : {len(results)}
Skipped/Corrupt Audios  : {skipped_count}
Total Elapsed Time     : {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)

-----------------------------------------------------------------
  MODEL PERFORMANCE COMPARISON (WER - Lower is Better)
-----------------------------------------------------------------
  1. Fine-tuned Whisper-Small (checkpoint-14900) : {WHISPER_BEST_WER:6.2f}% WER
  2. Vakyansh Bhojpuri (Wav2Vec2 60h on GPU)     : {final_wer_pct:6.2f}% WER

-----------------------------------------------------------------
  🏆 WINNER & ANALYSIS
-----------------------------------------------------------------
  Best Model      : {winner}
  Margin of Lead  : {diff_wer:.2f}% WER advantage
  
  Architecture Comparison:
  - Whisper-Small : Multilingual Encoder-Decoder Seq2Seq Transformer
  - Vakyansh      : Wav2Vec 2.0 CTC Acoustic Model (60h Bhojpuri)
=================================================================
"""
    print("\n" + summary_content)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_content)

    print(f"✅ Samples log saved to   : {REPORT_PATH.resolve()}")
    print(f"✅ Summary report saved to: {SUMMARY_PATH.resolve()}")
    print(f"✅ State checkpoint saved : {CHECKPOINT_PATH.resolve()}")
    print("=" * 65)

if __name__ == "__main__":
    main()
