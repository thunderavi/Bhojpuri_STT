"""
train_lora_whisper.py
=====================
LoRA fine-tuning of Whisper-Small (Bhojpuri) using PEFT.

Starts from:  models/LORAmodel  (copy of checkpoint-14900, WER = 40.58%)
Saves to:     models/LORAmodel/lora-checkpoints/

LoRA Config:
  - Rank (r):           8
  - Alpha (lora_alpha): 16
  - Dropout:            0.05
  - Target modules:     q_proj, v_proj (encoder + decoder attention)

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\train_lora_whisper.py
  .\\.venv\\Scripts\\python.exe scripts\\train_lora_whisper.py --num-train-epochs 2 --learning-rate 1e-4

Resume:
  .\\.venv\\Scripts\\python.exe scripts\\train_lora_whisper.py --resume
"""
from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Cache dirs (same as original training script) ────────────────────────────
HF_CACHE_DIR = Path("F:/bhojpuri-AI/.hf_cache")
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_from_disk
from jiwer import wer as compute_wer
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MODEL_DIR   = Path("models/LORAmodel")          # checkpoint-14900 copy
OUTPUT_DIR       = Path("models/LORAmodel/lora-checkpoints")
DATASET_PATH     = Path("data/merged_bhojpuri")
REPORT_PATH      = Path("models/LORAmodel/lora_wer_report.txt")
TARGET_SR        = 16_000
MAX_LABEL_LENGTH = 448

# ── LoRA Hyperparameters ──────────────────────────────────────────────────────
LORA_RANK        = 8       # low-rank decomposition dimension
LORA_ALPHA       = 16      # scaling factor (alpha/r = 2 -> moderate strength)
LORA_DROPOUT     = 0.05    # regularization dropout
# Target: q_proj + v_proj in ALL attention layers (encoder + decoder)
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DEFAULT_TARGET_STR = ",".join(LORA_TARGET_MODULES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune Whisper-Small for Bhojpuri ASR")
    p.add_argument("--base-model",         default=str(BASE_MODEL_DIR))
    p.add_argument("--dataset",            default=str(DATASET_PATH))
    p.add_argument("--output-dir",         default=str(OUTPUT_DIR))
    p.add_argument("--lora-rank",          type=int,   default=LORA_RANK)
    p.add_argument("--lora-alpha",         type=int,   default=LORA_ALPHA)
    p.add_argument("--lora-dropout",       type=float, default=LORA_DROPOUT)
    p.add_argument("--learning-rate",      type=float, default=1e-4)
    p.add_argument("--num-train-epochs",   type=float, default=2.0)
    p.add_argument("--max-steps",          type=int,   default=3500,
                   help="Total training steps (default: 3500, ~2h remaining from step 1500). Set to -1 to use --num-train-epochs.")
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--per-device-eval-batch-size",  type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--warmup-steps",       type=int,   default=100)
    p.add_argument("--eval-steps",         type=int,   default=500)
    p.add_argument("--save-steps",         type=int,   default=500)
    p.add_argument("--logging-steps",      type=int,   default=25)
    p.add_argument("--generation-max-length", type=int, default=225)
    p.add_argument("--max-train-samples",  type=int,   default=None)
    p.add_argument("--max-eval-samples",   type=int,   default=None)
    p.add_argument("--resume",             action="store_true",
                   help="Auto-resume from latest LoRA checkpoint in output-dir")
    p.add_argument("--lora-target-modules", type=str,
                   default=LORA_DEFAULT_TARGET_STR,
                   help="Comma-separated LoRA target modules (e.g. q_proj,k_proj,v_proj,out_proj)")
    p.add_argument("--report-path",        type=str,
                   default=str(REPORT_PATH),
                   help="Path to WER log file (default: models/LORAmodel/lora_wer_report.txt)")
    return p.parse_args()


# ── Data helpers (identical to original training script) ─────────────────────

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def load_audio_array(audio: dict, *, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio from embedded bytes or fallback path, resample to target_sr."""
    raw_bytes = audio.get("bytes")
    if raw_bytes:
        try:
            with io.BytesIO(raw_bytes) as buf:
                arr, sr = sf.read(buf, dtype="float32", always_2d=False)
            if arr.ndim > 1:
                arr = np.mean(arr, axis=1)
            if sr != target_sr:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
            return arr
        except Exception:
            pass
    audio_path = audio.get("path")
    if audio_path and Path(audio_path).exists():
        arr, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
        return arr
    raise ValueError("Could not load audio: no valid bytes or path.")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class WERLogger(TrainerCallback):
    """Append step / epoch / WER metrics to a text file for tracking."""

    def __init__(self, path: Path, append: bool):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append and path.exists() else "w"
        with path.open(mode, encoding="utf-8") as f:
            f.write(f"\n=== LoRA Training run: {datetime.now().isoformat(timespec='seconds')} ===\n")
            f.write("step | epoch | metrics\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        metrics = ", ".join(f"{k}={v}" for k, v in sorted(logs.items()))
        with self.path.open("a", encoding="utf-8") as f:
            epoch = "" if state.epoch is None else f"{state.epoch:.4f}"
            f.write(f"{state.global_step} | {epoch} | {metrics}\n")


def find_latest_lora_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for p in output_dir.glob("checkpoint-*"):
        if p.is_dir():
            try:
                checkpoints.append((int(p.name.split("-")[-1]), p))
            except ValueError:
                pass
    if not checkpoints:
        return None
    return sorted(checkpoints)[-1][1]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    device   = "cuda" if use_cuda else "cpu"

    if use_cuda:
        # Ampere RTX 3070 optimizations: TF32 matmuls & cuDNN benchmark
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print("  Bhojpuri Whisper-Small -- LoRA Fine-Tuning (FAST)")
    print("=" * 60)
    print(f"  Device        : {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_cuda else ""))
    print(f"  Base model    : {args.base_model}")
    print(f"  LoRA rank     : {args.lora_rank}  |  alpha: {args.lora_alpha}  |  dropout: {args.lora_dropout}")
    print(f"  Target modules: {LORA_TARGET_MODULES}")
    print(f"  Learning rate : {args.learning_rate}")
    if args.max_steps > 0:
        print(f"  Max steps     : {args.max_steps} (target completion: ~2h from checkpoint-1500)")
    else:
        print(f"  Epochs        : {args.num_train_epochs}")
    print(f"  Eval frequency: every {args.eval_steps} steps (batch size: {args.per_device_eval_batch_size})")
    print(f"  Output dir    : {args.output_dir}")
    print("=" * 60)

    # ── 1. Load processor & base model ───────────────────────────────────────
    print("\n[1/6] Loading processor and base Whisper model from LORAmodel folder...")
    processor = WhisperProcessor.from_pretrained(
        args.base_model, language="Hindi", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="Hindi", task="transcribe"
    )
    model.config.use_cache = False          # required for training
    model.config.dropout   = 0.0           # base architecture dropout off
    print(f"  Base model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 2. Apply LoRA adapters ────────────────────────────────────────────────
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    print("\n[2/6] Attaching LoRA adapters (PEFT)...")
    print(f"  Target modules: {target_modules}")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total       = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ── 3. Load & preprocess dataset ─────────────────────────────────────────
    print("\n[3/6] Loading merged Bhojpuri dataset from disk...")
    dataset  = load_from_disk(str(args.dataset))
    train_ds = dataset["train"]
    eval_ds  = dataset["eval"]

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds  = eval_ds.select(range(min(args.max_eval_samples,  len(eval_ds))))

    # Detect text column
    for col in ("text", "transcript", "sentence", "normalized_text", "transcription"):
        if col in train_ds.column_names:
            text_col = col
            break
    else:
        raise ValueError(f"No text column found in: {train_ds.column_names}")

    # Detect audio column
    for col in ("audio", "speech", "input_audio"):
        if col in train_ds.column_names:
            audio_col = col
            break
    else:
        raise ValueError(f"No audio column found in: {train_ds.column_names}")

    print(f"  Train: {len(train_ds):,} samples  |  Eval: {len(eval_ds):,} samples")
    print(f"  Audio column: {audio_col}  |  Text column: {text_col}")

    train_ds = train_ds.cast_column(audio_col, Audio(decode=False))
    eval_ds  = eval_ds.cast_column(audio_col,  Audio(decode=False))

    skipped = {"train": 0, "eval": 0}
    reported = 0

    def prepare_batch(batch: dict, split: str) -> dict:
        nonlocal reported
        in_feats, labs = [], []
        for audio, raw_text in zip(batch[audio_col], batch[text_col]):
            text = normalize_text(raw_text)
            if not text:
                skipped[split] += 1
                continue
            try:
                arr = load_audio_array(audio)
                feats = processor.feature_extractor(arr, sampling_rate=TARGET_SR)
                ids   = processor.tokenizer(
                    text, max_length=MAX_LABEL_LENGTH, truncation=True
                ).input_ids
                in_feats.append(feats.input_features[0])
                labs.append(ids)
            except Exception as exc:
                skipped[split] += 1
                if reported < 5:
                    print(f"  Skipping row ({split}): {exc}")
                    reported += 1
        return {"input_features": in_feats, "labels": labs}

    print("\n[4/6] Preprocessing audio -> mel spectrogram features...")
    train_ds = train_ds.map(
        lambda b: prepare_batch(b, "train"),
        batched=True, batch_size=8, remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda b: prepare_batch(b, "eval"),
        batched=True, batch_size=8, remove_columns=eval_ds.column_names,
    )
    print(f"  Ready -> Train: {len(train_ds):,}  |  Eval: {len(eval_ds):,}")
    print(f"  Skipped -> Train: {skipped['train']}  |  Eval: {skipped['eval']}")

    # ── 4. Metrics ────────────────────────────────────────────────────────────
    def compute_metrics(pred) -> dict:
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_ids  = np.asarray(pred_ids)
        label_ids = np.where(
            pred.label_ids != -100, pred.label_ids,
            processor.tokenizer.pad_token_id
        )
        hyp = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": compute_wer(reference=ref, hypothesis=hyp)}

    # ── 5. Trainer setup ──────────────────────────────────────────────────────
    print("\n[5/6] Configuring Seq2SeqTrainer...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resume from latest LoRA checkpoint
    resume_from = None
    if args.resume:
        latest = find_latest_lora_checkpoint(output_dir)
        if latest:
            resume_from = str(latest)
            print(f"  Resuming from: {latest}")
        else:
            print("  No existing LoRA checkpoint found -- starting fresh.")
    else:
        print("  Starting LoRA training (base weights loaded from models/LORAmodel).")

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = str(output_dir),
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size  = args.per_device_eval_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate               = args.learning_rate,
        warmup_steps                = args.warmup_steps,
        num_train_epochs            = args.num_train_epochs,
        max_steps                   = args.max_steps if args.max_steps > 0 else -1,
        fp16                        = use_cuda,
        dataloader_num_workers      = 1,
        dataloader_pin_memory       = use_cuda,
        logging_steps               = args.logging_steps,
        eval_strategy               = "steps",
        eval_steps                  = args.eval_steps,
        save_steps                  = args.save_steps,
        save_total_limit            = 3,
        predict_with_generate       = True,
        generation_max_length       = args.generation_max_length,
        report_to                   = "none",
        remove_unused_columns       = False,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        label_names                 = ["labels"],
    )

    trainer = Seq2SeqTrainer(
        args             = training_args,
        model            = model,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        processing_class = processor,
        data_collator    = DataCollatorSpeechSeq2SeqWithPadding(processor),
        compute_metrics  = compute_metrics,
    )
    trainer.add_callback(WERLogger(Path(args.report_path), append=(resume_from is not None)))

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("\n[6/6] Starting LoRA training on GPU...")
    print(f"  Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"  WER before LoRA (checkpoint-14900 baseline): 40.58%")
    print("-" * 60)

    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # ── Save LoRA adapters ────────────────────────────────────────────────────
    lora_adapter_path = output_dir.parent / f"{output_dir.name.replace('lora-checkpoints', 'lora-adapters-best').replace('checkpoints', 'adapters-best')}"
    model.save_pretrained(str(lora_adapter_path))
    processor.save_pretrained(str(lora_adapter_path))
    print(f"\n[OK] LoRA adapters saved -> {lora_adapter_path.resolve()}")

    # ── Merge LoRA into base & save merged model ──────────────────────────────
    print("\nMerging LoRA adapters into base model weights...")
    merged_model = model.merge_and_unload()
    merged_path  = output_dir.parent / f"{output_dir.name.replace('lora-checkpoints', 'lora-merged-final').replace('checkpoints', 'merged-final')}"
    merged_model.save_pretrained(str(merged_path))
    processor.save_pretrained(str(merged_path))
    print(f"[OK] Merged model saved -> {merged_path.resolve()}")
    print(f"     (Inference-ready standalone model, ~922 MB)")

    # ── Final eval metrics ────────────────────────────────────────────────────
    final_metrics = trainer.evaluate()
    final_wer = final_metrics.get("eval_wer", float("nan"))
    print(f"\n{'='*60}")
    print(f"  LoRA FINAL WER : {final_wer * 100:.2f}%")
    print(f"  Baseline WER   : 40.58%  (checkpoint-14900, full fine-tune)")
    improvement = 40.58 - (final_wer * 100)
    print(f"  Improvement    : {improvement:+.2f}%")
    print(f"{'='*60}")

    with Path(args.report_path).open("a", encoding="utf-8") as f:
        f.write(f"\nfinal_eval | {final_metrics}\n")
        f.write(f"baseline_wer=40.58% | lora_wer={final_wer*100:.2f}% | improvement={improvement:+.2f}%\n")

    print(f"\nFull WER log -> {Path(args.report_path).resolve()}")


if __name__ == "__main__":
    main()
