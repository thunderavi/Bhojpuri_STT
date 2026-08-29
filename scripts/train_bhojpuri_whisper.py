from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

HF_CACHE_DIR = Path("F:/bhojpuri-AI/.hf_cache")
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, DatasetDict, load_dataset, load_from_disk
from jiwer import wer
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


DEFAULT_MODEL = "openai/whisper-small"
DEFAULT_DATASET = "ai4bharat/Rural_Women_Bhojpuri"
TARGET_SR = 16000
DEFAULT_LANGUAGE = "Hindi"
DEFAULT_TASK = "transcribe"

TEXT_COLUMN_CANDIDATES = (
    "text",
    "transcript",
    "sentence",
    "normalized_text",
    "transcription",
)
AUDIO_COLUMN_CANDIDATES = ("audio", "speech", "input_audio")
DEFAULT_AUDIO_SEARCH_ROOTS = (
    Path("wav"),
    Path("training"),
    Path("training/wav"),
    Path("data"),
    Path("F:/bhojpuri-AI/.hf_cache/datasets"),
    Path("F:/bhojpuri-AI/.hf_cache/hub"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Bhojpuri speech data.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="models/bhojpuri-whisper-small")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--task", default=DEFAULT_TASK, choices=("transcribe", "translate"))
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--audio-column", default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--dataloader-num-workers", type=int, default=1)
    parser.add_argument("--preprocess-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--num-train-epochs", type=float, default=10.0)
    parser.add_argument("--max-train-steps", type=int, default=-1)
    parser.add_argument("--generation-max-length", type=int, default=225)
    parser.add_argument("--wer-eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--metrics-file", default="wer_report.txt")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--no-auto-resume", action="store_true")
    return parser.parse_args()


def pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def build_audio_index(search_roots: tuple[Path, ...]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.wav"):
            index.setdefault(path.name.lower(), path.resolve())
    return index


def load_audio_array(
    audio: dict[str, Any],
    *,
    dataset_root: Path,
    audio_index: dict[str, Path],
) -> tuple[np.ndarray, int]:
    raw_bytes = audio.get("bytes")
    byte_decode_error: Exception | None = None
    if raw_bytes:
        try:
            with io.BytesIO(raw_bytes) as buffer:
                audio_array, sample_rate = sf.read(buffer, dtype="float32", always_2d=False)
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)
            if sample_rate != TARGET_SR:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=TARGET_SR)
                sample_rate = TARGET_SR
            return audio_array, sample_rate
        except Exception as exc:
            byte_decode_error = exc

    audio_path = audio.get("path")
    if not audio_path:
        if byte_decode_error is not None:
            raise ValueError(
                "Embedded audio bytes could not be decoded and this row has no fallback audio path."
            ) from byte_decode_error
        raise ValueError("Audio row is missing both bytes and path data.")

    candidate = Path(audio_path)
    if candidate.exists():
        resolved = candidate.resolve()
    elif candidate.is_absolute():
        resolved = candidate
    else:
        resolved = (dataset_root / candidate).resolve()
        if not resolved.exists():
            resolved = audio_index.get(candidate.name.lower(), resolved)

    if not resolved.exists():
        if byte_decode_error is not None:
            raise ValueError(
                "Embedded audio bytes could not be decoded and the audio path "
                f"could not be resolved: '{audio_path}'."
            ) from byte_decode_error
        raise FileNotFoundError(
            f"Could not resolve audio file '{audio_path}'. "
            "Looked under the merged dataset folder and common audio roots."
        )

    audio_array, sample_rate = librosa.load(str(resolved), sr=TARGET_SR, mono=True)
    return audio_array, sample_rate


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def prepare_dataset_splits(
    dataset: DatasetDict,
    *,
    train_split: str | None,
    eval_split: str | None,
    seed: int,
    test_size: float,
) -> tuple[Any, Any]:
    if train_split and eval_split:
        return dataset[train_split], dataset[eval_split]
    if "train" in dataset and ("eval" in dataset or "validation" in dataset):
        eval_name = "eval" if "eval" in dataset else "validation"
        return dataset["train"], dataset[eval_name]
    if "train" in dataset:
        split = dataset["train"].train_test_split(test_size=test_size, seed=seed)
        return split["train"], split["test"]
    split_names = list(dataset.keys())
    if len(split_names) >= 2:
        return dataset[split_names[0]], dataset[split_names[1]]
    raise ValueError("Could not determine train/eval splits from dataset.")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class MetricsTextLogger(TrainerCallback):
    """Append training and evaluation metrics to a resumable text report."""

    def __init__(self, path: Path, *, append: bool) -> None:
        self.path = path
        self.append = append

    def on_train_begin(self, args, state, control, **kwargs):
        mode = "a" if self.append and self.path.exists() else "w"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open(mode, encoding="utf-8") as handle:
            handle.write(f"\n=== Training run: {datetime.now().isoformat(timespec='seconds')} ===\n")
            handle.write(f"output_dir={args.output_dir}\n")
            handle.write("step | epoch | metrics\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        metrics = ", ".join(f"{key}={value}" for key, value in sorted(logs.items()))
        with self.path.open("a", encoding="utf-8") as handle:
            epoch = "" if state.epoch is None else f"{state.epoch:.4f}"
            handle.write(f"{state.global_step} | {epoch} | {metrics}\n")


def append_metrics_report(path: Path, label: str, metrics: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        values = ", ".join(f"{key}={value}" for key, value in sorted(metrics.items()))
        handle.write(f"{label} | {values}\n")


def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    print(f"Device: {'cuda' if use_cuda else 'cpu'}")
    if use_cuda:
        device_name = torch.cuda.get_device_name(0)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA device: {device_name}")
        print(f"CUDA memory: {total_memory_gb:.1f} GB")
    print(f"Base model: {args.model}")
    print(f"Dataset: {args.dataset}")

    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language,
        task=args.task,
    )
    model.config.use_cache = False

    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        dataset = load_from_disk(str(dataset_path))
        print(f"Loaded dataset from disk: {dataset_path.resolve()}")
    else:
        dataset = load_dataset(args.dataset)
        print(f"Loaded dataset from Hugging Face Hub: {args.dataset}")

    train_ds, eval_ds = prepare_dataset_splits(
        dataset,
        train_split=args.train_split,
        eval_split=args.eval_split,
        seed=args.seed,
        test_size=args.test_size,
    )

    train_columns = train_ds.column_names
    audio_column = args.audio_column or pick_column(train_columns, AUDIO_COLUMN_CANDIDATES)
    text_column = args.text_column or pick_column(train_columns, TEXT_COLUMN_CANDIDATES)
    if audio_column is None:
        raise ValueError(f"Could not infer audio column from {train_columns}. Pass --audio-column explicitly.")
    if text_column is None:
        raise ValueError(f"Could not infer text column from {train_columns}. Pass --text-column explicitly.")

    print(f"Audio column: {audio_column}")
    print(f"Text column: {text_column}")

    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    print(f"Raw train rows selected: {len(train_ds)}")
    print(f"Raw eval rows selected: {len(eval_ds)}")

    train_ds = train_ds.cast_column(audio_column, Audio(decode=False))
    eval_ds = eval_ds.cast_column(audio_column, Audio(decode=False))

    audio_index = build_audio_index(DEFAULT_AUDIO_SEARCH_ROOTS)
    dataset_root = Path(args.dataset)
    max_label_length = getattr(model.config, "max_target_positions", 448)
    skipped_rows = {"train": 0, "eval": 0}
    reported_errors = 0

    def prepare_batch(batch: dict[str, Any], split_name: str) -> dict[str, Any]:
        nonlocal reported_errors
        input_features: list[Any] = []
        labels_list: list[list[int]] = []

        for audio, raw_text in zip(batch[audio_column], batch[text_column]):
            text = normalize_text(raw_text)
            audio_path = audio.get("path") if isinstance(audio, dict) else str(audio)
            if not text:
                skipped_rows[split_name] += 1
                continue
            try:
                audio_array, sample_rate = load_audio_array(
                    audio,
                    dataset_root=dataset_root,
                    audio_index=audio_index,
                )
                inputs = processor.feature_extractor(audio_array, sampling_rate=sample_rate)
                labels = processor.tokenizer(
                    text,
                    max_length=max_label_length,
                    truncation=True,
                ).input_ids
            except Exception as exc:
                skipped_rows[split_name] += 1
                if reported_errors < 5:
                    print(
                        f"Skipping {split_name} audio row '{audio_path}': "
                        f"{type(exc).__name__}: {exc}"
                    )
                    reported_errors += 1
                continue
            input_features.append(inputs.input_features[0])
            labels_list.append(labels)

        return {"input_features": input_features, "labels": labels_list}

    train_ds = train_ds.map(
        lambda batch: prepare_batch(batch, "train"),
        batched=True,
        batch_size=max(1, args.preprocess_batch_size),
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda batch: prepare_batch(batch, "eval"),
        batched=True,
        batch_size=max(1, args.preprocess_batch_size),
        remove_columns=eval_ds.column_names,
    )

    print(f"Skipped train rows: {skipped_rows['train']}")
    print(f"Skipped eval rows: {skipped_rows['eval']}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Eval samples: {len(eval_ds)}")

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_ids = np.asarray(pred_ids)
        label_ids = np.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer(reference=label_str, hypothesis=pred_str)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_train_steps,
        fp16=use_cuda,
        dataloader_num_workers=max(0, args.dataloader_num_workers),
        dataloader_pin_memory=use_cuda,
        dataloader_persistent_workers=(use_cuda and args.dataloader_num_workers > 0),
        logging_steps=max(1, args.logging_steps),
        eval_strategy="steps",
        eval_steps=max(1, args.wer_eval_steps),
        save_steps=max(1, args.save_steps),
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        compute_metrics=compute_metrics,
    )

    if args.resume_from_checkpoint:
        resume_path = Path(args.resume_from_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        resume_from_checkpoint: str | bool | None = str(resume_path)
        print(f"Resuming from explicit checkpoint: {resume_path}")
    elif not args.no_auto_resume:
        latest_checkpoint = find_latest_checkpoint(Path(args.output_dir))
        if latest_checkpoint is not None:
            resume_from_checkpoint = str(latest_checkpoint)
            print(f"Auto-resuming from latest checkpoint: {latest_checkpoint}")
        else:
            resume_from_checkpoint = None
            print("No checkpoint found. Starting a fresh training run.")
    else:
        resume_from_checkpoint = None
        print("Auto-resume disabled. Starting a fresh training run.")

    metrics_path = Path(args.metrics_file)
    if not metrics_path.is_absolute():
        metrics_path = Path(args.output_dir) / metrics_path
    trainer.add_callback(
        MetricsTextLogger(metrics_path, append=resume_from_checkpoint is not None)
    )
    print(f"Metrics report: {metrics_path.resolve()}")

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    append_metrics_report(metrics_path, "final_train", train_result.metrics)
    final_eval_metrics = trainer.evaluate()
    append_metrics_report(metrics_path, "final_eval", final_eval_metrics)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
