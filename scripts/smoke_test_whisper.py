from __future__ import annotations

import os
import argparse
from pathlib import Path

HF_CACHE_DIR = Path("F:/bhojpuri-AI/.hf_cache")
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))

import librosa
import numpy as np
import torch
from transformers import pipeline


DEFAULT_MODEL = "openai/whisper-small"
TARGET_SR = 16000


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    return audio, sample_rate


def build_fallback_audio(seconds: float) -> tuple[np.ndarray, int]:
    num_samples = int(TARGET_SR * seconds)
    audio = np.zeros(num_samples, dtype=np.float32)
    return audio, TARGET_SR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test Whisper GPU inference for Bhojpuri fine-tuning prep."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Base Whisper model to load (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Optional path to an audio file. If omitted, a short silent clip is used.",
    )
    parser.add_argument(
        "--language",
        default="Hindi",
        help='Whisper language token to bias generation (default: "Hindi").',
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=("transcribe", "translate"),
        help='Whisper generation task (default: "transcribe").',
    )
    parser.add_argument(
        "--fallback-seconds",
        type=float,
        default=3.0,
        help="Duration of the synthetic fallback clip when --audio is omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else -1
    torch_dtype = torch.float16 if use_cuda else torch.float32

    print(f"Loading model: {args.model}")
    print(f"Device: {'cuda' if use_cuda else 'cpu'}")
    if use_cuda:
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        torch_dtype=torch_dtype,
    )

    if args.audio:
        if not args.audio.exists():
            raise FileNotFoundError(f"Audio file not found: {args.audio}")
        audio, sample_rate = load_audio(args.audio)
        source_label = str(args.audio)
    else:
        audio, sample_rate = build_fallback_audio(args.fallback_seconds)
        source_label = f"synthetic silence ({args.fallback_seconds:.1f}s)"

    result = asr(
        {"array": audio, "sampling_rate": sample_rate},
        generate_kwargs={"task": args.task, "language": args.language},
    )

    print(f"Source: {source_label}")
    print("Transcript:")
    print(result["text"].strip())


if __name__ == "__main__":
    main()
