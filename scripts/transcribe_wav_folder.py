from __future__ import annotations

import os
import argparse
import csv
from pathlib import Path

HF_CACHE_DIR = Path("F:/bhojpuri-AI/.hf_cache")
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))

import librosa
import torch
from transformers import pipeline


DEFAULT_MODEL = "openai/whisper-small"
TARGET_SR = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe every .wav file in a folder with Whisper."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("wav"),
        help="Folder containing .wav files (default: wav).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs") / "wav_transcripts.csv",
        help="Where to save transcripts as CSV.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model to use (default: {DEFAULT_MODEL}).",
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
    return parser.parse_args()


def load_audio(audio_path: Path) -> tuple[list[float], int]:
    audio, sample_rate = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    return audio, sample_rate


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {args.input_dir}")

    wav_files = sorted(args.input_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {args.input_dir}")

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else -1
    torch_dtype = torch.float16 if use_cuda else torch.float32

    print(f"Loading model: {args.model}")
    print(f"Device: {'cuda' if use_cuda else 'cpu'}")
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        dtype=torch_dtype,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "transcript"])
        writer.writeheader()

        for index, wav_file in enumerate(wav_files, start=1):
            audio, sample_rate = load_audio(wav_file)
            result = asr(
                {"array": audio, "sampling_rate": sample_rate},
                generate_kwargs={"task": args.task, "language": args.language},
            )
            transcript = result["text"].strip()
            writer.writerow({"file": wav_file.name, "transcript": transcript})
            print(f"[{index}/{len(wav_files)}] {wav_file.name}: {transcript}")

    print(f"Saved transcripts to: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
