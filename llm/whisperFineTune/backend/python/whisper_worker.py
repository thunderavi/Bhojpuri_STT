from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import sys
from pathlib import Path


def emit(payload: dict) -> None:
    # ASCII JSON escapes keep the line protocol safe on Windows cp1252 consoles.
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def log(message: str) -> None:
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Whisper GPU worker.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--language", default="Hindi")
    parser.add_argument("--task", default="transcribe", choices=("transcribe", "translate"))
    return parser.parse_args()


def decode_audio(audio_base64: str, sample_rate: int):
    import librosa
    import numpy as np

    raw = base64.b64decode(audio_base64)
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if audio.size == 0:
        raise ValueError("Audio payload is empty.")
    if sample_rate != 16000:
        audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
    return audio


def main() -> None:
    args = parse_args()

    import torch
    from transformers import pipeline

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_dir}")

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else torch.float32
    device_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    log(f"Loading model from {args.model_dir}")
    log(f"Using device: {device_name}")

    # Keep model-loading output off stdout because stdout is the JSON protocol.
    with contextlib.redirect_stdout(sys.stderr):
        asr = pipeline(
            "automatic-speech-recognition",
            model=str(args.model_dir),
            device=device,
            dtype=dtype,
        )

    emit({
        "type": "ready",
        "device": device_name,
        "cuda": use_cuda,
        "modelDir": str(args.model_dir),
    })

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = request.get("id")
            request_type = request.get("type")

            if request_type == "ping":
                emit({"type": "pong", "id": request_id})
                continue
            if request_type != "transcribe":
                raise ValueError(f"Unsupported request type: {request_type}")

            sample_rate = int(request.get("sampleRate", 16000))
            audio = decode_audio(request["audioBase64"], sample_rate)
            result = asr(
                {"array": audio, "sampling_rate": 16000},
                generate_kwargs={"task": args.task, "language": args.language},
            )
            emit({
                "type": "transcript",
                "id": request_id,
                "text": result.get("text", "").strip(),
                "final": bool(request.get("final", True)),
            })
        except Exception as error:
            log(f"Request failed: {error}")
            emit({
                "type": "error",
                "id": request.get("id") if "request" in locals() and isinstance(request, dict) else None,
                "message": str(error),
            })


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"Worker failed to start: {error}")
        emit({"type": "error", "message": str(error)})
        raise
