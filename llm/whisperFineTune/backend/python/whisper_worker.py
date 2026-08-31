"""
whisper_worker.py
=================
Persistent Whisper GPU worker with:
  - Silero VAD (neural Voice Activity Detection) — filters noise before Whisper
  - Dual-model GPU cache — instant hot-switching, no reload delay
  - Repeat/hallucination suppressor

Protocol: JSON lines on stdin/stdout. Stderr → terminal log.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import sys
from pathlib import Path


def emit(payload: dict) -> None:
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
    """Decode base64 PCM-16 audio to float32 numpy array at 16 kHz."""
    import librosa
    import numpy as np
    raw = base64.b64decode(audio_base64)
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if audio.size == 0:
        raise ValueError("Audio payload is empty.")
    if sample_rate != 16000:
        audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
    return audio


def deduplicate_text(text: str) -> str:
    """Collapse repeated phrases produced by Whisper hallucinations."""
    import re
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    for n in range(1, 9):
        pattern = r"\b((?:\S+\s+){" + str(n - 1) + r"}\S+)(?:\s+\1){2,}\b"
        cleaned = re.sub(pattern, r"\1", cleaned, flags=re.UNICODE)
    tokens = cleaned.split()
    if len(tokens) >= 4:
        for chunk_len in range(2, 6):
            if len(tokens) >= chunk_len * 2:
                for i in range(len(tokens) - chunk_len * 2 + 1):
                    if " ".join(tokens[i:i+chunk_len]) == " ".join(tokens[i+chunk_len:i+chunk_len*2]):
                        tokens = tokens[:i+chunk_len] + tokens[i+chunk_len*2:]
                        cleaned = " ".join(tokens)
                        break
    return cleaned.strip()


def load_silero_vad():
    """Load Silero VAD from torch hub (cached locally after first download).
    Always runs on CPU — it is tiny (~2 MB) and fast enough for real-time use.
    """
    import torch
    log("Loading Silero VAD neural voice detector...")
    with contextlib.redirect_stdout(sys.stderr):
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
    vad_model = vad_model.to("cpu")   # Always CPU — avoids device-mismatch with CUDA audio tensors
    vad_model.eval()
    get_speech_timestamps = vad_utils[0]
    log("Silero VAD loaded OK (CPU).")
    return vad_model, get_speech_timestamps


def run_vad(audio, vad_model, get_speech_timestamps, torch, threshold: float = 0.50):
    """
    Returns True if Silero VAD detects real human speech in the audio.
    Audio tensor is always moved to CPU to match the VAD model device.
    """
    tensor = torch.from_numpy(audio).float().cpu()   # force CPU — VAD model is always on CPU

    with torch.no_grad():
        timestamps = get_speech_timestamps(
            tensor,
            vad_model,
            sampling_rate=16000,
            threshold=threshold,
            min_speech_duration_ms=180,
            min_silence_duration_ms=100,
        )
    return len(timestamps) > 0, timestamps


def load_whisper_pipeline(model_dir: str, device: int, dtype):
    """Load and return a HF ASR pipeline for the given model directory.
    Uses `dtype=` which is the correct parameter for transformers 5.x.
    """
    from transformers import pipeline
    log(f"Loading ASR pipeline: {model_dir}")
    with contextlib.redirect_stdout(sys.stderr):
        return pipeline(
            "automatic-speech-recognition",
            model=model_dir,
            device=device,
            dtype=dtype,
        )


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    from transformers import pipeline  # noqa: F401 (imported in helper)

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_dir}")

    use_cuda = torch.cuda.is_available()
    torch_device_idx = 0 if use_cuda else -1
    torch_device_str = "cuda:0" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    device_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"

    log(f"Device: {device_name}")
    log(f"Loading Whisper model from: {args.model_dir}")

    # ── Load Silero VAD (neural VAD — always CPU) ─────────────────────────────
    try:
        vad_model, get_speech_timestamps = load_silero_vad()
        use_vad = True
    except Exception as vad_err:
        log(f"WARNING: Silero VAD unavailable ({vad_err}). Falling back to energy gate only.")
        use_vad = False
        vad_model = None
        get_speech_timestamps = None

    # ── In-Memory Dual-Model GPU Cache ────────────────────────────────────────
    loaded_pipelines: dict = {}
    initial_key = str(args.model_dir)

    loaded_pipelines[initial_key] = load_whisper_pipeline(initial_key, torch_device_idx, dtype)
    current_model_key = initial_key

    emit({
        "type": "ready",
        "device": device_name,
        "cuda": use_cuda,
        "modelDir": initial_key,
        "vad": use_vad,
    })
    log(f"Worker ready. VAD: {'Silero Neural' if use_vad else 'Energy Gate Fallback'}")

    # ── Main stdin loop ───────────────────────────────────────────────────────
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = request.get("id")
            request_type = request.get("type")

            # ── ping ─────────────────────────────────────────────────────────
            if request_type == "ping":
                emit({"type": "pong", "id": request_id})
                continue

            # ── switch_model ─────────────────────────────────────────────────
            if request_type == "switch_model":
                new_model_dir = Path(request.get("model_dir", ""))
                if not new_model_dir.exists():
                    raise FileNotFoundError(f"Target model directory does not exist: {new_model_dir}")
                target_key = str(new_model_dir)

                if target_key in loaded_pipelines:
                    log(f"⚡ Instant cache hit — switching to: {target_key}")
                    current_model_key = target_key
                    emit({"type": "ready", "id": request_id, "device": device_name,
                          "cuda": use_cuda, "modelDir": target_key, "cached": True})
                    continue

                log(f"Loading & caching model: {target_key}")
                try:
                    loaded_pipelines[target_key] = load_whisper_pipeline(target_key, torch_device_idx, dtype)
                    current_model_key = target_key
                except Exception as oom_err:
                    log(f"OOM — clearing cache and retrying: {oom_err}")
                    loaded_pipelines.clear()
                    if use_cuda:
                        torch.cuda.empty_cache()
                    loaded_pipelines[target_key] = load_whisper_pipeline(target_key, torch_device_idx, dtype)
                    current_model_key = target_key

                log(f"Model cached & active: {target_key}")
                emit({"type": "ready", "id": request_id, "device": device_name,
                      "cuda": use_cuda, "modelDir": target_key, "cached": False})
                continue

            # ── transcribe ───────────────────────────────────────────────────
            if request_type != "transcribe":
                raise ValueError(f"Unsupported request type: {request_type}")

            sample_rate = int(request.get("sampleRate", 16000))
            audio = decode_audio(request["audioBase64"], sample_rate)

            # ── Stage 1: Basic energy gate (fast, pre-VAD) ───────────────────
            rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size > 0 else 0.0
            peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
            if audio.size < 16000 * 0.40 or rms < 0.008 or peak < 0.025:
                emit({"type": "transcript", "id": request_id, "text": "",
                      "final": bool(request.get("final", True))})
                continue

            # ── Stage 2: Silero VAD — only pass real human speech ────────────
            if use_vad:
                has_speech, _ = run_vad(audio, vad_model, get_speech_timestamps, torch, threshold=0.50)
                if not has_speech:
                    log(f"[VAD] No human speech detected (rms={rms:.4f}) — skipping")
                    emit({"type": "transcript", "id": request_id, "text": "",
                          "final": bool(request.get("final", True))})
                    continue

            # ── Stage 3: Whisper ASR ─────────────────────────────────────────
            asr = loaded_pipelines[current_model_key]
            result = asr(
                {"array": audio, "sampling_rate": 16000},
                generate_kwargs={
                    "task": args.task,
                    "language": args.language,
                    "no_repeat_ngram_size": 3,
                    "repetition_penalty": 1.15,
                    "temperature": 0.0,
                    "max_new_tokens": 128,
                },
            )
            raw_text = result.get("text", "").strip()

            # ── Stage 4: Post-process — deduplicate & filter hallucinations ───
            clean_text = deduplicate_text(raw_text)

            # Known Whisper hallucination phrases when background noise is near-silence
            NOISE_PHRASES = [
                "इसके लिए प्रदेश", "प्रदेश है", "प्रदेश में", "प्रदेश कर",
                "मुझे लगता", "मुझे प्रदेश", "मुंहूँ", "धन्यवाद",
                "सब्सक्राइब", "लाइक करें", "शेयर करें",
                "thank you", "thanks for watching", "subscribe",
            ]
            if any(phrase in clean_text for phrase in NOISE_PHRASES) and rms < 0.040:
                log(f"[HALLUCINATION SUPPRESSED] rms={rms:.4f} text={clean_text!r}")
                clean_text = ""

            log(f"[ASR] model={Path(current_model_key).name} rms={rms:.3f} → {clean_text!r}")
            emit({"type": "transcript", "id": request_id, "text": clean_text,
                  "final": bool(request.get("final", True))})

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
