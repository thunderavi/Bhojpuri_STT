# Bhojpuri Whisper Local Backend

This package will provide a local real-time Bhojpuri transcription GUI and developer API.

## Phases 1 through 4

The package now includes the Node.js server foundation, model-file checks, health endpoints, a persistent Python Whisper worker, browser microphone capture, voice activity handling, an animated waveform, and live transcript display.

## Start on Windows

From this directory, run PowerShell:

```powershell
.\setup.ps1
.\start.ps1
```

Then open <http://localhost:3000>.

The model is expected one level above this folder, in `whisperFineTune`.

The first startup loads the model on the GPU and may take around one minute. Later transcription requests reuse the loaded worker.

If the project-level `.venv` is not present, `setup.ps1` creates `backend\.venv` and installs the Python worker dependencies there.

## Optional storage

Storage is disabled by default. To enable SQLite, edit `.env`:

```env
STORAGE_ENABLED=true
DATABASE_PROVIDER=sqlite
DATABASE_URL=./data/transcripts.db
```

Other providers use these settings:

```env
# PostgreSQL
DATABASE_PROVIDER=postgres
DATABASE_URL=postgresql://user:password@localhost:5432/bhojpuri

# MySQL
DATABASE_PROVIDER=mysql
DATABASE_URL=mysql://user:password@localhost:3306/bhojpuri

# MongoDB
DATABASE_PROVIDER=mongodb
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=bhojpuri_whisper
```

Restart the backend after changing `.env`. Transcripts are saved only when a developer sends a `sessionId` with a WebSocket transcription request or uses the session API. Raw audio is not stored.

## Endpoints

- `GET /health`
- `GET /api/health`
- `GET /api/model-info`
- `GET /api/storage-status`
- `WS /ws/transcribe`

The WebSocket accepts either raw 16-bit little-endian PCM audio at 16 kHz as a binary message, or JSON like:

```json
{
  "type": "transcribe",
  "audioBase64": "...",
  "sampleRate": 16000,
  "final": true
}
```
