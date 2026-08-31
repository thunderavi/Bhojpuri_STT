import "dotenv/config";
import express from "express";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import { WebSocketServer } from "ws";
import { WorkerManager } from "./worker-manager.js";
import { StorageManager } from "./storage.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const backendRoot = __dirname;

const port = Number.parseInt(process.env.PORT || "3000", 10);
const host = process.env.HOST || "127.0.0.1";
const language = process.env.LANGUAGE || "Hindi";
const task = process.env.TASK || "transcribe";
const publicDir = path.join(backendRoot, "public");

// ── Available models registry ──────────────────────────────────────────────
const AVAILABLE_MODELS = {
  "full-finetune": {
    key: "full-finetune",
    label: "Full Fine-Tune",
    description: "Whisper-Small fully fine-tuned on Bhojpuri (14,900 steps)",
    wer: "40.58%",
    dir: path.resolve(backendRoot, process.env.MODEL_DIR || ".."),
  },
  "lora": {
    key: "lora",
    label: "LoRA Fine-Tune",
    description: "Whisper-Small + LoRA adapters merged (r=8, q+v, 3,500 steps)",
    wer: "38.91%",
    dir: path.resolve(backendRoot, process.env.LORA_MODEL_DIR || "../../../models/LORAmodel/lora-merged-final"),
  },
};

let activeModelKey = "full-finetune";
let modelDir = AVAILABLE_MODELS[activeModelKey].dir;

const worker = new WorkerManager({ modelDir, language, task });
const storage = new StorageManager();

const app = express();
app.disable("x-powered-by");
app.use(express.json({ limit: "1mb" }));
app.use(express.static(publicDir));

function modelStatus() {
  const requiredFiles = ["config.json", "model.safetensors", "tokenizer.json"];
  const missingFiles = requiredFiles.filter(
    (fileName) => !fs.existsSync(path.join(worker.modelDir, fileName)),
  );

  return {
    modelDirectory: worker.modelDir,
    activeModel: activeModelKey,
    activeModelLabel: AVAILABLE_MODELS[activeModelKey]?.label,
    activeModelWer: AVAILABLE_MODELS[activeModelKey]?.wer,
    ready: missingFiles.length === 0,
    missingFiles,
    device: worker.device,
    workerStatus: worker.status,
    language,
    task,
  };
}

function healthPayload() {
  return {
    ok: true,
    service: "whisper-finetune-local-backend",
    phase: 5,
    timestamp: new Date().toISOString(),
    model: modelStatus(),
    storage: storage.snapshot(),
  };
}

app.get("/health", (_request, response) => { response.json(healthPayload()); });
app.get("/api/health", (_request, response) => { response.json(healthPayload()); });
app.get("/api/model-info", (_request, response) => { response.json(modelStatus()); });
app.get("/api/storage-status", (_request, response) => { response.json(storage.snapshot()); });

// ── Models list endpoint ───────────────────────────────────────────────────
app.get("/api/models", (_request, response) => {
  const models = Object.values(AVAILABLE_MODELS).map((m) => ({
    key: m.key,
    label: m.label,
    description: m.description,
    wer: m.wer,
    active: m.key === activeModelKey,
    filesReady: fs.existsSync(path.join(m.dir, "model.safetensors")),
  }));
  response.json({ models, activeModel: activeModelKey });
});

// ── Switch model endpoint ──────────────────────────────────────────────────
app.post("/api/switch-model", async (request, response) => {
  const { modelKey } = request.body || {};
  if (!modelKey || !AVAILABLE_MODELS[modelKey]) {
    return response.status(400).json({ error: `Unknown model key: "${modelKey}". Available: ${Object.keys(AVAILABLE_MODELS).join(", ")}` });
  }
  if (modelKey === activeModelKey) {
    return response.json({ ok: true, message: "Already using this model.", activeModel: activeModelKey });
  }

  const targetModel = AVAILABLE_MODELS[modelKey];
  if (!fs.existsSync(path.join(targetModel.dir, "model.safetensors"))) {
    return response.status(503).json({ error: `Model files not found at: ${targetModel.dir}` });
  }

  console.log(`[server] Switching model: ${activeModelKey} → ${modelKey}`);
  console.log(`[server] New model dir: ${targetModel.dir}`);

  try {
    activeModelKey = modelKey;
    await worker.switchModel(targetModel.dir);
    console.log(`[server] Model switched successfully to ${modelKey} on ${worker.device}`);
    response.json({
      ok: true,
      activeModel: activeModelKey,
      label: targetModel.label,
      wer: targetModel.wer,
      device: worker.device,
    });
  } catch (error) {
    console.error(`[server] Model switch failed: ${error.message}`);
    activeModelKey = "full-finetune"; // rollback
    response.status(500).json({ error: `Model switch failed: ${error.message}` });
  }
});

app.post("/api/sessions", async (request, response) => {
  try {
    const session = await storage.createSession({
      language: request.body?.language,
      task: request.body?.task,
      model: request.body?.model || activeModelKey,
      metadata: request.body?.metadata,
    });
    response.status(201).json(session);
  } catch (error) {
    response.status(503).json({ error: error.message });
  }
});

app.get("/api/sessions", async (request, response) => {
  try {
    response.json(await storage.listSessions(request.query.limit));
  } catch (error) {
    response.status(503).json({ error: error.message });
  }
});

app.get("/api/sessions/:sessionId", async (request, response) => {
  try {
    const session = await storage.getSession(request.params.sessionId);
    if (!session) return response.status(404).json({ error: "Session not found." });
    response.json(session);
  } catch (error) {
    response.status(503).json({ error: error.message });
  }
});

app.post("/api/sessions/:sessionId/close", async (request, response) => {
  try {
    response.json(await storage.closeSession(request.params.sessionId));
  } catch (error) {
    response.status(503).json({ error: error.message });
  }
});

app.post("/api/sessions/:sessionId/utterances", async (request, response) => {
  try {
    const utterance = await storage.saveUtterance(request.params.sessionId, request.body || {});
    response.status(201).json(utterance);
  } catch (error) {
    response.status(503).json({ error: error.message });
  }
});

app.get("/api/config", (_request, response) => {
  response.json({
    language: process.env.LANGUAGE || "Hindi",
    task: process.env.TASK || "transcribe",
    realtime: worker.status === "ready",
    storageEnabled: process.env.STORAGE_ENABLED === "true",
  });
});

app.get("/", (_request, response) => {
  response.sendFile(path.join(publicDir, "index.html"));
});

const server = http.createServer(app);
const websocketServer = new WebSocketServer({ server, path: "/ws/transcribe" });

server.on("error", (error) => {
  if (error.code === "EADDRINUSE") {
    console.error(`Port ${port} is already in use. Stop the existing backend or change PORT in .env.`);
  } else {
    console.error(`Backend server error: ${error.message}`);
  }
  process.exitCode = 1;
});

websocketServer.on("error", (error) => {
  console.error(`WebSocket server error: ${error.message}`);
});

async function persistResult(requestPayload, result) {
  if (!storage.enabled || !requestPayload.sessionId || !result.text) return;
  if (!result.final && process.env.STORE_PARTIALS !== "true") return;
  try {
    await storage.saveUtterance(requestPayload.sessionId, {
      text: result.text,
      isFinal: result.final,
      startMs: requestPayload.startMs,
      endMs: requestPayload.endMs,
      confidence: result.confidence,
    });
  } catch (error) {
    console.error(`[storage] Could not save utterance: ${error.message}`);
  }
}

websocketServer.on("connection", (socket) => {
  socket.send(JSON.stringify({
    type: "connected",
    phase: 2,
    workerStatus: worker.status,
    message: "WebSocket connected.",
  }));

  socket.on("message", (message, isBinary) => {
    if (isBinary) {
      const requestPayload = {
        type: "transcribe",
        audioBase64: Buffer.from(message).toString("base64"),
        sampleRate: 16000,
      };
      const t0 = Date.now();
      worker.request(requestPayload).then(async (result) => {
        const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
        console.log(`[transcribe] [${activeModelKey.toUpperCase()}] (${elapsed}s) -> "${result.text}"`);
        await persistResult(requestPayload, result);
        if (socket.readyState === 1) socket.send(JSON.stringify(result));
      }).catch((error) => {
        console.error(`[transcribe] Error: ${error.message}`);
        if (socket.readyState === 1) socket.send(JSON.stringify({ type: "error", message: error.message }));
      });
      return;
    }

    try {
      const payload = JSON.parse(message.toString());
      if (payload.type === "ping") {
        socket.send(JSON.stringify({ type: "pong", timestamp: Date.now() }));
      } else if (payload.type === "transcribe") {
        const t0 = Date.now();
        worker.request(payload).then(async (result) => {
          const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
          console.log(`[transcribe] [${activeModelKey.toUpperCase()}] (${elapsed}s) -> "${result.text}"`);
          await persistResult(payload, result);
          if (socket.readyState === 1) socket.send(JSON.stringify(result));
        }).catch((error) => {
          console.error(`[transcribe] Error: ${error.message}`);
          if (socket.readyState === 1) socket.send(JSON.stringify({ type: "error", message: error.message }));
        });
      }
    } catch {
      socket.send(JSON.stringify({ type: "error", message: "Invalid JSON message." }));
    }
  });
});

server.listen(port, host, () => {
  console.log(`Whisper backend running at http://${host}:${port}`);
  console.log(`Active model: ${activeModelKey} → ${worker.modelDir}`);
  console.log(`Available models: ${Object.keys(AVAILABLE_MODELS).join(", ")}`);
  worker.start().then(() => {
    console.log(`Whisper worker ready on ${worker.device}.`);
  }).catch((error) => {
    console.error(`Whisper worker failed: ${error.message}`);
  });
  storage.init().then(() => {
    console.log(`Storage ready: ${storage.provider}.`);
  }).catch((error) => {
    console.error(`Storage unavailable: ${error.message}`);
  });
});

function shutdown(signal) {
  console.log(`Received ${signal}; shutting down.`);
  websocketServer.close();
  worker.stop();
  storage.close().finally(() => server.close(() => process.exit(0)));
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
