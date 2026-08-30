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
const modelDir = path.resolve(backendRoot, process.env.MODEL_DIR || "..");
const publicDir = path.join(backendRoot, "public");
const language = process.env.LANGUAGE || "Hindi";
const task = process.env.TASK || "transcribe";
const worker = new WorkerManager({ modelDir, language, task });
const storage = new StorageManager();

const app = express();
app.disable("x-powered-by");
app.use(express.json({ limit: "1mb" }));
app.use(express.static(publicDir));

function modelStatus() {
  const requiredFiles = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
  ];
  const missingFiles = requiredFiles.filter(
    (fileName) => !fs.existsSync(path.join(modelDir, fileName)),
  );

  return {
    modelDirectory: modelDir,
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

app.get("/health", (_request, response) => {
  response.json(healthPayload());
});

app.get("/api/health", (_request, response) => {
  response.json(healthPayload());
});

app.get("/api/model-info", (_request, response) => {
  response.json(modelStatus());
});

app.get("/api/storage-status", (_request, response) => {
  response.json(storage.snapshot());
});

app.post("/api/sessions", async (request, response) => {
  try {
    const session = await storage.createSession({
      language: request.body?.language,
      task: request.body?.task,
      model: request.body?.model || "whisperFineTune",
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
  socket.send(
    JSON.stringify({
      type: "connected",
      phase: 2,
      workerStatus: worker.status,
      message: "WebSocket connected.",
    }),
  );

  socket.on("message", (message, isBinary) => {
    if (isBinary) {
      const requestPayload = {
        type: "transcribe",
        audioBase64: Buffer.from(message).toString("base64"),
        sampleRate: 16000,
      };
      worker.request(requestPayload).then(async (result) => {
        await persistResult(requestPayload, result);
        if (socket.readyState === 1) socket.send(JSON.stringify(result));
      }).catch((error) => {
        if (socket.readyState === 1) socket.send(JSON.stringify({ type: "error", message: error.message }));
      });
      return;
    }

    try {
      const payload = JSON.parse(message.toString());
      if (payload.type === "ping") {
        socket.send(JSON.stringify({ type: "pong", timestamp: Date.now() }));
      } else if (payload.type === "transcribe") {
        worker.request(payload).then(async (result) => {
          await persistResult(payload, result);
          if (socket.readyState === 1) socket.send(JSON.stringify(result));
        }).catch((error) => {
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
  console.log(`Model directory: ${modelDir}`);
  console.log(`Model files ready: ${modelStatus().ready}`);
  console.log("Starting persistent Whisper worker...");
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
