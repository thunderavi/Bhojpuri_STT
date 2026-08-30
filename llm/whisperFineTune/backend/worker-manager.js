import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class WorkerManager {
  constructor({ modelDir, language, task }) {
    this.modelDir = modelDir;
    this.language = language;
    this.task = task;
    this.child = null;
    this.ready = false;
    this.status = "stopped";
    this.device = "unknown";
    this.queue = Promise.resolve();
    this.pending = new Map();
    this.nextRequestId = 1;
    this.readyPromise = null;
  }

  start() {
    if (this.readyPromise) return this.readyPromise;

    this.status = "loading";
    this.readyPromise = new Promise((resolve, reject) => {
      const python = process.env.PYTHON_EXECUTABLE || this.defaultPythonPath();
      const workerPath = path.join(__dirname, "python", "whisper_worker.py");
      this.child = spawn(
        python,
        [workerPath, "--model-dir", this.modelDir, "--language", this.language, "--task", this.task],
        { cwd: __dirname, env: process.env, windowsHide: true },
      );

      const lines = createInterface({ input: this.child.stdout });
      lines.on("line", (line) => this.handleLine(line, resolve));
      this.child.stderr.on("data", (chunk) => {
        process.stderr.write(`[whisper-worker] ${chunk.toString()}`);
      });
      this.child.on("error", (error) => {
        this.status = "error";
        reject(error);
        this.rejectPending(error);
      });
      this.child.on("exit", (code, signal) => {
        this.ready = false;
        this.status = code === 0 ? "stopped" : "error";
        const error = new Error(`Whisper worker stopped (code=${code}, signal=${signal})`);
        this.rejectPending(error);
        if (!this.ready) reject(error);
      });
    });

    this.readyPromise.catch((error) => {
      console.error(`[whisper-worker] ${error.message}`);
    });
    return this.readyPromise;
  }

  defaultPythonPath() {
    const projectRoot = path.resolve(__dirname, "..", "..", "..");
    const projectPython = process.platform === "win32"
      ? path.join(projectRoot, ".venv", "Scripts", "python.exe")
      : path.join(projectRoot, ".venv", "bin", "python");
    if (fs.existsSync(projectPython)) return projectPython;
    return process.platform === "win32"
      ? path.join(__dirname, ".venv", "Scripts", "python.exe")
      : path.join(__dirname, ".venv", "bin", "python");
  }

  handleLine(line, resolveReady) {
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      console.error(`[whisper-worker] invalid output: ${line}`);
      return;
    }

    if (message.type === "ready") {
      this.ready = true;
      this.status = "ready";
      this.device = message.device || "unknown";
      resolveReady(message);
      return;
    }

    const pending = this.pending.get(message.id);
    if (!pending) return;
    this.pending.delete(message.id);
    if (message.type === "error") pending.reject(new Error(message.message));
    else pending.resolve(message);
  }

  request(payload) {
    const run = async () => {
      await this.start();
      return new Promise((resolve, reject) => {
        const id = String(this.nextRequestId++);
        const timeout = setTimeout(() => {
          this.pending.delete(id);
          reject(new Error("Whisper worker timed out."));
        }, 120000);

        this.pending.set(id, {
          resolve: (message) => {
            clearTimeout(timeout);
            resolve(message);
          },
          reject: (error) => {
            clearTimeout(timeout);
            reject(error);
          },
        });
        this.child.stdin.write(`${JSON.stringify({ ...payload, id })}\n`);
      });
    };

    const result = this.queue.then(run, run);
    this.queue = result.catch(() => undefined);
    return result;
  }

  rejectPending(error) {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }

  stop() {
    if (this.child) this.child.kill();
  }
}
