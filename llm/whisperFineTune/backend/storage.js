import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function enabledFromEnv() {
  return String(process.env.STORAGE_ENABLED || "false").toLowerCase() === "true";
}

function resolveDatabasePath(value) {
  if (!value || value === ":memory:") return value || path.join(__dirname, "data", "transcripts.db");
  return path.isAbsolute(value) ? value : path.resolve(__dirname, value);
}

function jsonValue(value) {
  return value === undefined ? null : JSON.stringify(value);
}

export class StorageManager {
  constructor() {
    this.enabled = enabledFromEnv();
    this.provider = (process.env.DATABASE_PROVIDER || "sqlite").toLowerCase();
    this.status = this.enabled ? "connecting" : "disabled";
    this.error = null;
    this.client = null;
  }

  snapshot() {
    return {
      enabled: this.enabled,
      provider: this.provider,
      status: this.status,
      configured: this.enabled && Boolean(process.env.DATABASE_URL || process.env.MONGODB_URI),
      error: this.error,
    };
  }

  async init() {
    if (!this.enabled) return this.snapshot();
    try {
      if (this.provider === "sqlite") await this.initSqlite();
      else if (this.provider === "postgres" || this.provider === "postgresql") await this.initPostgres();
      else if (this.provider === "mysql") await this.initMysql();
      else if (this.provider === "mongodb" || this.provider === "mongo") await this.initMongo();
      else throw new Error(`Unsupported database provider: ${this.provider}`);
      this.status = "ready";
      this.error = null;
    } catch (error) {
      this.status = "error";
      this.error = error.message;
      throw error;
    }
    return this.snapshot();
  }

  async initSqlite() {
    const [nodeMajor, nodeMinor] = process.versions.node.split(".").map(Number);
    if (nodeMajor < 22 || (nodeMajor === 22 && nodeMinor < 5)) {
      throw new Error("SQLite storage requires Node.js 22.5 or newer.");
    }
    const { DatabaseSync } = await import("node:sqlite");
    const databasePath = resolveDatabasePath(process.env.DATABASE_URL);
    if (databasePath !== ":memory:") fs.mkdirSync(path.dirname(databasePath), { recursive: true });
    this.client = { kind: "sqlite", db: new DatabaseSync(databasePath) };
    this.client.db.exec(`
      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        language TEXT NOT NULL,
        task TEXT NOT NULL,
        model TEXT,
        metadata_json TEXT
      );
      CREATE TABLE IF NOT EXISTS utterances (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        sequence_number INTEGER NOT NULL,
        text TEXT NOT NULL,
        is_final INTEGER NOT NULL,
        start_ms INTEGER,
        end_ms INTEGER,
        confidence REAL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
      );
      CREATE INDEX IF NOT EXISTS idx_utterances_session ON utterances(session_id, sequence_number);
    `);
  }

  async initPostgres() {
    const { Pool } = await import("pg");
    const pool = new Pool({ connectionString: process.env.DATABASE_URL });
    await pool.query(`
      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        started_at TIMESTAMPTZ NOT NULL,
        ended_at TIMESTAMPTZ,
        language TEXT NOT NULL,
        task TEXT NOT NULL,
        model TEXT,
        metadata_json JSONB
      );
      CREATE TABLE IF NOT EXISTS utterances (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id),
        sequence_number INTEGER NOT NULL,
        text TEXT NOT NULL,
        is_final BOOLEAN NOT NULL,
        start_ms INTEGER,
        end_ms INTEGER,
        confidence DOUBLE PRECISION,
        created_at TIMESTAMPTZ NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_utterances_session ON utterances(session_id, sequence_number);
    `);
    this.client = { kind: "postgres", pool };
  }

  async initMysql() {
    const mysql = await import("mysql2/promise");
    const pool = mysql.createPool(process.env.DATABASE_URL);
    await pool.query(`
      CREATE TABLE IF NOT EXISTS sessions (
        id VARCHAR(36) PRIMARY KEY,
        started_at DATETIME(3) NOT NULL,
        ended_at DATETIME(3),
        language VARCHAR(64) NOT NULL,
        task VARCHAR(32) NOT NULL,
        model VARCHAR(255),
        metadata_json JSON
      );
      CREATE TABLE IF NOT EXISTS utterances (
        id VARCHAR(36) PRIMARY KEY,
        session_id VARCHAR(36) NOT NULL,
        sequence_number INT NOT NULL,
        text TEXT NOT NULL,
        is_final BOOLEAN NOT NULL,
        start_ms INT,
        end_ms INT,
        confidence DOUBLE,
        created_at DATETIME(3) NOT NULL,
        CONSTRAINT fk_utterances_session FOREIGN KEY (session_id) REFERENCES sessions(id),
        KEY idx_utterances_session (session_id, sequence_number)
      );
    `);
    this.client = { kind: "mysql", pool };
  }

  async initMongo() {
    const { MongoClient } = await import("mongodb");
    const client = new MongoClient(process.env.MONGODB_URI || process.env.DATABASE_URL);
    await client.connect();
    const databaseName = process.env.MONGODB_DATABASE || "bhojpuri_whisper";
    const db = client.db(databaseName);
    await db.collection("sessions").createIndex({ startedAt: -1 });
    await db.collection("utterances").createIndex({ sessionId: 1, sequenceNumber: 1 });
    this.client = { kind: "mongo", client, sessions: db.collection("sessions"), utterances: db.collection("utterances") };
  }

  assertReady() {
    if (!this.enabled) throw new Error("Storage is disabled. Set STORAGE_ENABLED=true in .env.");
    if (this.status !== "ready") throw new Error(this.error || "Storage is not ready.");
  }

  async createSession({ language, task, model, metadata } = {}) {
    this.assertReady();
    const session = {
      id: randomUUID(),
      startedAt: new Date(),
      endedAt: null,
      language: language || process.env.LANGUAGE || "Hindi",
      task: task || process.env.TASK || "transcribe",
      model: model || null,
      metadata: metadata || null,
    };
    if (this.client.kind === "sqlite") {
      this.client.db.prepare("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)").run(
        session.id, session.startedAt.toISOString(), null, session.language, session.task, session.model, jsonValue(session.metadata),
      );
    } else if (this.client.kind === "postgres") {
      await this.client.pool.query("INSERT INTO sessions (id, started_at, language, task, model, metadata_json) VALUES ($1, $2, $3, $4, $5, $6)", [session.id, session.startedAt, session.language, session.task, session.model, session.metadata]);
    } else if (this.client.kind === "mysql") {
      await this.client.pool.execute("INSERT INTO sessions (id, started_at, language, task, model, metadata_json) VALUES (?, ?, ?, ?, ?, ?)", [session.id, session.startedAt, session.language, session.task, session.model, jsonValue(session.metadata)]);
    } else {
      await this.client.sessions.insertOne(session);
    }
    return session;
  }

  async closeSession(sessionId) {
    this.assertReady();
    const endedAt = new Date();
    if (this.client.kind === "sqlite") this.client.db.prepare("UPDATE sessions SET ended_at = ? WHERE id = ?").run(endedAt.toISOString(), sessionId);
    else if (this.client.kind === "postgres") await this.client.pool.query("UPDATE sessions SET ended_at = $1 WHERE id = $2", [endedAt, sessionId]);
    else if (this.client.kind === "mysql") await this.client.pool.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", [endedAt, sessionId]);
    else await this.client.sessions.updateOne({ id: sessionId }, { $set: { endedAt } });
    return { sessionId, endedAt };
  }

  async saveUtterance(sessionId, { text, isFinal, startMs, endMs, confidence } = {}) {
    this.assertReady();
    if (!text) return null;
    const utterance = { id: randomUUID(), sessionId, sequenceNumber: await this.nextSequence(sessionId), text, isFinal: Boolean(isFinal), startMs: startMs ?? null, endMs: endMs ?? null, confidence: confidence ?? null, createdAt: new Date() };
    if (this.client.kind === "sqlite") this.client.db.prepare("INSERT INTO utterances VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)").run(utterance.id, sessionId, utterance.sequenceNumber, text, utterance.isFinal ? 1 : 0, utterance.startMs, utterance.endMs, utterance.confidence, utterance.createdAt.toISOString());
    else if (this.client.kind === "postgres") await this.client.pool.query("INSERT INTO utterances (id, session_id, sequence_number, text, is_final, start_ms, end_ms, confidence, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)", [utterance.id, sessionId, utterance.sequenceNumber, text, utterance.isFinal, utterance.startMs, utterance.endMs, utterance.confidence, utterance.createdAt]);
    else if (this.client.kind === "mysql") await this.client.pool.execute("INSERT INTO utterances (id, session_id, sequence_number, text, is_final, start_ms, end_ms, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [utterance.id, sessionId, utterance.sequenceNumber, text, utterance.isFinal, utterance.startMs, utterance.endMs, utterance.confidence, utterance.createdAt]);
    else await this.client.utterances.insertOne(utterance);
    return utterance;
  }

  async nextSequence(sessionId) {
    if (this.client.kind === "sqlite") return this.client.db.prepare("SELECT COALESCE(MAX(sequence_number), 0) + 1 AS next FROM utterances WHERE session_id = ?").get(sessionId).next;
    if (this.client.kind === "postgres") return Number((await this.client.pool.query("SELECT COALESCE(MAX(sequence_number), 0) + 1 AS next FROM utterances WHERE session_id = $1", [sessionId])).rows[0].next);
    if (this.client.kind === "mysql") return Number((await this.client.pool.query("SELECT COALESCE(MAX(sequence_number), 0) + 1 AS next FROM utterances WHERE session_id = ?", [sessionId]))[0][0].next);
    const last = await this.client.utterances.find({ sessionId }).sort({ sequenceNumber: -1 }).limit(1).next();
    return (last?.sequenceNumber || 0) + 1;
  }

  async listSessions(limit = 50) {
    this.assertReady();
    const safeLimit = Math.min(Math.max(Number(limit) || 50, 1), 200);
    if (this.client.kind === "sqlite") return this.client.db.prepare("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?").all(safeLimit);
    if (this.client.kind === "postgres") return (await this.client.pool.query("SELECT * FROM sessions ORDER BY started_at DESC LIMIT $1", [safeLimit])).rows;
    if (this.client.kind === "mysql") return (await this.client.pool.query("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", [safeLimit]))[0];
    return this.client.sessions.find({}).sort({ startedAt: -1 }).limit(safeLimit).toArray();
  }

  async getSession(sessionId) {
    this.assertReady();
    let session;
    let utterances;
    if (this.client.kind === "sqlite") {
      session = this.client.db.prepare("SELECT * FROM sessions WHERE id = ?").get(sessionId);
      utterances = this.client.db.prepare("SELECT * FROM utterances WHERE session_id = ? ORDER BY sequence_number").all(sessionId);
    } else if (this.client.kind === "postgres") {
      session = (await this.client.pool.query("SELECT * FROM sessions WHERE id = $1", [sessionId])).rows[0];
      utterances = (await this.client.pool.query("SELECT * FROM utterances WHERE session_id = $1 ORDER BY sequence_number", [sessionId])).rows;
    } else if (this.client.kind === "mysql") {
      session = (await this.client.pool.query("SELECT * FROM sessions WHERE id = ?", [sessionId]))[0][0];
      utterances = (await this.client.pool.query("SELECT * FROM utterances WHERE session_id = ? ORDER BY sequence_number", [sessionId]))[0];
    } else {
      session = await this.client.sessions.findOne({ id: sessionId });
      utterances = await this.client.utterances.find({ sessionId }).sort({ sequenceNumber: 1 }).toArray();
    }
    if (!session) return null;
    return { session, utterances };
  }

  async close() {
    if (!this.client) return;
    if (this.client.kind === "sqlite") this.client.db.close();
    else if (this.client.kind === "postgres" || this.client.kind === "mysql") await this.client.pool.end();
    else await this.client.client.close();
  }
}
