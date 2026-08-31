const serviceStatus = document.querySelector("#service-status");
const modelStatus = document.querySelector("#model-status");
const connectionStatus = document.querySelector("#connection-status");
const device = document.querySelector("#device");
const language = document.querySelector("#language");
const storage = document.querySelector("#storage");
const listenState = document.querySelector("#listen-state");
const permissionNote = document.querySelector("#permission-note");
const toggleButton = document.querySelector("#toggle-listening");
const clearButton = document.querySelector("#clear-transcript");
const saveSessionCheckbox = document.querySelector("#save-session");
const refreshHistoryButton = document.querySelector("#refresh-history");
const historyList = document.querySelector("#history-list");
const finalTranscript = document.querySelector("#final-transcript");
const partialTranscript = document.querySelector("#partial-transcript");
const canvas = document.querySelector("#waveform");
const canvasContext = canvas.getContext("2d");

// ── Model selector elements ────────────────────────────────────────────────
const modelSelect = document.querySelector("#model-select");
const switchModelBtn = document.querySelector("#switch-model-btn");
const switchStatus = document.querySelector("#switch-status");
const activeModelLabel = document.querySelector("#active-model-label");
const activeModelWer = document.querySelector("#active-model-wer");
const activeModelDesc = document.querySelector("#active-model-desc");
const currentModelName = document.querySelector("#current-model-name");
const currentModelWer = document.querySelector("#current-model-wer");

let audioContext;
let microphoneStream;
let microphoneSource;
let microphoneNode;
let muteGain;
let websocket;
let listening = false;
let sourceSampleRate = 16000;
let speechFrames = [];
let speechSamples = 0;
let silenceMilliseconds = 0;
let speechDetected = false;
let partialRequestInFlight = false;
let lastPartialSentSamples = 0;
let waveformLevel = 0;
let sessionId = null;
let storageEnabled = false;
let isSwitchingModel = false;
let availableModels = [];
let activeModelKey = null;

function setStatus(element, label, kind) {
  element.textContent = label;
  element.className = `status status-${kind}`;
}

// ── Model selector logic ───────────────────────────────────────────────────
async function loadModels() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    availableModels = data.models;
    activeModelKey = data.activeModel;

    // Populate dropdown
    modelSelect.replaceChildren();
    for (const model of availableModels) {
      const option = document.createElement("option");
      option.value = model.key;
      option.textContent = `${model.label} — WER ${model.wer}${model.filesReady ? "" : " ⚠ files missing"}`;
      option.disabled = !model.filesReady;
      if (model.active) option.selected = true;
      modelSelect.appendChild(option);
    }
    modelSelect.disabled = false;
    switchModelBtn.disabled = false;

    updateActiveModelInfo(activeModelKey);
  } catch (error) {
    modelSelect.replaceChildren();
    const opt = document.createElement("option");
    opt.textContent = "Failed to load models";
    modelSelect.appendChild(opt);
    console.error("Could not load models:", error);
  }
}

function updateActiveModelInfo(modelKey) {
  const model = availableModels.find((m) => m.key === modelKey);
  if (!model) return;
  activeModelKey = modelKey;
  activeModelLabel.textContent = model.label;
  activeModelWer.textContent = `WER ${model.wer}`;
  activeModelDesc.textContent = model.description;
  currentModelName.textContent = model.label;
  currentModelWer.textContent = model.wer;
}

async function switchModel() {
  const selectedKey = modelSelect.value;
  if (!selectedKey || selectedKey === activeModelKey) return;
  if (isSwitchingModel) return;

  if (listening) {
    stopListening();
    permissionNote.textContent = "Microphone stopped while switching model.";
  }

  isSwitchingModel = true;
  modelSelect.disabled = true;
  switchModelBtn.disabled = true;
  toggleButton.disabled = true;
  switchStatus.style.display = "";
  setStatus(switchStatus, "Switching...", "loading");
  setStatus(modelStatus, "Loading new model", "loading");

  try {
    const response = await fetch("/api/switch-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ modelKey: selectedKey }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || `HTTP ${response.status}`);
    }

    const result = await response.json();
    updateActiveModelInfo(selectedKey);
    setStatus(switchStatus, "Switched ✓", "ready");
    setStatus(modelStatus, "GPU ready", "ready");
    device.textContent = result.device || "GPU";
    permissionNote.textContent = `Now using: ${result.label} (WER ${result.wer})`;

    setTimeout(() => {
      switchStatus.style.display = "none";
    }, 3000);
  } catch (error) {
    setStatus(switchStatus, "Failed", "error");
    setStatus(modelStatus, "Error", "error");
    permissionNote.textContent = `Model switch failed: ${error.message}`;
    console.error("Switch model error:", error);
    // restore select to active model
    for (const option of modelSelect.options) {
      option.selected = option.value === activeModelKey;
    }
  } finally {
    isSwitchingModel = false;
    modelSelect.disabled = false;
    switchModelBtn.disabled = false;
    toggleButton.disabled = false;
  }
}

switchModelBtn.addEventListener("click", switchModel);

// ── Status polling ─────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    const workerReady = payload.model.workerStatus === "ready";
    const workerLoading = payload.model.workerStatus === "loading";
    const wasStorageEnabled = storageEnabled;
    storageEnabled = payload.storage.enabled && payload.storage.status === "ready";

    setStatus(serviceStatus, "Online", "ready");

    if (!isSwitchingModel) {
      setStatus(modelStatus, workerReady ? "GPU ready" : workerLoading ? "Loading" : "Unavailable", workerReady ? "ready" : workerLoading ? "loading" : "error");
      device.textContent = workerReady ? payload.model.device : payload.model.workerStatus;
      if (payload.model.activeModel && payload.model.activeModel !== activeModelKey) {
        updateActiveModelInfo(payload.model.activeModel);
      }
      if (payload.model.activeModelLabel) currentModelName.textContent = payload.model.activeModelLabel;
      if (payload.model.activeModelWer) currentModelWer.textContent = payload.model.activeModelWer;
    }

    language.textContent = payload.model.language;
    storage.textContent = storageEnabled ? payload.storage.provider : "Disabled";
    saveSessionCheckbox.disabled = !storageEnabled;
    if (!storageEnabled) saveSessionCheckbox.checked = false;
    if (storageEnabled && !wasStorageEnabled) refreshHistory();
  } catch (error) {
    setStatus(serviceStatus, "Offline", "error");
    setStatus(modelStatus, "Unavailable", "error");
    device.textContent = "Backend unavailable";
    saveSessionCheckbox.disabled = true;
    console.error(error);
  }
}

async function refreshHistory() {
  if (!storageEnabled) {
    historyList.textContent = "Storage is disabled.";
    return;
  }
  try {
    const response = await fetch("/api/sessions?limit=20");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const sessions = await response.json();
    historyList.replaceChildren();
    if (!sessions.length) {
      historyList.textContent = "No saved sessions yet.";
      return;
    }
    for (const session of sessions) {
      const item = document.createElement("div");
      item.className = "history-item";
      item.textContent = `${new Date(session.startedAt || session.started_at).toLocaleString()} - ${session.language || "Hindi"}`;
      historyList.appendChild(item);
    }
  } catch (error) {
    historyList.textContent = `History unavailable: ${error.message}`;
  }
}

async function beginSession() {
  if (!saveSessionCheckbox.checked || !storageEnabled) return;
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: activeModelKey || "whisperFineTune", language: language.textContent, task: "transcribe" }),
  });
  if (!response.ok) throw new Error((await response.json()).error || "Could not create storage session.");
  sessionId = (await response.json()).id;
}

async function endSession() {
  if (!sessionId) return;
  const closingSessionId = sessionId;
  sessionId = null;
  await fetch(`/api/sessions/${closingSessionId}/close`, { method: "POST" }).catch(() => undefined);
  refreshHistory();
}

function drawWaveform() {
  const width = canvas.width;
  const height = canvas.height;
  canvasContext.clearRect(0, 0, width, height);
  canvasContext.fillStyle = "#111719";
  canvasContext.fillRect(0, 0, width, height);
  canvasContext.strokeStyle = "#263235";
  canvasContext.beginPath();
  canvasContext.moveTo(0, height / 2);
  canvasContext.lineTo(width, height / 2);
  canvasContext.stroke();

  const bars = 64;
  const gap = 5;
  const barWidth = (width - gap * (bars - 1)) / bars;
  for (let index = 0; index < bars; index += 1) {
    const variation = 0.35 + Math.abs(Math.sin(index * 1.7 + performance.now() / 330));
    const barHeight = Math.max(3, Math.min(height * 0.72, waveformLevel * height * variation * 2.5));
    const x = index * (barWidth + gap);
    canvasContext.fillStyle = listening ? "#72d6a3" : "#40504d";
    canvasContext.fillRect(x, (height - barHeight) / 2, barWidth, barHeight);
  }
  requestAnimationFrame(drawWaveform);
}

function downsample(samples, inputRate, outputRate) {
  if (inputRate === outputRate) return samples;
  const ratio = inputRate / outputRate;
  const outputLength = Math.round(samples.length / ratio);
  const output = new Float32Array(outputLength);
  for (let index = 0; index < outputLength; index += 1) {
    const position = index * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, samples.length - 1);
    const weight = position - left;
    output[index] = samples[left] * (1 - weight) + samples[right] * weight;
  }
  return output;
}

function floatToBase64Pcm(samples) {
  const bytes = new Uint8Array(samples.length * 2);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < samples.length; index += 1) {
    const value = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(index * 2, value < 0 ? value * 32768 : value * 32767, true);
  }
  let binary = "";
  const blockSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += blockSize) binary += String.fromCharCode(...bytes.subarray(offset, offset + blockSize));
  return btoa(binary);
}

function flattenFrames(frames, sampleCount) {
  const samples = new Float32Array(sampleCount);
  let offset = 0;
  for (const frame of frames) {
    samples.set(frame, offset);
    offset += frame.length;
  }
  return samples;
}

let noiseFloor = 0.015;
let voicedFrameCount = 0;
let totalSpeechFrameCount = 0;

function resetSpeechBuffer() {
  speechFrames = [];
  speechSamples = 0;
  silenceMilliseconds = 0;
  speechDetected = false;
  lastPartialSentSamples = 0;
  voicedFrameCount = 0;
  totalSpeechFrameCount = 0;
}

function sendSpeechChunk(isFinal) {
  if (!websocket || websocket.readyState !== WebSocket.OPEN || speechSamples < sourceSampleRate * 0.45) {
    if (isFinal) resetSpeechBuffer();
    return;
  }
  const allSamples = flattenFrames(speechFrames, speechSamples);
  const previewSamples = Math.round(sourceSampleRate * 4);
  const samples = isFinal ? allSamples : allSamples.subarray(Math.max(0, allSamples.length - previewSamples));
  const normalized = downsample(samples, sourceSampleRate, 16000);
  websocket.send(JSON.stringify({ type: "transcribe", audioBase64: floatToBase64Pcm(normalized), sampleRate: 16000, final: isFinal, sessionId }));
  partialRequestInFlight = !isFinal;
  if (!isFinal) lastPartialSentSamples = speechSamples;
  if (isFinal) setStatus(connectionStatus, "Processing", "loading");
  if (isFinal) resetSpeechBuffer();
}

function handleAudioFrame(frame) {
  let sum = 0;
  for (const sample of frame) sum += sample * sample;
  const rms = Math.sqrt(sum / frame.length);
  waveformLevel = Math.max(rms, waveformLevel * 0.86);
  const frameMilliseconds = (frame.length / sourceSampleRate) * 1000;

  // Track adaptive background noise floor during silence
  if (!speechDetected) {
    noiseFloor = noiseFloor * 0.95 + rms * 0.05;
  }

  // Active voice detection: Speech must be significantly louder than ambient noise floor
  const voiceThreshold = Math.max(0.040, noiseFloor * 2.6);
  const speaking = rms > voiceThreshold;

  if (speaking) {
    speechDetected = true;
    silenceMilliseconds = 0;
    speechFrames.push(frame);
    speechSamples += frame.length;
    voicedFrameCount += 1;
    totalSpeechFrameCount += 1;
  } else if (speechDetected) {
    speechFrames.push(frame);
    speechSamples += frame.length;
    totalSpeechFrameCount += 1;
    silenceMilliseconds += frameMilliseconds;
  }

  if (speechDetected && speechSamples >= sourceSampleRate * 2.4 && speechSamples - lastPartialSentSamples >= sourceSampleRate * 1.2 && !partialRequestInFlight) sendSpeechChunk(false);
  if (speechDetected && silenceMilliseconds >= 750) sendSpeechChunk(true);
}

function connectWebsocket() {
  return new Promise((resolve, reject) => {
    websocket = new WebSocket(`ws://${window.location.host}/ws/transcribe`);
    websocket.addEventListener("open", () => { setStatus(connectionStatus, "Connected", "ready"); resolve(); }, { once: true });
    websocket.addEventListener("error", () => reject(new Error("WebSocket connection failed.")), { once: true });
    websocket.addEventListener("message", (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "connected") return;
      if (payload.type === "error") { setStatus(connectionStatus, "Error", "error"); permissionNote.textContent = payload.message; return; }
      if (payload.type === "transcript") {
        if (payload.final) {
          if (payload.text) {
            const current = finalTranscript.textContent.trim();
            finalTranscript.textContent = current ? `${current} \n${payload.text}` : payload.text;
          }
          partialTranscript.textContent = "";
          partialRequestInFlight = false;
          setStatus(connectionStatus, "Listening", "ready");
        } else {
          partialTranscript.textContent = payload.text ? `... ${payload.text}` : "Listening...";
          partialRequestInFlight = false;
          setStatus(connectionStatus, "Preview", "loading");
        }
      }
    });
  });
}

async function startListening() {
  if (listening) return;
  try {
    await beginSession();
    await connectWebsocket();
    microphoneStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      }
    });
    audioContext = new AudioContext({ sampleRate: 16000 });
    await audioContext.audioWorklet.addModule("/audio-processor.js");
    sourceSampleRate = audioContext.sampleRate;
    microphoneSource = audioContext.createMediaStreamSource(microphoneStream);

    // ── Human Vocal Bandpass Filter (85 Hz to 3800 Hz) ─────────────────────
    // Cuts out low-frequency fan hum/AC rumble (0-80 Hz) and high-frequency static (3800 Hz+)
    const highpass = audioContext.createBiquadFilter();
    highpass.type = "highpass";
    highpass.frequency.value = 85;

    const lowpass = audioContext.createBiquadFilter();
    lowpass.type = "lowpass";
    lowpass.frequency.value = 3800;

    microphoneNode = new AudioWorkletNode(audioContext, "microphone-processor");
    muteGain = audioContext.createGain();
    muteGain.gain.value = 0;
    microphoneNode.port.onmessage = (event) => handleAudioFrame(event.data);

    // Connect: Mic -> Highpass -> Lowpass -> AudioWorklet -> Mute -> Destination
    microphoneSource.connect(highpass);
    highpass.connect(lowpass);
    lowpass.connect(microphoneNode);
    microphoneNode.connect(muteGain);
    muteGain.connect(audioContext.destination);

    listening = true;
    toggleButton.textContent = "Stop microphone";
    listenState.textContent = "Listening for Bhojpuri speech (Voice Isolated)";
    permissionNote.textContent = "Speak clearly into your microphone. Background noise is filtered.";
    setStatus(connectionStatus, "Listening", "ready");
  } catch (error) {
    await endSession();
    stopListening();
    setStatus(connectionStatus, "Not connected", "error");
    permissionNote.textContent = error.message;
  }
}

function stopListening() {
  if (speechDetected) sendSpeechChunk(true);
  listening = false;
  resetSpeechBuffer();
  microphoneNode?.disconnect();
  microphoneSource?.disconnect();
  muteGain?.disconnect();
  microphoneStream?.getTracks().forEach((track) => track.stop());
  audioContext?.close();
  websocket?.close();
  microphoneNode = null;
  microphoneSource = null;
  microphoneStream = null;
  audioContext = null;
  websocket = null;
  toggleButton.textContent = "Start microphone";
  listenState.textContent = "Ready to listen";
  setStatus(connectionStatus, "Idle", "loading");
  endSession();
}

toggleButton.addEventListener("click", () => (listening ? stopListening() : startListening()));
clearButton.addEventListener("click", () => { finalTranscript.textContent = ""; partialTranscript.textContent = "Your transcript will appear here."; });
refreshHistoryButton.addEventListener("click", refreshHistory);

// ── Init ───────────────────────────────────────────────────────────────────
loadStatus();
loadModels();
setInterval(loadStatus, 2000);
drawWaveform();
