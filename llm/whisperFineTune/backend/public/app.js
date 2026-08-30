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

function setStatus(element, label, kind) {
  element.textContent = label;
  element.className = `status status-${kind}`;
}

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
    setStatus(modelStatus, workerReady ? "GPU ready" : workerLoading ? "Loading" : "Unavailable", workerReady ? "ready" : workerLoading ? "loading" : "error");
    device.textContent = workerReady ? payload.model.device : payload.model.workerStatus;
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
    body: JSON.stringify({ model: "whisperFineTune", language: language.textContent, task: "transcribe" }),
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

function resetSpeechBuffer() {
  speechFrames = [];
  speechSamples = 0;
  silenceMilliseconds = 0;
  speechDetected = false;
  lastPartialSentSamples = 0;
}

function sendSpeechChunk(isFinal) {
  if (!websocket || websocket.readyState !== WebSocket.OPEN || speechSamples < sourceSampleRate * 0.35) return;
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
  const speaking = rms > 0.018;

  if (speaking) {
    speechDetected = true;
    silenceMilliseconds = 0;
    speechFrames.push(frame);
    speechSamples += frame.length;
  } else if (speechDetected) {
    speechFrames.push(frame);
    speechSamples += frame.length;
    silenceMilliseconds += frameMilliseconds;
  }

  if (speechDetected && speechSamples >= sourceSampleRate * 2.4 && speechSamples - lastPartialSentSamples >= sourceSampleRate * 1.2 && !partialRequestInFlight) sendSpeechChunk(false);
  if (speechDetected && silenceMilliseconds >= 700) sendSpeechChunk(true);
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
          if (payload.text) finalTranscript.textContent += `${finalTranscript.textContent ? " " : ""}${payload.text}`;
          partialTranscript.textContent = "";
          partialRequestInFlight = false;
          setStatus(connectionStatus, "Listening", "ready");
        } else {
          partialTranscript.textContent = payload.text || "Listening...";
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
    microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    audioContext = new AudioContext({ sampleRate: 16000 });
    await audioContext.audioWorklet.addModule("/audio-processor.js");
    sourceSampleRate = audioContext.sampleRate;
    microphoneSource = audioContext.createMediaStreamSource(microphoneStream);
    microphoneNode = new AudioWorkletNode(audioContext, "microphone-processor");
    muteGain = audioContext.createGain();
    muteGain.gain.value = 0;
    microphoneNode.port.onmessage = (event) => handleAudioFrame(event.data);
    microphoneSource.connect(microphoneNode).connect(muteGain).connect(audioContext.destination);
    listening = true;
    toggleButton.textContent = "Stop microphone";
    listenState.textContent = "Listening for Bhojpuri speech";
    permissionNote.textContent = "Speak normally. Pause briefly to finish a sentence.";
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

loadStatus();
setInterval(loadStatus, 2000);
drawWaveform();
