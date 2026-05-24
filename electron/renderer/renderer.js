const state = document.getElementById('runState');
const logs = document.getElementById('logs');
const start = document.getElementById('start');
const stop = document.getElementById('stop');
const cameraSelect = document.getElementById('cameraSelect');
const source = document.getElementById('source');
const confidence = document.getElementById('confidence');
const frameInterval = document.getElementById('frameInterval');
const pythonReady = document.getElementById('pythonReady');
const modelReady = document.getElementById('modelReady');
const bestTarget = document.getElementById('bestTarget');
const detectionList = document.getElementById('detectionList');
const cameraPreview = document.getElementById('cameraPreview');
const overlay = document.getElementById('overlay');
const overlayContext = overlay.getContext('2d');

let mediaStream = null;
let lastFrame = null;

function log(line) {
  logs.textContent = `${new Date().toLocaleTimeString()}  ${line}\n${logs.textContent}`.slice(0, 9000);
}

function updateStatus(status) {
  pythonReady.textContent = status.pythonReady ? 'ready' : 'missing';
  modelReady.textContent = status.modelReady ? 'ready' : 'missing';
  state.textContent = status.perceptionRunning ? 'Running' : 'Idle';
  state.classList.toggle('running', status.perceptionRunning);
}

function renderDetections(frame) {
  lastFrame = frame;
  const detections = frame.detections || [];
  detectionList.innerHTML = '';
  drawOverlay(frame);

  if (!detections.length) {
    bestTarget.textContent = 'No target';
    return;
  }

  const best = [...detections].sort((a, b) => b.confidence - a.confidence)[0];
  bestTarget.textContent = `${best.label} ${(best.confidence * 100).toFixed(1)}%`;

  for (const detection of detections.slice(0, 8)) {
    const item = document.createElement('div');
    item.className = 'detection';
    item.innerHTML = `<strong>${detection.label}</strong><span>${(detection.confidence * 100).toFixed(1)}%</span>`;
    detectionList.appendChild(item);
  }
}

function resizeOverlay() {
  const rect = cameraPreview.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  overlay.width = Math.max(1, Math.round(rect.width * scale));
  overlay.height = Math.max(1, Math.round(rect.height * scale));
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  overlayContext.setTransform(scale, 0, 0, scale, 0, 0);
  if (lastFrame) drawOverlay(lastFrame);
}

function drawOverlay(frame) {
  const rect = cameraPreview.getBoundingClientRect();
  overlayContext.clearRect(0, 0, rect.width, rect.height);

  if (!frame || !frame.width || !frame.height) return;

  const videoRatio = frame.width / frame.height;
  const viewRatio = rect.width / rect.height;
  let drawWidth = rect.width;
  let drawHeight = rect.height;
  let offsetX = 0;
  let offsetY = 0;

  if (viewRatio > videoRatio) {
    drawWidth = rect.height * videoRatio;
    offsetX = (rect.width - drawWidth) / 2;
  } else {
    drawHeight = rect.width / videoRatio;
    offsetY = (rect.height - drawHeight) / 2;
  }

  const scaleX = drawWidth / frame.width;
  const scaleY = drawHeight / frame.height;

  for (const detection of frame.detections || []) {
    const [x1, y1, x2, y2] = detection.box;
    const x = offsetX + x1 * scaleX;
    const y = offsetY + y1 * scaleY;
    const width = (x2 - x1) * scaleX;
    const height = (y2 - y1) * scaleY;
    const label = `${detection.label} ${(detection.confidence * 100).toFixed(0)}%`;

    overlayContext.lineWidth = 3;
    overlayContext.strokeStyle = '#37f06f';
    overlayContext.fillStyle = 'rgba(55, 240, 111, 0.16)';
    overlayContext.strokeRect(x, y, width, height);
    overlayContext.fillRect(x, y, width, height);

    overlayContext.font = '13px ui-monospace, SFMono-Regular, Menlo, monospace';
    const textWidth = overlayContext.measureText(label).width + 12;
    const labelY = Math.max(0, y - 24);
    overlayContext.fillStyle = '#37f06f';
    overlayContext.fillRect(x, labelY, textWidth, 22);
    overlayContext.fillStyle = '#061009';
    overlayContext.fillText(label, x + 6, labelY + 15);
  }
}

async function loadCameras() {
  try {
    await navigator.mediaDevices.getUserMedia({ video: true, audio: false }).then((stream) => {
      stream.getTracks().forEach((track) => track.stop());
    });
  } catch (error) {
    log(`camera permission: ${error.message}`);
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === 'videoinput');
  cameraSelect.innerHTML = '';

  cameras.forEach((camera, index) => {
    const option = document.createElement('option');
    option.value = camera.deviceId;
    option.textContent = camera.label || `Camera ${index}`;
    option.dataset.index = String(index);
    cameraSelect.appendChild(option);
  });

  if (cameras.length) {
    cameraSelect.selectedIndex = 0;
    source.value = cameraSelect.selectedOptions[0].dataset.index || '0';
    await startPreview(cameraSelect.value);
  } else {
    log('no camera found');
  }
}

async function startPreview(deviceId) {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }

  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });

  cameraPreview.srcObject = mediaStream;
  await cameraPreview.play();
  resizeOverlay();
}

cameraSelect.addEventListener('change', async () => {
  const option = cameraSelect.selectedOptions[0];
  source.value = option?.dataset.index || '0';
  await startPreview(cameraSelect.value);
});

start.addEventListener('click', async () => {
  try {
    log('starting YOLO26 MLX perception');
    const status = await window.hornsby.startPerception({
      source: source.value,
      confidence: confidence.value,
      frameInterval: frameInterval.value,
    });
    updateStatus(status);
  } catch (error) {
    log(`error: ${error.message}`);
  }
});

stop.addEventListener('click', async () => {
  const status = await window.hornsby.stopPerception();
  updateStatus(status);
  log('perception stopped');
});

window.hornsby.onPerceptionLine((line) => {
  log(line);
  try {
    const parsed = JSON.parse(line);
    renderDetections(parsed);
  } catch {
    // Non-JSON model startup lines stay in logs only.
  }
});

window.hornsby.onPerceptionError((line) => log(line.trim()));
window.hornsby.onPerceptionExit(({ code, signal }) => log(`perception exited code=${code} signal=${signal}`));
window.hornsby.onPerceptionStatus(updateStatus);

window.hornsby.status().then(updateStatus);
window.addEventListener('resize', resizeOverlay);
cameraPreview.addEventListener('loadedmetadata', resizeOverlay);
loadCameras();
