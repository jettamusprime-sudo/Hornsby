const state = document.getElementById('runState');
const logs = document.getElementById('logs');
const start = document.getElementById('start');
const stop = document.getElementById('stop');
const source = document.getElementById('source');
const confidence = document.getElementById('confidence');
const pythonReady = document.getElementById('pythonReady');
const modelReady = document.getElementById('modelReady');
const bestTarget = document.getElementById('bestTarget');
const detectionList = document.getElementById('detectionList');

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
  const detections = frame.detections || [];
  detectionList.innerHTML = '';

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

start.addEventListener('click', async () => {
  try {
    log('starting YOLO26 MLX perception');
    const status = await window.hornsby.startPerception({
      source: source.value,
      confidence: confidence.value,
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
