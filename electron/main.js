const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

let windowRef = null;
let perceptionProcess = null;

const rootDir = path.resolve(__dirname, '..');
const pythonPath = path.join(rootDir, '.venv', 'bin', 'python');
const perceptionScript = path.join(rootDir, 'hornsby_ai', 'physical_ai.py');
const modelPath = path.join(rootDir, 'models', 'yolo26n.npz');

function createWindow() {
  windowRef = new BrowserWindow({
    width: 1180,
    height: 760,
    minWidth: 920,
    minHeight: 620,
    title: 'Hornsby Control',
    backgroundColor: '#0c0f0d',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  windowRef.loadFile(path.join(__dirname, 'renderer', 'index.html'));
}

function send(channel, payload) {
  if (windowRef && !windowRef.isDestroyed()) {
    windowRef.webContents.send(channel, payload);
  }
}

function environmentStatus() {
  return {
    pythonReady: fs.existsSync(pythonPath),
    modelReady: fs.existsSync(modelPath),
    pythonPath,
    modelPath,
    perceptionRunning: Boolean(perceptionProcess),
  };
}

function stopPerception() {
  if (!perceptionProcess) return;
  perceptionProcess.kill('SIGTERM');
  perceptionProcess = null;
  send('perception:status', environmentStatus());
}

ipcMain.handle('app:status', () => environmentStatus());

ipcMain.handle('perception:start', (_event, options = {}) => {
  if (perceptionProcess) {
    return environmentStatus();
  }

  if (!fs.existsSync(pythonPath)) {
    throw new Error('Python environment not found. Run scripts/setup_yolo26_mlx_macos.sh first.');
  }

  if (!fs.existsSync(modelPath)) {
    throw new Error('YOLO26 MLX model not found. Run scripts/setup_yolo26_mlx_macos.sh first.');
  }

  const source = String(options.source || '0');
  const confidence = String(options.confidence || '0.25');

  perceptionProcess = spawn(
    pythonPath,
    [
      perceptionScript,
      '--model',
      modelPath,
      '--source',
      source,
      '--conf',
      confidence,
    ],
    {
      cwd: rootDir,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    }
  );

  perceptionProcess.stdout.on('data', (chunk) => {
    const lines = chunk.toString().split(/\r?\n/).filter(Boolean);
    for (const line of lines) {
      send('perception:line', line);
    }
  });

  perceptionProcess.stderr.on('data', (chunk) => {
    send('perception:error', chunk.toString());
  });

  perceptionProcess.on('exit', (code, signal) => {
    send('perception:exit', { code, signal });
    perceptionProcess = null;
    send('perception:status', environmentStatus());
  });

  send('perception:status', environmentStatus());
  return environmentStatus();
});

ipcMain.handle('perception:stop', () => {
  stopPerception();
  return environmentStatus();
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  stopPerception();
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
