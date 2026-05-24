const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('hornsby', {
  status: () => ipcRenderer.invoke('app:status'),
  startPerception: (options) => ipcRenderer.invoke('perception:start', options),
  stopPerception: () => ipcRenderer.invoke('perception:stop'),
  onPerceptionLine: (callback) => ipcRenderer.on('perception:line', (_event, line) => callback(line)),
  onPerceptionError: (callback) => ipcRenderer.on('perception:error', (_event, line) => callback(line)),
  onPerceptionExit: (callback) => ipcRenderer.on('perception:exit', (_event, data) => callback(data)),
  onPerceptionStatus: (callback) => ipcRenderer.on('perception:status', (_event, data) => callback(data)),
});
