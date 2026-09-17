const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let serverProcess = null;
let mainWindow = null;

function getServerPath() {
  // In packaged app, the server exe is in resources/server/
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'server', 'sim_server.exe');
  }
  // In development, use the Python script directly
  return null; // Will use python command instead
}

function startServer(serialPort) {
  return new Promise((resolve) => {
    if (serverProcess) {
      // On Windows, use taskkill to kill the process tree
      const pid = serverProcess.pid;
      try {
        require('child_process').execSync(`taskkill /PID ${pid} /T /F`, { stdio: 'ignore' });
      } catch (_) {}
      serverProcess = null;
      // Wait for port to be freed
      setTimeout(() => { _doStartServer(serialPort); resolve(); }, 2000);
    } else {
      _doStartServer(serialPort);
      resolve();
    }
  });
}

function _doStartServer(serialPort) {
  const serverExe = getServerPath();
  const args = [];
  if (serialPort) {
    args.push('--serial-port', serialPort);
  }

  if (serverExe) {
    // Packaged: run the bundled exe
    serverProcess = spawn(serverExe, args, { stdio: 'pipe' });
  } else {
    // Development: run via python from the venv
    const appRoot = path.resolve(__dirname, '..');
    const projectRoot = path.resolve(appRoot, '..');
    const pythonPath = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
    const scriptPath = path.join(projectRoot, 'sim_server.py');
    console.log(`[server] Starting: ${pythonPath} ${scriptPath} ${args.join(' ')}`);
    serverProcess = spawn(pythonPath, [scriptPath, ...args], { 
      stdio: 'pipe',
      cwd: projectRoot,
    });
  }

  serverProcess.stdout.on('data', (data) => {
    console.log(`[server] ${data.toString().trim()}`);
  });
  serverProcess.stderr.on('data', (data) => {
    console.log(`[server:err] ${data.toString().trim()}`);
  });
  serverProcess.on('error', (err) => {
    console.error(`[server] Failed to start: ${err.message}`);
    serverProcess = null;
  });
  serverProcess.on('close', (code) => {
    console.log(`[server] exited with code ${code}`);
    serverProcess = null;
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    title: 'Acoustic Visualiser',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  // Load the built Vite output
  mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
}

// IPC: renderer can request server start/restart with a port
ipcMain.handle('start-server', async (event, serialPort) => {
  await startServer(serialPort || null);
  return { status: 'started', serialPort: serialPort || null };
});

ipcMain.handle('stop-server', () => {
  if (serverProcess) {
    serverProcess.kill();
    serverProcess = null;
  }
  return { status: 'stopped' };
});

ipcMain.handle('get-server-status', () => {
  return { running: serverProcess !== null };
});

app.whenReady().then(() => {
  // Start server without serial port by default (simulator-only)
  startServer(null);
  // Give server time to boot before loading the page
  setTimeout(() => {
    createWindow();
  }, 3000);
});

app.on('window-all-closed', () => {
  if (serverProcess) {
    serverProcess.kill();
    serverProcess = null;
  }
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
