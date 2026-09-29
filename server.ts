/**
 * server.ts - Full-stack server running Express with Vite dev middleware and proxying to FastAPI backend.
 */

import express from 'express';
import { spawn } from 'child_process';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
const FASTAPI_PORT = 8001;

// 1. Launch FastAPI Python backend process
let fastApiProcess: any = null;

function startFastAPI() {
  console.log('[Server] Starting FastAPI backend on port', FASTAPI_PORT, '...');
  const pythonPath = '/app/applet/.python_packages:/app/applet:' + (process.env.PYTHONPATH || '');
  fastApiProcess = spawn('python3', ['-m', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', String(FASTAPI_PORT)], {
    cwd: __dirname,
    stdio: 'inherit',
    detached: false,
    env: {
      ...process.env,
      PYTHONPATH: pythonPath,
    }
  });

  fastApiProcess.on('error', (err: any) => {
    console.error('[Server] Failed to start FastAPI process:', err);
  });

  fastApiProcess.on('exit', (code: any, signal: any) => {
    console.log(`[Server] FastAPI process exited with code ${code}, signal ${signal}`);
  });
}

// Start FastAPI
startFastAPI();

// 2. Reverse proxy `/api/*` to FastAPI
app.use('/api', (req, res) => {
  const options: http.RequestOptions = {
    hostname: '127.0.0.1',
    port: FASTAPI_PORT,
    path: req.originalUrl,
    method: req.method,
    headers: {
      ...req.headers,
      host: `127.0.0.1:${FASTAPI_PORT}`,
    },
  };

  const proxyReq = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode || 500, proxyRes.headers);
    proxyRes.pipe(res, { end: true });
  });

  proxyReq.on('error', (err) => {
    console.error('[Proxy Error]', err.message);
    if (!res.headersSent) {
      res.status(502).json({
        error: 'Bad Gateway',
        message: 'Could not connect to FastAPI backend on port ' + FASTAPI_PORT + '. Please wait a moment while the ML backend initializes.',
      });
    }
  });

  if (['GET', 'HEAD', 'DELETE', 'OPTIONS'].includes(req.method || '') || !req.readable) {
    proxyReq.end();
  } else {
    req.pipe(proxyReq, { end: true });
  }
});

// 3. Static assets from public folder (background images, icons)
app.use(express.static(path.resolve(__dirname, 'public')));

// 4. Vite middleware for React Frontend
async function setupVite() {
  const isProduction = process.env.NODE_ENV === 'production';
  
  if (!isProduction) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[Server] AI What-If Season Planner running at http://0.0.0.0:${PORT}`);
  });
}

setupVite().catch(console.error);

// Clean up child process on exit
process.on('SIGINT', () => {
  if (fastApiProcess) fastApiProcess.kill();
  process.exit();
});

process.on('SIGTERM', () => {
  if (fastApiProcess) fastApiProcess.kill();
  process.exit();
});
