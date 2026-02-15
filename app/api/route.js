// spawn process for Python model training
// added LRU cache for short term caching support

// for more info on LRU cache, check https://en.wikipedia.org/wiki/Cache_replacement_policies

import { spawn } from 'child_process';
import path from 'path';

const CACHE_MAX = 500;
const CACHE_TTL_MS = 5 * 60 * 1000;
const cache = new Map();

function getCached(key) {
  const entry = cache.get(key);
  if (!entry) return null;
  if (entry.expiresAt <= Date.now()) {
    cache.delete(key);
    return null;
  }
  cache.delete(key);
  cache.set(key, entry);
  return entry.value;
}

function setCached(key, value) {
  cache.set(key, { value, expiresAt: Date.now() + CACHE_TTL_MS });
  while (cache.size > CACHE_MAX) {
    const oldestKey = cache.keys().next().value;
    cache.delete(oldestKey);
  }
}

async function trySpawn(command, args) {
  return new Promise((resolve) => {
    const process = spawn(command, args);
    let result = '';
    let error = '';

    process.stdout.on('data', (data) => { result += data.toString(); });
    process.stderr.on('data', (data) => { error += data.toString(); });

    process.on('error', (err) => {
      resolve({ success: false, error: err.message });
    });

    process.on('close', (code) => {
      if (code === 0) {
        resolve({ success: true, data: result });
      } else {
        resolve({ success: false, error: error || `Exit code ${code}` });
      }
    });
  });
}

export async function POST(req) {
  const { expression } = await req.json();
  const scriptPath = path.join(process.cwd(), 'src/main.py');
  const payload = typeof expression === 'string' ? expression : JSON.stringify(expression);
  const cached = getCached(payload);
  if (cached) {
    return new Response(cached, {
      headers: { 'Content-Type': 'application/json' },
    });
  }

  // Try common python commands
  const commands = [
    path.join(process.cwd(), 'venv/bin/python'),
    'python', 'python3', 'py'
  ];
  let lastError = '';

  for (const cmd of commands) {
    const outcome = await trySpawn(cmd, [scriptPath, payload]);
    if (outcome.success) {
      setCached(payload, outcome.data);
      return new Response(outcome.data, {
        headers: { 'Content-Type': 'application/json' },
      });
    }
    lastError = outcome.error;
    // If the error is 'ENOENT', it means the command doesn't exist, try next one
    // If it's a different error (like a python script error), we should probably stop and report it
    if (!lastError.includes('ENOENT') && !lastError.includes('not found') && !lastError.includes('9009')) {
      break;
    }
  }

  return new Response(JSON.stringify({ 
    status: 'error', 
    message: `Failed to parse the text. Python error: ${lastError}` 
  }), {
    status: 500,
    headers: { 'Content-Type': 'application/json' },
  });
}
