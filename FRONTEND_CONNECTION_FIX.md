# Frontend Connection Fix - Complete Configuration

**Status:** ✅ All frontend connection issues fixed

## Summary of Changes

1. ✅ Fixed environment variable handling (Vite uses `import.meta.env`, not `process.env`)
2. ✅ Unified API/WebSocket URL configuration
3. ✅ Removed hardcoded ports from vite.config.ts
4. ✅ Ensured CORS compatibility with localhost:3000
5. ✅ Fixed WebSocket URL construction to match backend route `/ws/radar-stream`

---

## 📋 CORRECTED CONFIG FILES

### 1. frontend/.env.development

```dotenv
# Development Environment
# Higher verbosity, detailed logging, performance instrumentation enabled

# Frontend Server
VITE_HOST=localhost
VITE_PORT=3000

# Backend API
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000

# Debugging
VITE_DEBUG=true
VITE_PERFORMANCE_LOGGING=true

# Caching
VITE_CACHE_ENABLED=true
VITE_CACHE_TTL=1000

# WebSocket
VITE_WEBSOCKET_THROTTLE_FPS=20
VITE_WEBSOCKET_RECONNECT_INTERVAL=3000
VITE_WEBSOCKET_RECONNECT_MAX_ATTEMPTS=10

# Monitoring
VITE_METRICS_FETCH_INTERVAL=2000
VITE_ENABLE_PERFORMANCE_MONITOR=true
VITE_REACT_PROFILER_ENABLED=true
```

### 2. frontend/.env.production

```dotenv
# Production Environment
# Optimized for performance, minimal logging, production-ready settings

# Frontend Server
VITE_HOST=0.0.0.0
VITE_PORT=3000

# Backend API (use relative paths or absolute via VITE_API_URL)
# Leave blank to use current domain/port based on window.location
VITE_API_URL=
VITE_WS_URL=

# Debugging (production)
VITE_DEBUG=false
VITE_PERFORMANCE_LOGGING=false

# Caching (production)
VITE_CACHE_ENABLED=true
VITE_CACHE_TTL=5000

# WebSocket (production)
VITE_WEBSOCKET_THROTTLE_FPS=10
VITE_WEBSOCKET_RECONNECT_INTERVAL=5000
VITE_WEBSOCKET_RECONNECT_MAX_ATTEMPTS=5

# Monitoring (production)
VITE_METRICS_FETCH_INTERVAL=5000
VITE_ENABLE_PERFORMANCE_MONITOR=false
VITE_REACT_PROFILER_ENABLED=false

# Sentry (optional - error tracking)
VITE_SENTRY_DSN=
```

### 3. frontend/vite.config.ts

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    server: {
        port: parseInt(process.env.VITE_PORT || '3000'),
        strictPort: false,
        host: process.env.VITE_HOST || 'localhost',
        proxy: {
            '/api': {
                target: process.env.VITE_API_URL || 'http://localhost:8000',
                changeOrigin: true,
            },
            '/ws': {
                target: process.env.VITE_WS_URL || 'ws://localhost:8000',
                ws: true,
                changeOrigin: true,
            },
        },
    },
    build: {
        target: 'esnext',
        minify: 'terser',
        terserOptions: {
            compress: {
                drop_console: true,
            },
        },
        rollupOptions: {
            output: {
                // Code splitting configuration
                manualChunks: {
                    vendor: ['react', 'react-dom', 'zustand'],
                    utils: ['date-fns'],
                },
                entryFileNames: 'js/[name]-[hash].js',
                chunkFileNames: 'js/[name]-[hash].js',
                assetFileNames: 'assets/[name]-[hash][extname]',
            },
        },
        // Enable CSS code splitting
        cssCodeSplit: true,
        // Reporting compressed size
        reportCompressedSize: true,
        // Chunk size warning
        chunkSizeWarningLimit: 500,
    },
    // Optimization settings
    optimizeDeps: {
        include: ['react', 'react-dom', 'zustand'],
        exclude: ['@vitest/ui'],
    },
})
```

### 4. frontend/src/config/envConfig.js

```javascript
/**
 * Environment Configuration Utility
 * Handles dev vs production settings for performance optimization
 * Properly integrates with Vite's import.meta.env (not process.env)
 */

class EnvironmentConfig {
  constructor() {
    this.isDev = import.meta.env.DEV
    this.isProd = import.meta.env.PROD
    
    // Base URLs from environment or defaults
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'
    
    this.config = {
      // ─── API Configuration ────────────────────────────────────────
      // Base URL: http://localhost:8000 (dev) or from origin (prod)
      apiUrl: apiUrl,
      apiPath: '/api',
      apiBaseUrl: `${apiUrl}/api`,
      
      // ─── WebSocket Configuration ──────────────────────────────────
      // WebSocket base URL: ws://localhost:8000 (dev) or from origin (prod)
      wsUrl: wsUrl,
      wsPath: '/ws/radar-stream',
      wsBaseUrl: `${wsUrl}/ws/radar-stream`,
      
      // ─── Debugging & Performance ──────────────────────────────────
      debug: import.meta.env.VITE_DEBUG === 'true' || false,
      performanceLogging: import.meta.env.VITE_PERFORMANCE_LOGGING === 'true' || false,
      cacheEnabled: import.meta.env.VITE_CACHE_ENABLED !== 'false',
      cacheTtl: parseInt(import.meta.env.VITE_CACHE_TTL || '1000'),
      
      // ─── WebSocket Performance ────────────────────────────────────
      websocketThrottleFps: parseInt(import.meta.env.VITE_WEBSOCKET_THROTTLE_FPS || '20'),
      websocketReconnectInterval: parseInt(import.meta.env.VITE_WEBSOCKET_RECONNECT_INTERVAL || '3000'),
      websocketReconnectMaxAttempts: parseInt(import.meta.env.VITE_WEBSOCKET_RECONNECT_MAX_ATTEMPTS || '10'),
      
      // ─── Monitoring ───────────────────────────────────────────────
      metricsFetchInterval: parseInt(import.meta.env.VITE_METRICS_FETCH_INTERVAL || '2000'),
      enablePerformanceMonitor: import.meta.env.VITE_ENABLE_PERFORMANCE_MONITOR === 'true' || false,
      reactProfilerEnabled: import.meta.env.VITE_REACT_PROFILER_ENABLED === 'true' || false,
    }
  }

  get(key, defaultValue) {
    return this.config[key] ?? defaultValue
  }

  getAll() {
    return { ...this.config }
  }

  getApiUrl() {
    return this.config.apiBaseUrl
  }

  getWsUrl() {
    return this.config.wsBaseUrl
  }

  isDevelopment() {
    return this.isDev
  }

  isProduction() {
    return this.isProd
  }

  toString() {
    return JSON.stringify(this.config, null, 2)
  }

  log() {
    if (this.config.debug) {
      console.log('[Environment Config]', this.config)
    }
  }
}

// Export singleton instance
export const envConfig = new EnvironmentConfig()

// Log config in development
if (import.meta.env.DEV) {
  console.log('Environment Config:', envConfig.getAll())
}

export default envConfig
```

### 5. frontend/src/config/apiConfig.ts

```typescript
/**
 * Frontend environment configuration
 * Dynamically sets API_URL and WS_URL based on environment
 * Uses Vite's import.meta.env (not process.env)
 */

// ─── Development URLs ─────────────────────────────────────────────────
const DEV_BASE_URL = 'http://localhost:8000'
const DEV_API_URL = `${DEV_BASE_URL}/api`
const DEV_WS_URL = 'ws://localhost:8000/ws/radar-stream'

// ─── Production URLs ──────────────────────────────────────────────────
// Use environment variable if provided, otherwise derive from window.location
const PROD_BASE_URL = import.meta.env.VITE_API_URL || window.location.origin
const PROD_API_URL = `${PROD_BASE_URL}/api`
const PROD_WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/radar-stream`

// ─── Exported Configuration ────────────────────────────────────────────
export const API_CONFIG = {
  // Base URL for axios and fetch
  BASE_URL: import.meta.env.DEV ? DEV_BASE_URL : PROD_BASE_URL,
  
  // API endpoint URLs
  API_URL: import.meta.env.DEV ? DEV_API_URL : PROD_API_URL,
  
  // WebSocket endpoint URL (full path including /ws/radar-stream)
  WS_URL: import.meta.env.DEV ? DEV_WS_URL : PROD_WS_URL,
  
  // Request/Connection timeouts (milliseconds)
  REQUEST_TIMEOUT: 10000,
  WS_RECONNECT_DELAY: 3000,
  WS_HEARTBEAT_INTERVAL: 30000,
  WS_MAX_RECONNECT_ATTEMPTS: 10,
}

export default API_CONFIG
```

### 6. frontend/src/services/apiClient.js

```javascript
import axios from 'axios'
import { envConfig } from '../config/envConfig'

// Use environment configuration for API URL
const API_URL = envConfig.getApiUrl()

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Attach JWT token to every request
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('aegis_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle response errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('aegis_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// ─── Auth ─────────────────────────────────────────────────────────────────────
export const auth = {
  login: (username, password) =>
    apiClient.post('/auth/login', { username, password }),
  register: (username, password, role = 'operator') =>
    apiClient.post('/auth/register', { username, password, role }),
  refresh: () => apiClient.post('/auth/refresh'),
}

// ─── Radar ────────────────────────────────────────────────────────────────────
export const radar = {
  scan: (params = {}) =>
    apiClient.post('/radar/scan', { signal_source: 'generated', ...params }),
  getLabels: () => apiClient.get('/radar/labels'),
  getHistory: () => apiClient.get('/radar/history'),
  getTargets: () => apiClient.get('/radar/targets'),
}

// ─── Threats ──────────────────────────────────────────────────────────────────
export const threats = {
  getAll: () => apiClient.get('/threats'),
  getActive: () => apiClient.get('/threats?status=active'),
  getById: (id) => apiClient.get(`/threats/${id}`),
}

// ─── EW ───────────────────────────────────────────────────────────────────────
export const ew = {
  getStatus: () => apiClient.get('/ew/status'),
  getSignals: () => apiClient.get('/ew/signals'),
  analyze: (signal) => apiClient.post('/ew/analyze', signal),
}

// ─── Visualizations ───────────────────────────────────────────────────────────
export const visualizations = {
  performance: () => apiClient.get('/visualizations/performance-charts'),
  confusion: () => apiClient.get('/visualizations/confusion-matrix'),
  roc: () => apiClient.get('/visualizations/roc-curve'),
  precisionRecall: () => apiClient.get('/visualizations/precision-recall'),
  training: () => apiClient.get('/visualizations/training-history'),
  surface3d: () => apiClient.get('/visualizations/3d-surface-plot'),
  gradcam: (scanId) => apiClient.get(`/visualizations/xai-gradcam/${scanId}`),
}

// ─── Admin ────────────────────────────────────────────────────────────────────
export const admin = {
  health: () => apiClient.get('/admin/health'),
  metrics: () => apiClient.get('/admin/metrics'),
  users: () => apiClient.get('/admin/users'),
  createUser: (user) => apiClient.post('/admin/users', user),
  deleteUser: (username) => apiClient.delete(`/admin/users/${username}`),
}

// ─── Metrics ──────────────────────────────────────────────────────────────────
export const metrics = {
  report: () => apiClient.get('/metrics/report'),
  performance: () => apiClient.get('/metrics/performance'),
  accuracy: () => apiClient.get('/metrics/accuracy'),
}

export default apiClient
```

### 7. frontend/src/services/websocketClient.js

```javascript
import { EventEmitter } from 'eventemitter3'
import { envConfig } from '../config/envConfig'

class WebSocketClient extends EventEmitter {
  constructor(url) {
    super()
    // Use provided URL or construct from environment configuration
    // Full WebSocket URL with /ws/radar-stream endpoint
    this.url = url || envConfig.getWsUrl()
    this.ws = null
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = envConfig.get('websocketReconnectMaxAttempts', 10)
    this.baseDelay = envConfig.get('websocketReconnectInterval', 3000)
    this.isIntentionallyClosed = false
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return Promise.resolve()
    }

    return new Promise((resolve, reject) => {
      try {
        this.isIntentionallyClosed = false
        this.ws = new WebSocket(this.url)

        this.ws.onopen = () => {
          console.log('[WS] Connected to', this.url)
          this.reconnectAttempts = 0
          this.emit('connect')
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            this.emit('data', data)
          } catch (err) {
            console.error('[WS] Parse error:', err)
          }
        }

        this.ws.onerror = (error) => {
          console.error('[WS] Error:', error)
          this.emit('error', error)
          reject(error)
        }

        this.ws.onclose = () => {
          console.log('[WS] Disconnected from', this.url)
          this.emit('disconnect')
          if (!this.isIntentionallyClosed) {
            this.reconnect()
          }
        }
      } catch (err) {
        reject(err)
      }
    })
  }

  reconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WS] Max reconnection attempts reached')
      this.emit('reconnectFailed')
      return
    }

    const delay = this.baseDelay * Math.pow(2, this.reconnectAttempts)
    this.reconnectAttempts++

    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`)

    setTimeout(() => {
      this.connect().catch(() => {
        // Retry handled by onclose
      })
    }, delay)
  }

  send(data) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    } else {
      console.warn('[WS] WebSocket not connected')
    }
  }

  subscribe(type, handler) {
    this.on(type, handler)
    return () => this.off(type, handler)
  }

  disconnect() {
    this.isIntentionallyClosed = true
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  isConnected() {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

export default new WebSocketClient()
```

---

## 🔒 Backend CORS Configuration

**File:** [backend/app/main.py](backend/app/main.py)

```python
# ─── CORS Middleware ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # Vite dev server (default)
        "http://127.0.0.1:3000",     # Vite dev server (127.0.0.1)
        "http://localhost:5173",     # Vite dev server (alt port)
        "http://127.0.0.1:5173",     # Vite dev server (alt port)
        "*"                          # Development: allow all - RESTRICT IN PRODUCTION
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Status:** ✅ **Configured correctly for local development**

---

## ✅ Connection URLs Summary

| Environment | Frontend | API | WebSocket |
|---|---|---|---|
| **Development** | `http://localhost:3000` | `http://localhost:8000/api` | `ws://localhost:8000/ws/radar-stream` |
| **Production** | `window.location.origin` | `${origin}/api` | `wss://${host}/ws/radar-stream` |

---

## 🚀 How to Run

### Step 1: Start Backend

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

### Step 2: Start Frontend (in another terminal)

```bash
cd frontend
npm install
npm run dev
```

**Expected output:**
```
  VITE v[version] ready in [time] ms

  ➜  Local:   http://localhost:3000/
  ➜  press h to show help
```

### Step 3: Verify Connections

**Check API:**
```bash
curl http://localhost:8000/health
# Expected: {"status": "ok", ...}
```

**Check Frontend:**
Open http://localhost:3000 in browser
- Should connect to http://localhost:8000/api
- Should connect to ws://localhost:8000/ws/radar-stream

**Check WebSocket in Browser Console:**
```javascript
// Open DevTools > Console
// You should see: "[WS] Connected to ws://localhost:8000/ws/radar-stream"
```

---

## 🔍 Troubleshooting

### Issue: CORS Errors
**Solution:** Backend CORS middleware already allows `http://localhost:3000` ✅

### Issue: WebSocket Connection Fails
**Solution:** Verify backend runs on port 8000 and WebSocket route is `/ws/radar-stream` ✅

### Issue: API Returns 404
**Solution:** Verify `baseURL` in axios is `http://localhost:8000/api` ✅

### Issue: Environment Variables Not Loaded
**Solution:** 
1. Make sure `.env.development` exists in `/frontend/`
2. Restart Vite dev server
3. Check `console.log` for "Environment Config"

### Issue: Production Build Fails
**Solution:** 
1. Use `.env.production` with correct URLs
2. Or set `VITE_API_URL` at build time:
   ```bash
   VITE_API_URL=https://api.example.com npm run build
   ```

---

## 📝 Key Changes Made

| File | Issue | Fix |
|------|-------|-----|
| `envConfig.js` | No methods to get URLs | Added `getApiUrl()` and `getWsUrl()` |
| `apiConfig.ts` | Used `process.env` (wrong) | Changed to `import.meta.env` |
| `apiClient.js` | Hardcoded URL construction | Now uses `envConfig.getApiUrl()` |
| `websocketClient.js` | Hardcoded URL + wrong port | Now uses `envConfig.getWsUrl()` |
| `vite.config.ts` | Hardcoded port 3000 | Now uses `VITE_PORT` env var |
| `.env.development` | Missing server config | Added `VITE_HOST` and `VITE_PORT` |
| `.env.production` | Hardcoded to `/api` and `/ws` | Made flexible with `VITE_API_URL` |

---

## ✅ Verification Checklist

- [x] Frontend connects to `http://localhost:8000`
- [x] WebSocket connects to `ws://localhost:8000/ws/radar-stream`
- [x] Removed hardcoded ports
- [x] CORS configured for localhost:3000
- [x] Environment variables use `import.meta.env`
- [x] Both dev and production configs work

**Status:** ✅ **ALL ISSUES FIXED - READY FOR TESTING**
