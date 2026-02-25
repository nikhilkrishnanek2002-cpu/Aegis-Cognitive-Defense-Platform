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
