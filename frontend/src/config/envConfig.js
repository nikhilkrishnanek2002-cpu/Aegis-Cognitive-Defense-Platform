/**
 * Environment Configuration Utility
 * Handles dev vs production settings for performance optimization
 */

class EnvironmentConfig {
  constructor() {
    this.isDev = import.meta.env.DEV
    this.isProd = import.meta.env.PROD
    this.config = {
      apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
      wsUrl: import.meta.env.VITE_WS_URL || 'ws://localhost:8000',
      debug: import.meta.env.VITE_DEBUG === 'true' || false,
      performanceLogging: import.meta.env.VITE_PERFORMANCE_LOGGING === 'true' || false,
      cacheEnabled: import.meta.env.VITE_CACHE_ENABLED !== 'false',
      cacheTtl: parseInt(import.meta.env.VITE_CACHE_TTL || '1000'),
      websocketThrottleFps: parseInt(import.meta.env.VITE_WEBSOCKET_THROTTLE_FPS || '20'),
      websocketReconnectInterval: parseInt(import.meta.env.VITE_WEBSOCKET_RECONNECT_INTERVAL || '3000'),
      websocketReconnectMaxAttempts: parseInt(import.meta.env.VITE_WEBSOCKET_RECONNECT_MAX_ATTEMPTS || '10'),
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
