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
    this.maxReconnectAttempts = 999 // Infinite retry
    this.baseDelay = envConfig.get('websocketReconnectInterval', 2000)
    this.isIntentionallyClosed = false
    this.connectionTimestamp = null
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return Promise.resolve()
    }

    return new Promise((resolve, reject) => {
      try {
        this.isIntentionallyClosed = false
        this.ws = new WebSocket(this.url)
        this.connectionTimestamp = Date.now()

        this.ws.onopen = () => {
          console.log('✅ [WS] Connected to backend')
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
          console.warn('[WS] Connection error:', error)
          this.emit('error', error)
          reject(error)
        }

        this.ws.onclose = () => {
          console.log('[WS] Disconnected from backend')
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
    // Always retry with exponential backoff (no max attempt limit)
    const delay = Math.min(this.baseDelay * Math.pow(2, this.reconnectAttempts), 30000) // Cap at 30 seconds
    this.reconnectAttempts++

    console.warn(`[WS] Retrying in ${delay}ms (attempt ${this.reconnectAttempts})`)

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
