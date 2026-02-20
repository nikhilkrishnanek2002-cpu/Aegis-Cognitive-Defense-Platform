import { EventEmitter } from 'eventemitter3'

class WebSocketClient extends EventEmitter {
  constructor(url) {
    super()
    this.url = url || `${import.meta.env.VITE_WS_URL || 'ws://localhost:8000'}/ws/radar-stream`
    this.ws = null
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = 10
    this.baseDelay = 1000
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
          console.log('[WS] Connected')
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
          console.log('[WS] Disconnected')
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
