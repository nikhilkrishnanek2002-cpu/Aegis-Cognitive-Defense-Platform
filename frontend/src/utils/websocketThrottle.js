/**
 * WebSocket Message Throttling Utility
 * Prevents excessive re-renders by limiting update frequency
 * Target: 10-20 FPS (50-100ms between updates)
 */

/**
 * Throttle function calls based on time interval
 * @param {Function} callback - Function to throttle
 * @param {number} delay - Minimum milliseconds between calls (default: 50ms for 20 FPS)
 * @returns {Function} Throttled callback
 */
export function throttle(callback, delay = 50) {
  let lastCall = 0
  let timeoutId = null

  return function throttled(...args) {
    const now = Date.now()
    const timeSinceLastCall = now - lastCall

    if (timeSinceLastCall >= delay) {
      lastCall = now
      callback(...args)
    } else {
      // Schedule the call for later if not already scheduled
      clearTimeout(timeoutId)
      timeoutId = setTimeout(() => {
        lastCall = Date.now()
        callback(...args)
      }, delay - timeSinceLastCall)
    }
  }
}

/**
 * Debounce function calls - waits for silence before executing
 * @param {Function} callback - Function to debounce
 * @param {number} delay - Milliseconds to wait before executing (default: 100ms)
 * @returns {Function} Debounced callback
 */
export function debounce(callback, delay = 100) {
  let timeoutId = null

  return function debounced(...args) {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => {
      callback(...args)
    }, delay)
  }
}

/**
 * Request animation frame throttle - aligns with browser refresh rate
 * @param {Function} callback - Function to throttle
 * @returns {Function} RAF-throttled callback
 */
export function rafThrottle(callback) {
  let scheduledId = null
  let lastArgs = null

  return function rafThrottled(...args) {
    lastArgs = args

    if (!scheduledId) {
      scheduledId = requestAnimationFrame(() => {
        callback(...lastArgs)
        scheduledId = null
      })
    }
  }
}

/**
 * Create a throttled batch processor for WebSocket messages
 * Accumulates messages and processes them in batches
 */
export class ThrottledBatchProcessor {
  constructor(processFn, delay = 100, maxBatchSize = 50) {
    this.processFn = processFn
    this.delay = delay
    this.maxBatchSize = maxBatchSize
    this.batch = []
    this.timeoutId = null
    this.lastProcessed = 0
  }

  add(item) {
    this.batch.push(item)

    // Process immediately if batch size limit reached
    if (this.batch.length >= this.maxBatchSize) {
      this.flush()
      return
    }

    // Schedule processing if not already scheduled
    if (!this.timeoutId) {
      this.timeoutId = setTimeout(() => {
        this.flush()
      }, this.delay)
    }
  }

  flush() {
    if (this.batch.length === 0) {
      clearTimeout(this.timeoutId)
      this.timeoutId = null
      return
    }

    const itemsToProcess = this.batch.splice(0, this.maxBatchSize)
    this.lastProcessed = Date.now()
    this.processFn(itemsToProcess)

    // Keep processing if more items remain
    if (this.batch.length > 0) {
      this.timeoutId = setTimeout(() => {
        this.flush()
      }, this.delay)
    } else {
      clearTimeout(this.timeoutId)
      this.timeoutId = null
    }
  }

  clear() {
    this.batch = []
    clearTimeout(this.timeoutId)
    this.timeoutId = null
  }

  getStats() {
    return {
      batchSize: this.batch.length,
      maxBatchSize: this.maxBatchSize,
      delay: this.delay,
      lastProcessed: this.lastProcessed,
    }
  }
}

/**
 * Frame rate limiter - limits updates to specific FPS
 */
export class FrameRateLimiter {
  constructor(fps = 20) {
    this.fps = fps
    this.interval = 1000 / fps
    this.lastFrameTime = 0
    this.frameCount = 0
  }

  isTimeForFrame() {
    const now = Date.now()
    if (now - this.lastFrameTime >= this.interval) {
      this.lastFrameTime = now
      this.frameCount++
      return true
    }
    return false
  }

  getStats() {
    return {
      fps: this.fps,
      interval: this.interval,
      framesProcessed: this.frameCount,
      lastFrameTime: this.lastFrameTime,
    }
  }

  reset() {
    this.lastFrameTime = 0
    this.frameCount = 0
  }
}

// Export pre-configured throttled update function
export const throttledWsUpdate = throttle((data) => {
  // This will be called at most every 50ms (20 FPS)
  // Replace with your actual update logic
}, 50)

export default {
  throttle,
  debounce,
  rafThrottle,
  ThrottledBatchProcessor,
  FrameRateLimiter,
  throttledWsUpdate,
}
