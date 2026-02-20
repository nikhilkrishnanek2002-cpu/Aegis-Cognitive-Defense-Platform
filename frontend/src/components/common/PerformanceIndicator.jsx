import { useEffect, useState, useRef } from 'react'
import { Card } from './Card'
import './PerformanceIndicator.css'

/**
 * Real-time performance indicator showing FPS, latency, and system metrics
 */
export function PerformanceIndicator() {
  const [metrics, setMetrics] = useState({
    fps: 0,
    latency: 0,
    cpu: 0,
    memory: 0,
    connections: 0,
    messageRate: 0,
  })
  const [history, setHistory] = useState({
    latencies: [],
    fps: [],
  })
  const frameCountRef = useRef(0)
  const lastFrameTimeRef = useRef(Date.now())
  const messageCountRef = useRef(0)
  const lastMessageTimeRef = useRef(Date.now())

  // Measure FPS
  useEffect(() => {
    const measureFps = () => {
      frameCountRef.current++
      const now = Date.now()
      const delta = now - lastFrameTimeRef.current

      if (delta >= 1000) {
        const fps = Math.round((frameCountRef.current * 1000) / delta)
        setMetrics((prev) => ({
          ...prev,
          fps: Math.min(fps, 60), // Cap at 60 FPS
        }))
        setHistory((prev) => ({
          ...prev,
          fps: [...prev.fps, fps].slice(-60), // Keep last 60 samples
        }))
        frameCountRef.current = 0
        lastFrameTimeRef.current = now
      }

      requestAnimationFrame(measureFps)
    }

    const rafId = requestAnimationFrame(measureFps)
    return () => cancelAnimationFrame(rafId)
  }, [])

  // Fetch performance metrics from backend
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        // Fetch performance data
        const perfResponse = await fetch('/api/metrics/performance')
        if (perfResponse.ok) {
          const perfData = await perfResponse.json()
          const latencies = perfData.stages?.map((s) => s.latest || 0) || []
          const avgLatency = latencies.length > 0 
            ? latencies.reduce((a, b) => a + b, 0) / latencies.length 
            : 0

          setMetrics((prev) => ({
            ...prev,
            latency: Math.round(avgLatency),
            connections: perfData.websocket?.connections || 0,
            messageRate: perfData.websocket?.messages_sent || 0,
          }))

          setHistory((prev) => ({
            ...prev,
            latencies: [...prev.latencies, avgLatency].slice(-120), // 2 min history
          }))
        }

        // Fetch system metrics
        const sysResponse = await fetch('/api/health/cpu-memory')
        if (sysResponse.ok) {
          const sysData = await sysResponse.json()
          setMetrics((prev) => ({
            ...prev,
            cpu: Math.round(sysData.cpu_percent || 0),
            memory: Math.round(sysData.memory_percent || 0),
          }))
        }
      } catch (error) {
        console.error('Failed to fetch performance metrics:', error)
      }
    }

    const interval = setInterval(fetchMetrics, 2000) // Update every 2 seconds
    fetchMetrics() // Initial fetch

    return () => clearInterval(interval)
  }, [])

  // Determine color based on metric value
  const getMetricColor = (value, type) => {
    switch (type) {
      case 'fps':
        return value >= 55 ? '#22c55e' : value >= 30 ? '#eab308' : '#ef4444'
      case 'latency':
        return value <= 50 ? '#22c55e' : value <= 150 ? '#eab308' : '#ef4444'
      case 'cpu':
        return value <= 50 ? '#22c55e' : value <= 80 ? '#eab308' : '#ef4444'
      case 'memory':
        return value <= 60 ? '#22c55e' : value <= 85 ? '#eab308' : '#ef4444'
      default:
        return '#06b6d4'
    }
  }

  return (
    <Card title="Performance Monitor" subtitle="Real-time metrics">
      <div className="performance-grid">
        <div className="metric-box">
          <div className="metric-label">FPS</div>
          <div
            className="metric-value"
            style={{ color: getMetricColor(metrics.fps, 'fps') }}
          >
            {metrics.fps}
          </div>
          <div className="metric-unit">frames/sec</div>
          <div className="metric-subtext">Target: 60 FPS</div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Latency</div>
          <div
            className="metric-value"
            style={{ color: getMetricColor(metrics.latency, 'latency') }}
          >
            {metrics.latency}
          </div>
          <div className="metric-unit">ms</div>
          <div className="metric-subtext">Avg pipeline latency</div>
        </div>

        <div className="metric-box">
          <div className="metric-label">CPU</div>
          <div
            className="metric-value"
            style={{ color: getMetricColor(metrics.cpu, 'cpu') }}
          >
            {metrics.cpu}
          </div>
          <div className="metric-unit">%</div>
          <div className="metric-subtext">System usage</div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Memory</div>
          <div
            className="metric-value"
            style={{ color: getMetricColor(metrics.memory, 'memory') }}
          >
            {metrics.memory}
          </div>
          <div className="metric-unit">%</div>
          <div className="metric-subtext">System memory</div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Connections</div>
          <div className="metric-value" style={{ color: '#06b6d4' }}>
            {metrics.connections}
          </div>
          <div className="metric-unit">active</div>
          <div className="metric-subtext">WebSocket clients</div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Messages/sec</div>
          <div className="metric-value" style={{ color: '#06b6d4' }}>
            {metrics.messageRate}
          </div>
          <div className="metric-unit">msgs</div>
          <div className="metric-subtext">Broadcast rate</div>
        </div>
      </div>

      <div className="metric-chart">
        <div className="chart-title">Latency Trend (ms)</div>
        <div className="sparkline">
          {history.latencies.map((latency, idx) => (
            <div
              key={idx}
              className="sparkline-bar"
              style={{
                height: `${Math.min(latency / 2, 100)}%`,
                backgroundColor: getMetricColor(latency, 'latency'),
              }}
              title={`${Math.round(latency)}ms`}
            />
          ))}
        </div>
      </div>

      <div className="metric-status">
        <div className="status-indicator green" />
        <span>All systems healthy</span>
      </div>
    </Card>
  )
}

export default PerformanceIndicator
