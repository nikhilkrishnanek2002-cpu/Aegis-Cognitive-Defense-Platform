import { useEffect, useRef, memo } from 'react'
import { selectRadarCanvasData } from '../../store/radarStore'
import { useRadarStore } from '../../store/radarStore'

function RadarCanvasComponent() {
  const canvasRef = useRef(null)
  // Use optimized selector to prevent re-renders when other state changes
  const { targets, frame } = useRadarStore(selectRadarCanvasData)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    const width = canvas.width
    const height = canvas.height
    const centerX = width / 2
    const centerY = height / 2
    const radius = Math.min(width, height) / 2 - 10

    // Clear canvas
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = '#1e293b'
    ctx.lineWidth = 1

    // Concentric circles
    for (let i = 1; i <= 4; i++) {
      ctx.beginPath()
      ctx.arc(centerX, centerY, (radius / 4) * i, 0, Math.PI * 2)
      ctx.stroke()
    }

    // Radial lines
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 8) {
      ctx.beginPath()
      ctx.moveTo(centerX, centerY)
      ctx.lineTo(
        centerX + Math.cos(angle) * radius,
        centerY + Math.sin(angle) * radius
      )
      ctx.stroke()
    }

    // Draw compass labels
    ctx.fillStyle = '#64748b'
    ctx.font = '12px monospace'
    ctx.textAlign = 'center'
    ctx.fillText('N', centerX, 15)
    ctx.fillText('S', centerX, height - 5)
    ctx.textAlign = 'right'
    ctx.fillText('W', 10, centerY + 4)
    ctx.textAlign = 'left'
    ctx.fillText('E', width - 10, centerY + 4)

    // Draw targets
    if (targets && targets.length > 0) {
      targets.forEach((target, idx) => {
        const distance = target.distance || 50
        const bearing = ((target.bearing || 0) * Math.PI) / 180

        const x = centerX + Math.sin(bearing) * (distance / 100) * radius
        const y = centerY - Math.cos(bearing) * (distance / 100) * radius

        // Target dot
        const colorMap = {
          Critical: '#ef4444',
          High: '#f97316',
          Medium: '#eab308',
          Low: '#22c55e',
        }
        ctx.fillStyle = colorMap[target.level] || '#06b6d4'
        ctx.beginPath()
        ctx.arc(x, y, 6, 0, Math.PI * 2)
        ctx.fill()

        // Target label
        ctx.fillStyle = '#e2e8f0'
        ctx.font = 'bold 10px monospace'
        ctx.textAlign = 'center'
        ctx.fillText(idx + 1, x, y + 15)
      })
    }

    // Draw heading indicator
    ctx.strokeStyle = '#06b6d4'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(centerX, centerY - radius)
    ctx.lineTo(centerX - 5, centerY - radius + 10)
    ctx.lineTo(centerX + 5, centerY - radius + 10)
    ctx.closePath()
    ctx.stroke()
  }, [targets, frame])

  return (
    <canvas
      ref={canvasRef}
      width={400}
      height={400}
      className="w-full border border-slate-700 rounded-lg bg-slate-900"
    />
  )
}

// Custom comparison: only re-render if targets array length or frame ID changes
function arePropsEqual(prev, next) {
  // For function components with hooks, this comparison doesn't apply directly
  // Instead, we rely on Zustand's internal selector optimization
  return true
}

export const RadarCanvas = memo(RadarCanvasComponent, arePropsEqual)

export default RadarCanvas
