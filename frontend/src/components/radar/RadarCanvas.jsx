import { useEffect, useRef, memo } from 'react'
import { selectRadarCanvasData, useRadarStore } from '../../store/radarStore'

function RadarCanvasComponent() {
  const canvasRef = useRef(null)
  const { targets, frame } = useRadarStore(selectRadarCanvasData)

  // Use a ref to keep track of rotation angle for the sweep
  const sweepAngleRef = useRef(0)
  const animationRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    const render = () => {
      const width = canvas.width
      const height = canvas.height
      const centerX = width / 2
      const centerY = height / 2
      const radius = Math.min(width, height) / 2 - 40 // Leave room for outer labels

      // 1. Clear background (Deep dark blue/gray space)
      ctx.fillStyle = '#111724'
      ctx.fillRect(0, 0, width, height)

      // 2. Base Grid & Concentric Circles
      ctx.lineWidth = 1
      const numCircles = 6
      for (let i = 1; i <= numCircles; i++) {
        ctx.beginPath()
        ctx.arc(centerX, centerY, (radius / numCircles) * i, 0, Math.PI * 2)
        ctx.strokeStyle = i === numCircles ? '#38bdf8' : 'rgba(56, 189, 248, 0.2)'
        ctx.stroke()
      }

      // 3. Radial Lines & Degree Labels (Polar Coordinates)
      ctx.font = '10px "Orbitron", "Courier New", monospace'
      ctx.fillStyle = '#64748b'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'

      for (let angle = 0; angle < 360; angle += 30) {
        const rad = (angle * Math.PI) / 180 - Math.PI / 2 // -90 deg so 0 is Top (North)

        // Draw Radial Lines
        ctx.beginPath()
        ctx.moveTo(centerX, centerY)
        ctx.lineTo(
          centerX + Math.cos(rad) * radius,
          centerY + Math.sin(rad) * radius
        )
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.15)'
        ctx.stroke()

        // Draw Outer Tick Marks & Text
        const textRad = rad
        const textDist = radius + 20
        const isMajor = angle % 30 === 0

        if (isMajor) {
          ctx.fillText(`${angle}`, centerX + Math.cos(textRad) * textDist, centerY + Math.sin(textRad) * textDist)
          ctx.beginPath()
          ctx.moveTo(centerX + Math.cos(rad) * radius, centerY + Math.sin(rad) * radius)
          ctx.lineTo(centerX + Math.cos(rad) * (radius + 8), centerY + Math.sin(rad) * (radius + 8))
          ctx.strokeStyle = '#38bdf8'
          ctx.stroke()
        }
      }

      // Add smaller ticks for every 10 degrees at the edge
      for (let angle = 0; angle < 360; angle += 10) {
        if (angle % 30 !== 0) {
          const rad = (angle * Math.PI) / 180 - Math.PI / 2
          ctx.beginPath()
          ctx.moveTo(centerX + Math.cos(rad) * radius, centerY + Math.sin(rad) * radius)
          ctx.lineTo(centerX + Math.cos(rad) * (radius + 4), centerY + Math.sin(rad) * (radius + 4))
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)'
          ctx.stroke()
        }
      }

      // 4. Center Scanner Box
      ctx.fillStyle = '#2dd4bf'
      ctx.shadowColor = '#2dd4bf'
      ctx.shadowBlur = 15
      ctx.fillRect(centerX - 8, centerY - 8, 16, 16)
      ctx.shadowBlur = 0 // reset

      // 5. Kalman Prediction Cone
      const predictionStart = sweepAngleRef.current - 0.4
      const predictionEnd = sweepAngleRef.current + 0.1

      ctx.beginPath()
      ctx.moveTo(centerX, centerY)
      ctx.arc(centerX, centerY, radius, predictionStart, predictionEnd)
      ctx.closePath()

      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius)
      gradient.addColorStop(0, 'rgba(45, 212, 191, 0.4)')
      gradient.addColorStop(1, 'rgba(45, 212, 191, 0.0)')
      ctx.fillStyle = gradient
      ctx.fill()

      // Arc outline for prediction
      ctx.beginPath()
      ctx.arc(centerX, centerY, radius, predictionStart, predictionEnd)
      ctx.strokeStyle = '#2dd4bf'
      ctx.setLineDash([5, 5])
      ctx.stroke()
      ctx.setLineDash([])

      // 6. Draw Simulated Kalman Nodes (Cyan Dots forming an arc)
      const kalmanRadius = radius * 0.7
      for (let i = 0; i < 8; i++) {
        const nodeAngle = predictionStart + (Math.abs(predictionEnd - predictionStart) / 8) * i
        const nx = centerX + Math.cos(nodeAngle) * kalmanRadius
        const ny = centerY + Math.sin(nodeAngle) * kalmanRadius
        ctx.beginPath()
        ctx.arc(nx, ny, 3, 0, Math.PI * 2)
        ctx.fillStyle = '#2dd4bf'
        ctx.fill()
      }

      // Texts
      ctx.fillStyle = '#2dd4bf'
      ctx.font = 'bold 13px "Orbitron", monospace'
      ctx.textAlign = 'right'
      ctx.fillText('KALMAN PREDICTION', centerX + radius - 10, centerY + 30)

      ctx.textAlign = 'left'
      ctx.fillText('CONFIRMED TRACK', centerX - radius + 20, centerY + 30)

      // 7. Draw Active Targets
      // We will merge explicit `targets` array with `frame.active_tracks` if they exist
      const combinedTargets = []
      
      // Threat level to color mapping
      const threatColorMap = {
        'Critical': '#ef4444',
        'High': '#f97316',
        'Medium': '#eab308',
        'Low': '#22c55e',
        'Unknown': '#2dd4bf'
      }

      if (targets && targets.length > 0) {
        targets.forEach(t => combinedTargets.push({
          distance: t.distance || 0,
          bearing: t.bearing || 0,
          confidence: 1.0,
          threat_level: t.threat_level || 'Low'
        }))
      }

      if (frame && frame.active_tracks) {
        Object.entries(frame.active_tracks).forEach(([trackId, track], idx) => {
          // Assign threat levels in a round-robin fashion for demo purposes
          const threatLevels = ['Critical', 'High', 'Medium', 'Low']
          const assignedThreatLevel = threatLevels[idx % threatLevels.length]
          
          combinedTargets.push({
            distance: track.position ? track.position[0] * 5 : 50,
            bearing: track.position ? track.position[1] * 20 : Math.random() * 360,
            confidence: track.confidence || 1.0,
            threat_level: track.threat_level || assignedThreatLevel
          })
        })
      }

      // If no real targets, draw some fake ones to show off the UI look exactly like the image
      const displayTargets = combinedTargets.length > 0 ? combinedTargets : [
        { distance: 60, bearing: 210, confidence: 0.9, threat_level: 'Critical' },
        { distance: 80, bearing: 140, confidence: 0.8, threat_level: 'High' },
        { distance: 40, bearing: 160, confidence: 0.95, threat_level: 'Medium' },
        { distance: 90, bearing: 45, confidence: 0.6, threat_level: 'Low' },
      ]

      displayTargets.forEach(t => {
        // Normalize inputs
        const dist = (t.distance / 100) * radius // Map distance 0-100 to canvas radius
        const clampedDist = Math.max(10, Math.min(dist, radius - 5))
        const angleRad = (t.bearing * Math.PI) / 180 - Math.PI / 2

        const tx = centerX + Math.cos(angleRad) * clampedDist
        const ty = centerY + Math.sin(angleRad) * clampedDist
        
        // Get threat level color
        const threatColor = threatColorMap[t.threat_level] || threatColorMap['Unknown']

        // Draw fading trail extending backward slightly
        ctx.beginPath()
        ctx.moveTo(tx, ty)
        const trailAngle = angleRad - 0.2 // Fake trail orientation
        const tr_x = centerX + Math.cos(trailAngle) * (clampedDist + 20)
        const tr_y = centerY + Math.sin(trailAngle) * (clampedDist + 20)
        ctx.lineTo(tr_x, tr_y)
        const lineGrad = ctx.createLinearGradient(tx, ty, tr_x, tr_y)
        lineGrad.addColorStop(0, threatColor)
        lineGrad.addColorStop(1, threatColor + '00') // fade to transparent
        ctx.strokeStyle = lineGrad
        ctx.lineWidth = 2
        ctx.stroke()

        // Draw glowing dot with threat color
        ctx.beginPath()
        ctx.arc(tx, ty, 4, 0, Math.PI * 2)
        ctx.fillStyle = threatColor
        ctx.shadowColor = threatColor
        ctx.shadowBlur = 15
        ctx.fill()

        // Outer ring with threat color
        ctx.beginPath()
        ctx.arc(tx, ty, 8, 0, Math.PI * 2)
        ctx.strokeStyle = threatColor
        ctx.lineWidth = 1.5
        ctx.stroke()
        ctx.shadowBlur = 0 // reset
      })

      // Increment sweep angle
      sweepAngleRef.current += 0.015
      if (sweepAngleRef.current > Math.PI * 2) {
        sweepAngleRef.current = 0
      }

      animationRef.current = requestAnimationFrame(render)
    }

    render()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [targets, frame])

  return (
    <div style={{ position: 'relative', width: '100%', display: 'flex', justifyContent: 'center' }}>
      <canvas
        ref={canvasRef}
        width={500}
        height={500}
        style={{
          background: '#111724',
          border: '2px solid #38bdf8',
          borderRadius: '8px',
          boxShadow: '0 0 15px rgba(56, 189, 248, 0.2), inset 0 0 20px rgba(56, 189, 248, 0.1)',
          maxWidth: '100%',
          maxHeight: '100%',
          objectFit: 'contain'
        }}
      />
      {/* Decorative Sci-Fi Corners */}
      <div style={{ position: 'absolute', top: 0, left: '50%', transform: 'translateX(-250px)', width: '20px', height: '20px', borderTop: '2px solid #e0f2fe', borderLeft: '2px solid #e0f2fe', opacity: 0.8 }} />
    </div>
  )
}

function arePropsEqual(prev, next) {
  return true
}

export const RadarCanvas = memo(RadarCanvasComponent, arePropsEqual)

export default RadarCanvas
