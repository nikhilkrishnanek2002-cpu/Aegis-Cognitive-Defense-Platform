import { useState, useRef, useCallback, useEffect } from 'react'

/**
 * useMissileControl
 * Manages missile projectile and explosion animations for the radar canvas.
 *
 * missiles:   [ { id, fromX, fromY, toX, toY, progress (0-1) } ]
 * explosions: [ { id, x, y, radius, maxRadius, opacity } ]
 */
export function useMissileControl() {
  const [missiles, setMissiles] = useState([])
  const [explosions, setExplosions] = useState([])
  const rafRef = useRef(null)
  const missilesRef = useRef([])
  const explosionsRef = useRef([])

  // Keep refs in sync with state for use inside rAF loop
  useEffect(() => { missilesRef.current = missiles }, [missiles])
  useEffect(() => { explosionsRef.current = explosions }, [explosions])

  // Animation loop
  const animate = useCallback(() => {
    let hasWork = false

    const nextMissiles = []
    const newExplosions = []

    for (const m of missilesRef.current) {
      const next = { ...m, progress: m.progress + 0.025 } // speed: ~40 frames to impact
      if (next.progress >= 1) {
        // Missile reached target — spawn explosion
        newExplosions.push({
          id: `exp-${Date.now()}-${Math.random()}`,
          x: m.toX,
          y: m.toY,
          radius: 0,
          maxRadius: 40,
          opacity: 1,
        })
      } else {
        nextMissiles.push(next)
        hasWork = true
      }
    }

    const nextExplosions = []
    for (const e of [...explosionsRef.current, ...newExplosions]) {
      const ne = { ...e, radius: e.radius + 2.5, opacity: e.opacity - 0.035 }
      if (ne.opacity > 0) {
        nextExplosions.push(ne)
        hasWork = true
      }
    }

    if (nextMissiles.length !== missilesRef.current.length || nextExplosions.length !== explosionsRef.current.length) {
      setMissiles(nextMissiles)
      setExplosions(nextExplosions)
    } else if (newExplosions.length > 0) {
      setExplosions(nextExplosions)
    } else {
      // Update progress
      setMissiles(nextMissiles)
      setExplosions(nextExplosions)
    }

    if (hasWork || newExplosions.length > 0) {
      rafRef.current = requestAnimationFrame(animate)
    } else {
      rafRef.current = null
    }
  }, [])

  const launchMissile = useCallback((target, canvasWidth = 500, canvasHeight = 500) => {
    const centerX = canvasWidth / 2
    const centerY = canvasHeight / 2
    const radius = Math.min(canvasWidth, canvasHeight) / 2 - 40

    // Convert bearing + distance to canvas coords (mirrors RadarCanvas logic)
    const dist = (target.distance / 100) * radius
    const clampedDist = Math.max(10, Math.min(dist, radius - 5))
    const angleRad = ((target.bearing || 0) * Math.PI) / 180 - Math.PI / 2

    const toX = centerX + Math.cos(angleRad) * clampedDist
    const toY = centerY + Math.sin(angleRad) * clampedDist

    const missile = {
      id: `missile-${Date.now()}-${Math.random()}`,
      fromX: centerX,
      fromY: centerY,
      toX,
      toY,
      progress: 0,
    }

    setMissiles((prev) => [...prev, missile])

    // Start animation loop if not already running
    if (!rafRef.current) {
      rafRef.current = requestAnimationFrame(animate)
    }
  }, [animate])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  return { missiles, explosions, launchMissile }
}

export default useMissileControl
