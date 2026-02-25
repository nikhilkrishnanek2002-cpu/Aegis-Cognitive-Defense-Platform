/**
 * Frontend Radar Simulation - Fallback when backend is unavailable
 * Generates realistic moving targets to ensure UI never goes blank
 */

export interface SimulatedTarget {
  id: string
  name: string
  type: 'DRONE' | 'AIRCRAFT' | 'HELICOPTER' | 'BIRD' | 'MISSILE' | 'UNKNOWN'
  x: number
  y: number
  velocity: number
  heading: number
  strength: number
  threat_level: 'LOW' | 'MEDIUM' | 'HIGH'
  timestamp: string
}

export interface SimulatedMetrics {
  cycle: number
  timestamp: string
  targets_detected: number
  threats_high: number
  threats_medium: number
  threats_low: number
  cpu_usage: number
  memory_usage: number
  cycle_time_ms: number
}

class SimulatedTarget_ implements SimulatedTarget {
  id: string
  name: string
  type: 'DRONE' | 'AIRCRAFT' | 'HELICOPTER' | 'BIRD' | 'MISSILE' | 'UNKNOWN'
  x: number
  y: number
  vx: number
  vy: number
  velocity: number
  heading: number
  strength: number
  threat_level: 'LOW' | 'MEDIUM' | 'HIGH' = 'LOW'
  created_at: number
  timestamp: string

  private static targetCounter = 0
  private static typeNames = ['DRONE', 'AIRCRAFT', 'HELICOPTER', 'BIRD', 'MISSILE'] as const

  constructor() {
    const types = SimulatedTarget_.typeNames
    this.type = types[Math.floor(Math.random() * types.length)]
    SimulatedTarget_.targetCounter++
    this.id = `sim_${SimulatedTarget_.targetCounter}`
    this.name = `${this.type}-${SimulatedTarget_.targetCounter}`
    this.x = Math.random() * 1000 - 500
    this.y = Math.random() * 1000 - 500
    this.vx = Math.random() * 60 - 30
    this.vy = Math.random() * 60 - 30
    this.velocity = Math.sqrt(this.vx ** 2 + this.vy ** 2)
    this.heading = (Math.atan2(this.y, this.x) * 180) / Math.PI
    this.strength = Math.random() * 0.5 + 0.5
    this.created_at = Date.now()
    this.timestamp = new Date().toISOString()
  }

  update(dt: number = 0.3) {
    // Add slight acceleration
    this.vx += (Math.random() - 0.5) * 10
    this.vy += (Math.random() - 0.5) * 10

    // Clamp velocity
    const speed = Math.sqrt(this.vx ** 2 + this.vy ** 2)
    if (speed > 40) {
      this.vx = (this.vx / speed) * 40
      this.vy = (this.vy / speed) * 40
    }

    // Update position
    this.x += this.vx * dt
    this.y += this.vy * dt

    // Boundary wrap
    if (this.x > 500 || this.x < -500) this.vx *= -1
    if (this.y > 500 || this.y < -500) this.vy *= -1

    this.x = Math.max(-500, Math.min(500, this.x))
    this.y = Math.max(-500, Math.min(500, this.y))

    // Update derived values
    this.velocity = Math.sqrt(this.vx ** 2 + this.vy ** 2)
    this.heading = (Math.atan2(this.y, this.x) * 180) / Math.PI
    this.strength = Math.max(0.3, Math.min(1.0, this.strength + (Math.random() - 0.5) * 0.1))

    // Randomly set threat level
    if (Math.random() < 0.1) {
      const r = Math.random()
      if (r < 0.7) this.threat_level = 'LOW'
      else if (r < 0.9) this.threat_level = 'MEDIUM'
      else this.threat_level = 'HIGH'
    }

    this.timestamp = new Date().toISOString()
  }

  toJSON(): SimulatedTarget {
    return {
      id: this.id,
      name: this.name,
      type: this.type,
      x: this.x,
      y: this.y,
      velocity: this.velocity,
      heading: this.heading,
      strength: this.strength,
      threat_level: this.threat_level,
      timestamp: this.timestamp
    }
  }
}

export class RadarSimulator {
  private targets: SimulatedTarget_[] = []
  private cycleCount = 0
  private metricsHistory: SimulatedMetrics[] = []
  private lifespan = 50

  executeSimulationCycle(): {
    targets: SimulatedTarget[]
    metrics: SimulatedMetrics
  } {
    this.cycleCount++

    // Remove old targets
    this.targets = this.targets.filter(
      (t) => (Date.now() - t.created_at) < this.lifespan * 300
    )

    // Update existing targets
    this.targets.forEach((t) => t.update())

    // Spawn 2-6 new targets
    const numNew = Math.floor(Math.random() * 5) + 2
    for (let i = 0; i < numNew && this.targets.length < 50; i++) {
      this.targets.push(new SimulatedTarget_())
    }

    // Generate metrics
    const metrics: SimulatedMetrics = {
      cycle: this.cycleCount,
      timestamp: new Date().toISOString(),
      targets_detected: this.targets.length,
      threats_high: this.targets.filter((t) => t.threat_level === 'HIGH').length,
      threats_medium: this.targets.filter((t) => t.threat_level === 'MEDIUM').length,
      threats_low: this.targets.filter((t) => t.threat_level === 'LOW').length,
      cpu_usage: 15 + (Math.random() - 0.5) * 10,
      memory_usage: 250 + (Math.random() - 0.5) * 40,
      cycle_time_ms: 2.5 + (Math.random() - 0.5) * 2
    }

    this.metricsHistory.push(metrics)
    if (this.metricsHistory.length > 100) {
      this.metricsHistory = this.metricsHistory.slice(-100)
    }

    return {
      targets: this.targets.map((t) => t.toJSON()),
      metrics
    }
  }

  getMetricsHistory(): SimulatedMetrics[] {
    return this.metricsHistory.slice(-50)
  }
}
