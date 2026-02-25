import { useEffect, useRef, useCallback, useState } from 'react'
import { RadarSimulator } from '../utils/radarSimulator'

type FrameHandler = (data: Record<string, unknown>) => void

export function useRadarWebSocket(onFrame: FrameHandler) {
    const wsRef = useRef<WebSocket | null>(null)
    const handlerRef = useRef(onFrame)
    const simulatorRef = useRef<RadarSimulator | null>(null)
    const simulationIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
    const connectionTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
    const [isConnected, setIsConnected] = useState(false)
    const [isSimulating, setIsSimulating] = useState(false)

    handlerRef.current = onFrame

    const startSimulation = useCallback(() => {
        console.log('🎮 Starting frontend radar simulation (backend unavailable)')
        setIsSimulating(true)

        if (!simulatorRef.current) {
            simulatorRef.current = new RadarSimulator()
        }

        // Generate and emit simulated data every 300ms
        simulationIntervalRef.current = setInterval(() => {
            try {
                const { targets, metrics } = simulatorRef.current!.executeSimulationCycle()

                const simulatedFrame = {
                    detected: targets.length > 0 ? targets[0].id : 'SIM-0',
                    confidence: 0.85,
                    priority: 'MEDIUM',
                    is_alert: targets.some(t => t.threat_level === 'HIGH'),
                    threshold: 0.5,
                    num_detections: targets.length,
                    active_tracks: targets.reduce((acc, t, i) => {
                        acc[t.id] = {
                            position: [t.x, t.y],
                            velocity: [Math.cos(t.heading) * t.velocity, Math.sin(t.heading) * t.velocity],
                            state: 'confirmed',
                            confidence: t.strength
                        }
                        return acc
                    }, {} as Record<string, any>),
                    ew: { active: false, threat_level: 'clear', num_threats: 0 },
                    cognitive: { is_adaptive: false, suggested_gain_db: 0 },
                    photonic: {
                        bandwidth_mhz: 100,
                        noise_power: 0.1,
                        clutter_power: 0.05,
                        pulse_width_us: 1.0,
                        chirp_slope_thz: 0.5,
                        ttd_vector: []
                    },
                    rd_map: [],
                    spec: [],
                    meta: [0, 0, 0],
                    timestamp: Date.now() / 1000,
                    targets: targets.map(t => ({
                        id: t.id,
                        x: t.x,
                        y: t.y,
                        velocity: t.velocity,
                        threat_level: t.threat_level
                    }))
                }
                handlerRef.current(simulatedFrame)
            } catch (e) {
                console.error('Simulation error:', e)
            }
        }, 300)
    }, [])

    const stopSimulation = useCallback(() => {
        if (simulationIntervalRef.current) {
            clearInterval(simulationIntervalRef.current)
            simulationIntervalRef.current = null
        }
        setIsSimulating(false)
    }, [])

    const connect = useCallback(() => {
        // Clear existing connection timeout
        if (connectionTimeoutRef.current) {
            clearTimeout(connectionTimeoutRef.current)
        }

        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
        const wsUrl = `${protocol}://localhost:8000/ws/stream`

        console.log('🔌 Connecting to WebSocket:', wsUrl)

        try {
            const ws = new WebSocket(wsUrl)
            wsRef.current = ws

            ws.onopen = () => {
                console.log('✅ WebSocket connected')
                setIsConnected(true)
                stopSimulation()

                // Clear fallback timeout
                if (connectionTimeoutRef.current) {
                    clearTimeout(connectionTimeoutRef.current)
                }
            }

            ws.onmessage = (ev) => {
                try {
                    const data = JSON.parse(ev.data)
                    if (data.type !== 'ping') handlerRef.current(data)
                } catch (_) { }
            }

            ws.onclose = () => {
                console.log('❌ WebSocket disconnected')
                setIsConnected(false)

                // Start simulation if not already running
                if (!isSimulating) {
                    startSimulation()
                }

                // Retry in 2 seconds
                setTimeout(connect, 2000)
            }

            ws.onerror = (err) => {
                console.warn('WebSocket error:', err)
                ws.close()
            }
        } catch (e) {
            console.warn('Failed to create WebSocket:', e)
            // Start simulation as fallback
            if (!isSimulating) {
                startSimulation()
            }
        }

        // If not connected after 3 seconds, start simulation
        connectionTimeoutRef.current = setTimeout(() => {
            if (wsRef.current?.readyState !== WebSocket.OPEN && !isSimulating) {
                console.warn('⏱️ WebSocket connection timeout after 3s, starting simulation')
                startSimulation()
            }
        }, 3000)
    }, [isSimulating, startSimulation, stopSimulation])

    useEffect(() => {
        connect()
        return () => {
            if (connectionTimeoutRef.current) {
                clearTimeout(connectionTimeoutRef.current)
            }
            if (simulationIntervalRef.current) {
                clearInterval(simulationIntervalRef.current)
            }
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.close()
            }
            stopSimulation()
        }
    }, [])
}

