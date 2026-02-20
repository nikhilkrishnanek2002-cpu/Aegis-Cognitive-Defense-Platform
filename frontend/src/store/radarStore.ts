import { create } from 'zustand'

export interface Track {
    position: [number, number]
    velocity: [number, number]
    state: string
    confidence: number
    measurement_count?: number
}

export interface RadarFrame {
    detected: string
    confidence: number
    priority: string
    is_alert: boolean
    threshold: number
    num_detections: number
    active_tracks: Record<string, Track>
    ew: { active: boolean; threat_level: string; num_threats: number }
    cognitive: { is_adaptive: boolean; suggested_gain_db: number }
    photonic: {
        bandwidth_mhz: number
        noise_power: number
        clutter_power: number
        pulse_width_us: number
        chirp_slope_thz: number
        ttd_vector: number[]
    }
    rd_map: number[][]
    spec: number[][]
    meta: number[]
    timestamp: number
}

interface RadarState {
    frame: RadarFrame | null
    trackHistory: Array<{ time: number; tracks: Record<string, Track> }>
    setFrame: (frame: RadarFrame) => void
}

export const useRadarStore = create<RadarState>((set) => ({
    frame: null,
    trackHistory: [],
    setFrame: (frame) =>
        set((state) => {
            const newHistory = [
                ...state.trackHistory,
                { time: frame.timestamp, tracks: frame.active_tracks },
            ].slice(-100) // Keep last 100 frames
            return { frame, trackHistory: newHistory }
        }),
}))
