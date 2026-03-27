import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

// Optimized selectors
export const selectActiveThreats = (state) => state.activeThreats
export const selectThreats = (state) => state.threats
export const selectEWThreats = (state) => state.ewThreats
export const selectSelectedThreat = (state) => state.selectedThreat
export const selectNeutralizedThreats = (state) => state.neutralizedThreats
export const selectEngagementLog = (state) => state.engagementLog

// Computed selectors
export const selectThreatCount = (state) => state.activeThreats.length
export const selectCriticalCount = (state) =>
  state.activeThreats.filter((t) => t.level === 'Critical').length

export const useThreatStore = create(
  devtools((set) => ({
    // State
    threats: [],
    activeThreats: [],
    threatHistory: [],
    ewThreats: [],
    selectedThreat: null,
    neutralizedThreats: [],
    engagementLog: [],

    // Setters (optimized)
    setThreats: (threats) =>
      set((state) => {
        const activeThreats = threats.filter((t) => t.status === 'Active')
        // Only update if actual changes
        const threatsChanged = 
          state.threats.length !== threats.length ||
          state.threats.some((t, i) => t !== threats[i])
        return threatsChanged ? { threats, activeThreats } : state
      }),
    addThreat: (threat) =>
      set((state) => ({
        threats: [threat, ...state.threats],
        activeThreats: threat.status === 'Active' ? [threat, ...state.activeThreats] : state.activeThreats,
        threatHistory: [threat, ...state.threatHistory].slice(0, 100),
      })),
    updateThreat: (threatId, updates) =>
      set((state) => ({
        threats: state.threats.map((t) => (t.id === threatId ? { ...t, ...updates } : t)),
        activeThreats: state.activeThreats.map((t) => (t.id === threatId ? { ...t, ...updates } : t)),
      })),
    removeThreat: (threatId) =>
      set((state) => ({
        threats: state.threats.filter((t) => t.id !== threatId),
        activeThreats: state.activeThreats.filter((t) => t.id !== threatId),
      })),
    setEWThreats: (ewThreats) => set((state) =>
      state.ewThreats === ewThreats ? state : { ewThreats }
    ),
    setSelectedThreat: (threat) => set((state) =>
      state.selectedThreat === threat ? state : { selectedThreat: threat }
    ),

    // Intercept / Neutralize a threat
    launchInterceptor: (threatId) =>
      set((state) => {
        const threat = state.activeThreats.find((t) => t.id === threatId)
        if (!threat) return state
        const neutralized = { ...threat, status: 'Neutralized', neutralizedAt: new Date() }
        const logEntry = {
          id: `eng-${Date.now()}`,
          threatId,
          type: threat.type || 'Unknown',
          level: threat.level || 'Unknown',
          timestamp: new Date(),
          bearing: threat.bearing,
          distance: threat.distance,
        }
        return {
          activeThreats: state.activeThreats.map((t) =>
            t.id === threatId ? { ...t, status: 'Neutralized' } : t
          ),
          threats: state.threats.map((t) =>
            t.id === threatId ? { ...t, status: 'Neutralized' } : t
          ),
          neutralizedThreats: [neutralized, ...state.neutralizedThreats],
          engagementLog: [logEntry, ...state.engagementLog].slice(0, 50),
        }
      }),

    // Computed
    getThreatCount: () => useThreatStore.getState().activeThreats.length,
    getCriticalCount: () =>
      useThreatStore.getState().activeThreats.filter((t) => t.level === 'Critical').length,
    reset: () =>
      set({
        threats: [],
        activeThreats: [],
        threatHistory: [],
        ewThreats: [],
        selectedThreat: null,
        neutralizedThreats: [],
        engagementLog: [],
      }),
  }))
)
