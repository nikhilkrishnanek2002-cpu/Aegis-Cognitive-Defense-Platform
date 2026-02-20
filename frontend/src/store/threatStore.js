import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

// Optimized selectors
export const selectActiveThreats = (state) => state.activeThreats
export const selectThreats = (state) => state.threats
export const selectEWThreats = (state) => state.ewThreats
export const selectSelectedThreat = (state) => state.selectedThreat

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
      }),
  }))
)
