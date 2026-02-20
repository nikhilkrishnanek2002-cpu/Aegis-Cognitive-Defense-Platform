import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

// Optimized selectors to prevent unnecessary re-renders
export const selectTargets = (state) => state.targets
export const selectFrame = (state) => state.frame
export const selectConnectionState = (state) => state.isConnected
export const selectIsScanning = (state) => state.isScanning
export const selectScanError = (state) => state.error

// Combined selector for canvas rendering (targets + frame)
export const selectRadarCanvasData = (state) => ({
  targets: state.targets,
  frame: state.frame,
})

// Memoized selector for threat level computation
const computeThreatLevel = (targets) => {
  if (targets.length === 0) return 'None'
  const levels = targets.map((t) => t.threat_level || 'Low')
  const priority = { Critical: 0, High: 1, Medium: 2, Low: 3 }
  return levels.sort((a, b) => priority[a] - priority[b])[0]
}

export const useRadarStore = create(
  devtools((set) => ({
    // State
    frame: null,
    targets: [],
    scanHistory: [],
    isConnected: false,
    isScanning: false,
    lastScanTime: null,
    error: null,

    // Setters (optimized to avoid unnecessary renders)
    setFrame: (frame) => set((state) => 
      state.frame === frame ? state : { frame }
    ),
    setTargets: (targets) => set((state) => {
      // Only update if targets array actually changed
      const targetsChanged = 
        state.targets.length !== targets.length ||
        state.targets.some((t, i) => t !== targets[i])
      return targetsChanged ? { targets } : state
    }),
    addScanHistoryEntry: (entry) =>
      set((state) => ({
        scanHistory: [entry, ...state.scanHistory].slice(0, 50),
      })),
    setScanState: (isScanning) => set((state) =>
      state.isScanning === isScanning ? state : { isScanning }
    ),
    setConnectionState: (isConnected) => set((state) =>
      state.isConnected === isConnected ? state : { isConnected }
    ),
    setLastScanTime: (lastScanTime) => set({ lastScanTime }),
    setError: (error) => set((state) =>
      state.error === error ? state : { error }
    ),
    clearError: () => set((state) =>
      state.error === null ? state : { error: null }
    ),

    // Computed
    getTargetCount: () => {
      const state = useRadarStore.getState()
      return state.targets.length
    },
    getHighestThreatLevel: () => {
      const state = useRadarStore.getState()
      return computeThreatLevel(state.targets)
    },
    reset: () =>
      set({
        frame: null,
        targets: [],
        scanHistory: [],
        isScanning: false,
        error: null,
      }),
  }))
)

// Export hooks for optimized subscriptions
export const useRadarSelector = (selector) => useRadarStore(selector)
export const useTargets = () => useRadarStore(selectTargets)
export const useRadarFrame = () => useRadarStore(selectFrame)
export const useRadarConnected = () => useRadarStore(selectConnectionState)
export const useRadarScanning = () => useRadarStore(selectIsScanning)
