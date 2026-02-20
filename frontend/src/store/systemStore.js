import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

export const useSystemStore = create(
  devtools((set) => ({
    // State
    health: null,
    metrics: {},
    status: 'operational',
    events: [],
    alerts: [],
    uptime: 0,
    lastUpdate: null,

    // Setters
    setHealth: (health) => set({ health, lastUpdate: new Date().toISOString() }),
    setMetrics: (metrics) => set({ metrics }),
    addEvent: (event) =>
      set((state) => ({
        events: [{ ...event, id: Date.now(), timestamp: new Date().toISOString() }, ...state.events].slice(0, 100),
      })),
    addAlert: (alert) =>
      set((state) => ({
        alerts: [{ ...alert, id: Date.now(), timestamp: new Date().toISOString() }, ...state.alerts].slice(0, 50),
      })),
    removeAlert: (alertId) =>
      set((state) => ({
        alerts: state.alerts.filter((a) => a.id !== alertId),
      })),
    setStatus: (status) => set({ status }),
    setUptime: (uptime) => set({ uptime }),

    // Computed
    getSystemHealth: () => {
      const state = useSystemStore.getState()
      return state.health || {
        db_connected: false,
        kafka_available: false,
        status: 'unknown',
      }
    },
    getRecentEvents: (count = 10) => useSystemStore.getState().events.slice(0, count),
    getCriticalAlerts: () => useSystemStore.getState().alerts.filter((a) => a.severity === 'Critical'),
    reset: () =>
      set({
        health: null,
        metrics: {},
        status: 'operational',
        events: [],
        alerts: [],
        uptime: 0,
        lastUpdate: null,
      }),
  }))
)
