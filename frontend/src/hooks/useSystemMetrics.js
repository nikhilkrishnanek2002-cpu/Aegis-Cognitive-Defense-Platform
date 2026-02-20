import { useEffect, useCallback } from 'react'
import { useSystemStore } from '../store/systemStore'
import { admin, metrics } from '../services/apiClient'

export const useSystemMetrics = (interval = 5000) => {
  const { setHealth, setMetrics, setStatus, addEvent } = useSystemStore()

  const fetchMetrics = useCallback(async () => {
    try {
      const [healthRes, metricsRes] = await Promise.all([admin.health(), metrics.report().catch(() => null)])

      if (healthRes.data) {
        setHealth(healthRes.data)
        setStatus(healthRes.data.status || 'operational')

        // Add event if status changed
        if (healthRes.data.status !== 'operational') {
          addEvent({
            type: 'system',
            level: 'warning',
            message: `System status: ${healthRes.data.status}`,
          })
        }
      }

      if (metricsRes?.data) {
        setMetrics(metricsRes.data)
      }
    } catch (err) {
      console.error('Failed to fetch system metrics:', err)
      addEvent({
        type: 'error',
        level: 'error',
        message: 'Failed to fetch system metrics',
      })
    }
  }, [setHealth, setMetrics, setStatus, addEvent])

  useEffect(() => {
    // Fetch immediately
    fetchMetrics()

    // Then poll at interval
    const timer = setInterval(fetchMetrics, interval)

    return () => clearInterval(timer)
  }, [fetchMetrics, interval])

  return fetchMetrics
}

export const useSystemHealth = () => {
  const { health, status } = useSystemStore()

  return {
    isHealthy: status === 'operational',
    health,
    status,
  }
}
