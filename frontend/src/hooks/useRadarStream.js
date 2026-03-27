import { useEffect, useCallback } from 'react'
import { useRadarStore } from '../store/radarStore'
import { useThreatStore } from '../store/threatStore'
import wsClient from '../services/websocketClient'
import { radar } from '../services/apiClient'

export const useRadarStream = () => {
  const { setFrame, setTargets, addScanHistoryEntry, setConnectionState, setError, clearError } = useRadarStore()

  useEffect(() => {
    let unsubscribeData = null

    const initStream = async () => {
      try {
        await wsClient.connect()
        setConnectionState(true)
        clearError()

        unsubscribeData = wsClient.subscribe('data', (radarData) => {
          if (radarData.frame) {
            setFrame(radarData.frame)
          }
          if (radarData.targets) {
            setTargets(radarData.targets)
            
            // Sync active targets to threatStore for the intercept feature to use
            const { setThreats, threats } = useThreatStore.getState()
            
            // Generate standard threat models from radar targets
            const mappedThreats = radarData.targets.map(t => {
                // Find existing threat to preserve neutralized status
                const existing = threats.find(eth => eth.id === t.id)
                if (existing) return { ...t, ...existing }
                
                return {
                    id: t.id,
                    type: t.threat_level === 'HIGH' ? 'missile' : (t.velocity > 50 ? 'aircraft' : 'drone'),
                    level: t.threat_level === 'HIGH' ? 'Critical' : (t.threat_level === 'MEDIUM' ? 'Medium' : 'Low'),
                    status: 'Active',
                    distance: t.y ? Math.round(Math.abs(t.y)) : 0, 
                    bearing: t.x ? Math.round(Math.abs((Math.atan2(t.x, t.y) * 180 / Math.PI) + 180)) : 0,
                    ...t
                }
            })
            setThreats(mappedThreats)
          }
          if (radarData.scan_id) {
            addScanHistoryEntry({
              scanId: radarData.scan_id,
              timestamp: new Date().toISOString(),
              targetCount: radarData.targets?.length || 0,
            })
          }
        })

        wsClient.subscribe('disconnect', () => {
          setConnectionState(false)
        })

        wsClient.subscribe('error', (error) => {
          setError(`Connection error: ${error.message}`)
        })
      } catch (err) {
        console.error('Failed to connect to radar stream:', err)
        setError('Failed to connect to radar stream')
      }
    }

    initStream()

    return () => {
      if (unsubscribeData) unsubscribeData()
      // Don't disconnect on unmount - stream should persist
    }
  }, [setFrame, setTargets, addScanHistoryEntry, setConnectionState, setError, clearError])
}

export const useTriggerScan = () => {
  const { setScanState, setError, clearError } = useRadarStore()

  const triggerScan = useCallback(async (params = {}) => {
    try {
      setScanState(true)
      clearError()
      const response = await radar.scan(params)
      return response.data
    } catch (err) {
      const message = err.response?.data?.detail || err.message || 'Scan failed'
      setError(message)
      console.error('Scan error:', err)
    } finally {
      setScanState(false)
    }
  }, [setScanState, setError, clearError])

  return triggerScan
}
