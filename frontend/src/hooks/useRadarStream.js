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
