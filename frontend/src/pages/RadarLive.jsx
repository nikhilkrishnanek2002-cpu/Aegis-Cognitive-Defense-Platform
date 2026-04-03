import { useEffect } from 'react'
import { Card } from '../components/common/Card'
import { RadarCanvas } from '../components/radar/RadarCanvas'
import { TargetOverlay } from '../components/radar/TargetOverlay'
import { StatusBadge } from '../components/common/StatusBadge'
import { useTriggerScan } from '../hooks/useRadarStream'
import { useRadarStore } from '../store/radarStore'
import { useThreatStore, selectEngagementLog } from '../store/threatStore'

export function RadarLive() {
  const { frame, isScanning, error } = useRadarStore()
  // The global websocket sets connection state inside useRadarWebSocket locally
  // We'll consider it connected if we have a frame
  const isConnected = !!frame

  const triggerScan = useTriggerScan()

  const { setThreats, launchInterceptor, neutralizedThreats, engagementLog, activeThreats } = useThreatStore()

  // Sync frame targets to threat store
  useEffect(() => {
    if (frame?.targets) {
      const mappedThreats = frame.targets.map(t => ({
          ...t,
          type: (t.velocity > 50 ? 'aircraft' : 'drone'),
          level: t.threat_level === 'HIGH' ? 'Critical' : (t.threat_level === 'MEDIUM' ? 'Medium' : 'Low'),
          status: 'Active',
          distance: t.y ? Math.round(Math.abs(t.y)) : 0, 
          bearing: t.x ? Math.round(Math.abs((Math.atan2(t.x, t.y) * 180 / Math.PI) + 180)) : 0,
      }))
      setThreats(mappedThreats)
    }
  }, [frame, setThreats])

  const neutralizedIds = neutralizedThreats.map((t) => t.id)

  const handleScan = async () => { await triggerScan() }

  const handleIntercept = (threat) => {
    launchInterceptor(threat.id)
  }

  const formatTime = (date) => {
    if (!date) return '--'
    const d = date instanceof Date ? date : new Date(date)
    return d.toLocaleTimeString()
  }

  const levelColor = {
    Critical: '#ef4444', High: '#f97316', Medium: '#eab308', Low: '#22c55e', Unknown: '#64748b',
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Live Radar</h1>
          <p className="text-slate-400">Real-time target tracking and monitoring</p>
        </div>
        <StatusBadge status={isConnected ? 'Connected' : 'Disconnected'} size="md" />
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-400">
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* Main Radar Display */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card
            title="Radar Display"
            subtitle={`${activeThreats.length} targets detected`}
            action={
              <button
                onClick={handleScan}
                disabled={isScanning}
                className="px-3 py-1 bg-cyan-500 hover:bg-cyan-600 text-white text-sm rounded font-medium disabled:opacity-50 transition-colors"
              >
                {isScanning ? 'Scanning...' : 'Scan'}
              </button>
            }
          >
            <div className="flex justify-center">
              <RadarCanvas missiles={missiles} explosions={explosions} neutralizedIds={neutralizedIds} />
            </div>
          </Card>
        </div>

        <Card title="Target List" subtitle={`${activeThreats.length} active`}>
          <TargetOverlay onIntercept={handleIntercept} />
        </Card>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card title="Target Classes" subtitle="Classification breakdown">
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">Drones</span>
              <span className="text-cyan-400 font-semibold">{activeThreats.filter((t) => t.type === 'drone').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Aircraft</span>
              <span className="text-cyan-400 font-semibold">{activeThreats.filter((t) => t.type === 'aircraft').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Missiles</span>
              <span className="text-cyan-400 font-semibold">{activeThreats.filter((t) => t.type === 'missile').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Unknown</span>
              <span className="text-cyan-400 font-semibold">{activeThreats.filter((t) => !t.type).length}</span>
            </div>
            <div className="flex justify-between border-t border-slate-700 pt-2 mt-2">
              <span className="text-slate-400">Neutralized</span>
              <span className="text-green-400 font-semibold">{neutralizedThreats.length}</span>
            </div>
          </div>
        </Card>

        <Card title="Threat Levels" subtitle="Current distribution">
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">Critical</span>
              <span className="text-red-400 font-semibold">{activeThreats.filter((t) => t.level === 'Critical').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">High</span>
              <span className="text-orange-400 font-semibold">{activeThreats.filter((t) => t.level === 'High').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Medium</span>
              <span className="text-yellow-400 font-semibold">{activeThreats.filter((t) => t.level === 'Medium').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Low</span>
              <span className="text-green-400 font-semibold">{activeThreats.filter((t) => t.level === 'Low').length}</span>
            </div>
          </div>
        </Card>

        <Card title="Connection Status" subtitle="Stream health">
          <div className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-slate-400">WebSocket</span>
              <StatusBadge status={isConnected ? 'Connected' : 'Disconnected'} size="sm" />
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Last Update</span>
              <span className="text-cyan-400 text-xs font-mono">{new Date().toLocaleTimeString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Engagements</span>
              <span className="text-orange-400 font-semibold">{engagementLog.length}</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Engagement Log */}
      <Card
        title="🚀 Engagement Log"
        subtitle={`${engagementLog.length} intercept${engagementLog.length !== 1 ? 's' : ''} logged`}
      >
        {engagementLog.length === 0 ? (
          <div className="text-center py-6 text-slate-500 text-sm">
            <p>No engagements logged. Click <strong className="text-red-400">🚀 INTERCEPT</strong> on a target to launch a countermeasure.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-400 border-b border-slate-700">
                  <th className="text-left py-2 pr-4 font-medium">Time</th>
                  <th className="text-left py-2 pr-4 font-medium">Target ID</th>
                  <th className="text-left py-2 pr-4 font-medium">Type</th>
                  <th className="text-left py-2 pr-4 font-medium">Threat Level</th>
                  <th className="text-left py-2 pr-4 font-medium">Bearing</th>
                  <th className="text-left py-2 font-medium">Distance</th>
                </tr>
              </thead>
              <tbody>
                {engagementLog.map((entry) => (
                  <tr key={entry.id} className="border-b border-slate-800 hover:bg-slate-800/30 transition-colors">
                    <td className="py-2 pr-4 font-mono text-xs text-cyan-400">{formatTime(entry.timestamp)}</td>
                    <td className="py-2 pr-4 text-white font-medium text-xs">{entry.threatId}</td>
                    <td className="py-2 pr-4 text-slate-300 capitalize text-xs">{entry.type}</td>
                    <td className="py-2 pr-4 text-xs font-semibold" style={{ color: levelColor[entry.level] || '#64748b' }}>{entry.level}</td>
                    <td className="py-2 pr-4 text-slate-300 text-xs">{entry.bearing != null ? `${entry.bearing}°` : 'N/A'}</td>
                    <td className="py-2 text-slate-300 text-xs">{entry.distance != null ? `${entry.distance} km` : 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}

export default RadarLive
