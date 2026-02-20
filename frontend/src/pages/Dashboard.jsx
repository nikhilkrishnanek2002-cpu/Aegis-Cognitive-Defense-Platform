import { useEffect } from 'react'
import { Card } from '../components/common/Card'
import { SystemHealth } from '../components/system/SystemHealth'
import { ThreatTable } from '../components/threat/ThreatTable'
import { useSystemMetrics } from '../hooks/useSystemMetrics'
import { useRadarStore } from '../store/radarStore'
import { useThreatStore } from '../store/threatStore'
import { useSystemStore } from '../store/systemStore'

export function Dashboard() {
  // Initialize metrics polling
  useSystemMetrics()

  // Get store data
  const radarTargets = useRadarStore((state) => state.targets)
  const scanHistory = useRadarStore((state) => state.scanHistory)
  const threats = useThreatStore((state) => state.activeThreats)
  const events = useSystemStore((state) => state.getRecentEvents(5))

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Dashboard</h1>
        <p className="text-slate-400">Real-time defense system overview</p>
      </div>

      {/* Top Row - Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card title="Active Threats" subtitle={`${threats.length} detected`}>
          <p className="text-4xl font-bold text-red-400">{threats.length}</p>
        </Card>
        <Card title="Radar Targets" subtitle={`${radarTargets.length} tracked`}>
          <p className="text-4xl font-bold text-cyan-400">{radarTargets.length}</p>
        </Card>
        <Card title="Recent Scans" subtitle={`${scanHistory.length} total`}>
          <p className="text-4xl font-bold text-green-400">{scanHistory.length}</p>
        </Card>
        <Card title="System Status" subtitle="Live">
          <p className="text-4xl font-bold text-blue-400">✓</p>
        </Card>
      </div>

      {/* Middle Row - System Health & Threats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <SystemHealth />

        <div className="lg:col-span-2">
          <ThreatTable />
        </div>
      </div>

      {/* Bottom Row - Recent Events */}
      <Card title="Recent Events" subtitle="Last 5 activities">
        <div className="space-y-2">
          {events.length === 0 ? (
            <p className="text-slate-400 text-sm">No recent events</p>
          ) : (
            events.map((event) => (
              <div
                key={event.id}
                className="flex items-center justify-between p-2 bg-slate-700/20 rounded border border-slate-700/50"
              >
                <div>
                  <p className="text-sm text-white">{event.message}</p>
                  <p className="text-xs text-slate-500">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </p>
                </div>
                <span className="text-lg">
                  {event.level === 'error' && '❌'}
                  {event.level === 'warning' && '⚠️'}
                  {event.level === 'info' && 'ℹ️'}
                  {event.level === 'success' && '✅'}
                </span>
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  )
}

export default Dashboard
