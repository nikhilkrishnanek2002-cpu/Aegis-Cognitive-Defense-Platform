import { memo } from 'react'
import { Card } from '../common/Card'
import { StatusBadge } from '../common/StatusBadge'
import { useThreatStore, selectActiveThreats } from '../../store/threatStore'

function ThreatTableComponent() {
  // Use optimized selector to only re-render when activeThreats changes
  const activeThreats = useThreatStore(selectActiveThreats)

  if (activeThreats.length === 0) {
    return (
      <Card title="Active Threats" subtitle={`${activeThreats.length} threats detected`}>
        <div className="py-8 text-center text-slate-400">No active threats</div>
      </Card>
    )
  }

  return (
    <Card title="Active Threats" subtitle={`${activeThreats.length} threats detected`}>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="border-b border-slate-700">
            <tr>
              <th className="px-3 py-2 text-left text-slate-400 font-medium">Type</th>
              <th className="px-3 py-2 text-left text-slate-400 font-medium">Distance</th>
              <th className="px-3 py-2 text-left text-slate-400 font-medium">Bearing</th>
              <th className="px-3 py-2 text-left text-slate-400 font-medium">Level</th>
              <th className="px-3 py-2 text-left text-slate-400 font-medium">Time</th>
            </tr>
          </thead>
          <tbody>
            {activeThreats.map((threat) => (
              <tr key={threat.id} className="border-b border-slate-700 hover:bg-slate-700/30 transition">
                <td className="px-3 py-2 text-white">{threat.type}</td>
                <td className="px-3 py-2 text-slate-300">{threat.distance} km</td>
                <td className="px-3 py-2 text-slate-300">{threat.bearing}°</td>
                <td className="px-3 py-2">
                  <StatusBadge status={threat.level} size="sm" />
                </td>
                <td className="px-3 py-2 text-slate-500 text-xs">
                  {new Date(threat.timestamp || Date.now()).toLocaleTimeString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

// Only re-render if threat count changes (Zustand handles the rest)
function areThreatPropsEqual(prev, next) {
  return true
}

export const ThreatTable = memo(ThreatTableComponent, areThreatPropsEqual)

export default ThreatTable
