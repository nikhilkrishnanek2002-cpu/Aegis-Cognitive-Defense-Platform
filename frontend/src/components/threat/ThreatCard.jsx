import { Card } from '../common/Card'
import { StatusBadge } from '../common/StatusBadge'

export function ThreatCard({ threat, onClick }) {
  if (!threat) return null

  const levelColor = {
    Critical: 'text-red-400',
    High: 'text-orange-400',
    Medium: 'text-yellow-400',
    Low: 'text-green-400',
  }

  const threatType = {
    drone: '🛸 Drone',
    aircraft: '✈️ Aircraft',
    missile: '🚀 Missile',
    jamming: '📡 JAMMING',
    spoofing: '🔀 SPOOFING',
  }

  return (
    <div
      onClick={onClick}
      className="bg-slate-800/50 border border-slate-700 hover:border-cyan-700 rounded-lg p-3 cursor-pointer transition-colors"
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1">
          <p className="text-sm font-medium text-white">{threatType[threat.type] || threat.type}</p>
          <p className="text-xs text-slate-400">{threat.id}</p>
        </div>
        <StatusBadge status={threat.level} size="sm" />
      </div>
      <div className="space-y-1 text-xs">
        <div className="flex justify-between">
          <span className="text-slate-500">Distance:</span>
          <span className={levelColor[threat.level]}>{threat.distance || 'N/A'} km</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500">Bearing:</span>
          <span className="text-slate-300">{threat.bearing || 'N/A'}°</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500">Velocity:</span>
          <span className="text-slate-300">{threat.velocity || 'N/A'} m/s</span>
        </div>
      </div>
    </div>
  )
}

export default ThreatCard
