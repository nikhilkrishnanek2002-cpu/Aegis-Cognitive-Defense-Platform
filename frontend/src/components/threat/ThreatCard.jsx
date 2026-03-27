import { StatusBadge } from '../common/StatusBadge'

export function ThreatCard({ threat, onClick, onIntercept }) {
  if (!threat) return null

  const isNeutralized = threat.status === 'Neutralized'

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

  const handleIntercept = (e) => {
    e.stopPropagation()
    if (!isNeutralized && onIntercept) onIntercept(threat)
  }

  return (
    <div
      onClick={onClick}
      className="bg-slate-800/50 border border-slate-700 hover:border-cyan-700 rounded-lg p-3 cursor-pointer transition-colors"
      style={{ opacity: isNeutralized ? 0.65 : 1 }}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1">
          <p className="text-sm font-medium text-white">{threatType[threat.type] || threat.type}</p>
          <p className="text-xs text-slate-400">{threat.id}</p>
        </div>
        <StatusBadge status={isNeutralized ? 'Neutralized' : threat.level} size="sm" />
      </div>
      <div className="space-y-1 text-xs mb-3">
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

      {/* Intercept Button */}
      <button
        onClick={handleIntercept}
        disabled={isNeutralized}
        style={{
          width: '100%',
          padding: '4px 0',
          borderRadius: '6px',
          fontSize: '11px',
          fontWeight: 'bold',
          letterSpacing: '0.05em',
          cursor: isNeutralized ? 'not-allowed' : 'pointer',
          border: isNeutralized ? '1px solid #334155' : '1px solid #ef4444',
          background: isNeutralized
            ? 'rgba(30,41,59,0.5)'
            : 'linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05))',
          color: isNeutralized ? '#64748b' : '#ef4444',
          boxShadow: isNeutralized ? 'none' : '0 0 8px rgba(239,68,68,0.3)',
          transition: 'all 0.2s ease',
        }}
      >
        {isNeutralized ? '✓ NEUTRALIZED' : '🚀 INTERCEPT'}
      </button>
    </div>
  )
}

export default ThreatCard
