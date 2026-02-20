import { useThreatStore } from '../../store/threatStore'
import { ThreatCard } from '../threat/ThreatCard'

export function TargetOverlay() {
  const { activeThreats, setSelectedThreat } = useThreatStore()

  if (activeThreats.length === 0) {
    return (
      <div className="text-center py-8 text-slate-400">
        <p className="text-sm">No active threats</p>
      </div>
    )
  }

  return (
    <div className="space-y-2 max-h-96 overflow-y-auto">
      {activeThreats.map((threat) => (
        <ThreatCard
          key={threat.id}
          threat={threat}
          onClick={() => setSelectedThreat(threat)}
        />
      ))}
    </div>
  )
}

export default TargetOverlay
