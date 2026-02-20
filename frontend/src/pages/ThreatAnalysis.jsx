import { Card } from '../components/common/Card'
import { ThreatTable } from '../components/threat/ThreatTable'
import { useThreatStore } from '../store/threatStore'

export function ThreatAnalysis() {
  const { activeThreats, threatHistory } = useThreatStore()

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Threat Analysis</h1>
        <p className="text-slate-400">Detailed threat assessment and history</p>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card title="Active Threats">
          <p className="text-4xl font-bold text-red-400">{activeThreats.length}</p>
        </Card>
        <Card title="Critical">
          <p className="text-4xl font-bold text-red-600">
            {activeThreats.filter((t) => t.level === 'Critical').length}
          </p>
        </Card>
        <Card title="High">
          <p className="text-4xl font-bold text-orange-400">
            {activeThreats.filter((t) => t.level === 'High').length}
          </p>
        </Card>
        <Card title="Total History">
          <p className="text-4xl font-bold text-blue-400">{threatHistory.length}</p>
        </Card>
      </div>

      {/* Threat Table */}
      <ThreatTable />

      {/* History */}
      <Card title="Threat History" subtitle="Last 20 detections">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-700">
              <tr>
                <th className="px-3 py-2 text-left text-slate-400">Time</th>
                <th className="px-3 py-2 text-left text-slate-400">Type</th>
                <th className="px-3 py-2 text-left text-slate-400">Status</th>
              </tr>
            </thead>
            <tbody>
              {threatHistory.slice(0, 20).map((threat) => (
                <tr key={threat.id} className="border-b border-slate-700 hover:bg-slate-700/30">
                  <td className="px-3 py-2 text-slate-300">
                    {new Date(threat.timestamp || Date.now()).toLocaleTimeString()}
                  </td>
                  <td className="px-3 py-2 text-white font-medium">{threat.type}</td>
                  <td className="px-3 py-2 text-slate-400">{threat.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

export default ThreatAnalysis
