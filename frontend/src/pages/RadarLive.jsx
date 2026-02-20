import { Card } from '../components/common/Card'
import { RadarCanvas } from '../components/radar/RadarCanvas'
import { TargetOverlay } from '../components/radar/TargetOverlay'
import { StatusBadge } from '../components/common/StatusBadge'
import { useRadarStream, useTriggerScan } from '../hooks/useRadarStream'
import { useRadarStore } from '../store/radarStore'

export function RadarLive() {
  // Connect to radar stream
  useRadarStream()

  // Get radar state
  const { isConnected, isScanning, targets, error } = useRadarStore()
  const triggerScan = useTriggerScan()

  const handleScan = async () => {
    await triggerScan()
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

      {/* Error Banner */}
      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-400">
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* Main Radar Display */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Radar Canvas */}
        <div className="lg:col-span-2">
          <Card
            title="Radar Display"
            subtitle={`${targets.length} targets detected`}
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
              <RadarCanvas />
            </div>
          </Card>
        </div>

        {/* Targets Panel */}
        <Card title="Target List" subtitle={`${targets.length} active`}>
          <TargetOverlay />
        </Card>
      </div>

      {/* Target Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card title="Target Classes" subtitle="Classification breakdown">
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">Drones</span>
              <span className="text-cyan-400 font-semibold">{targets.filter((t) => t.type === 'drone').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Aircraft</span>
              <span className="text-cyan-400 font-semibold">{targets.filter((t) => t.type === 'aircraft').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Missiles</span>
              <span className="text-cyan-400 font-semibold">{targets.filter((t) => t.type === 'missile').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Unknown</span>
              <span className="text-cyan-400 font-semibold">{targets.filter((t) => !t.type).length}</span>
            </div>
          </div>
        </Card>

        <Card title="Threat Levels" subtitle="Current distribution">
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">Critical</span>
              <span className="text-red-400 font-semibold">{targets.filter((t) => t.level === 'Critical').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">High</span>
              <span className="text-orange-400 font-semibold">{targets.filter((t) => t.level === 'High').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Medium</span>
              <span className="text-yellow-400 font-semibold">{targets.filter((t) => t.level === 'Medium').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Low</span>
              <span className="text-green-400 font-semibold">{targets.filter((t) => t.level === 'Low').length}</span>
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
              <span className="text-cyan-400 text-xs font-mono">
                {new Date().toLocaleTimeString()}
              </span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}

export default RadarLive
