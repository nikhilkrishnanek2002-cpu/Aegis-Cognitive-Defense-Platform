import { Card } from '../components/common/Card'

export function EWControl() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Electronic Warfare Control</h1>
        <p className="text-slate-400">EW signal detection and counter-measures</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card title="Active EW Signals">
          <p className="text-4xl font-bold text-orange-400">0</p>
        </Card>
        <Card title="Jamming Detected">
          <p className="text-4xl font-bold text-red-400">0</p>
        </Card>
        <Card title="Spoofing Attempts">
          <p className="text-4xl font-bold text-yellow-400">0</p>
        </Card>
      </div>

      <Card title="EW Signals" subtitle="Real-time detection">
        <div className="text-center py-8 text-slate-400">
          <p>No active EW signals detected</p>
        </div>
      </Card>
    </div>
  )
}

export default EWControl
