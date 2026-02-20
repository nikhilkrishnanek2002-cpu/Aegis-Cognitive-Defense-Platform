import { Card } from '../components/common/Card'

export function ModelMonitor() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Model Monitor</h1>
        <p className="text-slate-400">AI/ML model performance tracking</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card title="Model Accuracy">
          <p className="text-4xl font-bold text-green-400">94.2%</p>
        </Card>
        <Card title="Precision">
          <p className="text-4xl font-bold text-blue-400">96.1%</p>
        </Card>
        <Card title="Recall">
          <p className="text-4xl font-bold text-purple-400">92.3%</p>
        </Card>
        <Card title="F1-Score">
          <p className="text-4xl font-bold text-cyan-400">94.1%</p>
        </Card>
      </div>

      <Card title="Performance Metrics" subtitle="Classification performance">
        <div className="space-y-3">
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-sm text-slate-400">Accuracy</span>
              <span className="text-sm text-cyan-400">94.2%</span>
            </div>
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <div className="h-full bg-cyan-500" style={{ width: '94.2%' }} />
            </div>
          </div>
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-sm text-slate-400">Precision</span>
              <span className="text-sm text-blue-400">96.1%</span>
            </div>
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <div className="h-full bg-blue-500" style={{ width: '96.1%' }} />
            </div>
          </div>
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-sm text-slate-400">Recall</span>
              <span className="text-sm text-purple-400">92.3%</span>
            </div>
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <div className="h-full bg-purple-500" style={{ width: '92.3%' }} />
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}

export default ModelMonitor
