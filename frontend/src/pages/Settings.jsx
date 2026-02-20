import { Card } from '../components/common/Card'

export function Settings() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Settings</h1>
        <p className="text-slate-400">System configuration and preferences</p>
      </div>

      {/* General Settings */}
      <Card title="General Settings">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">System Name</label>
            <input
              type="text"
              defaultValue="Aegis Defense System"
              className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Threat Level Threshold</label>
            <select className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white text-sm">
              <option>Low</option>
              <option>Medium</option>
              <option>High</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Radar Settings */}
      <Card title="Radar Configuration">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Scan Interval (seconds)</label>
            <input
              type="number"
              defaultValue="5"
              className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white text-sm"
            />
          </div>
          <div className="flex items-center space-x-3">
            <input type="checkbox" defaultChecked className="w-4 h-4" />
            <label className="text-sm text-slate-300">Enable Continuous Scanning</label>
          </div>
        </div>
      </Card>

      {/* Notification Settings */}
      <Card title="Notifications">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm text-slate-300">Critical Threats</label>
            <input type="checkbox" defaultChecked className="w-4 h-4" />
          </div>
          <div className="flex items-center justify-between">
            <label className="text-sm text-slate-300">System Alerts</label>
            <input type="checkbox" defaultChecked className="w-4 h-4" />
          </div>
          <div className="flex items-center justify-between">
            <label className="text-sm text-slate-300">Connected Status</label>
            <input type="checkbox" className="w-4 h-4" />
          </div>
        </div>
      </Card>

      {/* Action Buttons */}
      <div className="flex space-x-4">
        <button className="px-6 py-2 bg-cyan-500 hover:bg-cyan-600 text-white text-sm font-medium rounded transition-colors">
          Save Changes
        </button>
        <button className="px-6 py-2 bg-slate-700 hover:bg-slate-600 text-white text-sm font-medium rounded transition-colors">
          Reset to Defaults
        </button>
      </div>
    </div>
  )
}

export default Settings
