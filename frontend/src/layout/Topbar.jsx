import { useSystemHealth } from '../hooks/useSystemMetrics'
import { StatusBadge } from '../components/common/StatusBadge'

export function Topbar() {
  const { isHealthy } = useSystemHealth()

  const handleLogout = () => {
    localStorage.removeItem('aegis_token')
    window.location.href = '/login'
  }

  return (
    <header className="bg-slate-900 border-b border-slate-700 px-6 py-3 flex items-center justify-between">
      {/* Left */}
      <div className="flex items-center space-x-4">
        <h2 className="text-lg font-semibold text-white">Real-Time Defense Monitor</h2>
      </div>

      {/* Right */}
      <div className="flex items-center space-x-6">
        {/* Status */}
        <StatusBadge status={isHealthy ? 'Connected' : 'Disconnected'} size="sm" />

        {/* Time */}
        <div className="text-sm text-slate-400 font-mono">
          {new Date().toLocaleTimeString()}
        </div>

        {/* User Menu */}
        <div className="flex items-center space-x-3 pl-6 border-l border-slate-700">
          <div className="w-8 h-8 rounded-full bg-cyan-500/20 border border-cyan-500 flex items-center justify-center">
            <span className="text-sm font-bold text-cyan-400">👤</span>
          </div>
          <button
            onClick={handleLogout}
            className="text-sm text-slate-400 hover:text-cyan-400 transition-colors"
          >
            Logout
          </button>
        </div>
      </div>
    </header>
  )
}

export default Topbar
