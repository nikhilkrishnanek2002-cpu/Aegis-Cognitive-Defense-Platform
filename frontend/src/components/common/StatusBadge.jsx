export function StatusBadge({ status, size = 'md' }) {
  const statusConfig = {
    Active: { bg: 'bg-green-900/30', border: 'border-green-700', dot: 'bg-green-500', text: 'text-green-400' },
    Inactive: { bg: 'bg-slate-900/30', border: 'border-slate-700', dot: 'bg-slate-500', text: 'text-slate-400' },
    Critical: { bg: 'bg-red-900/30', border: 'border-red-700', dot: 'bg-red-500', text: 'text-red-400' },
    Warning: { bg: 'bg-yellow-900/30', border: 'border-yellow-700', dot: 'bg-yellow-500', text: 'text-yellow-400' },
    Connected: { bg: 'bg-cyan-900/30', border: 'border-cyan-700', dot: 'bg-cyan-500', text: 'text-cyan-400' },
    Disconnected: { bg: 'bg-red-900/30', border: 'border-red-700', dot: 'bg-red-500', text: 'text-red-400' },
  }

  const config = statusConfig[status] || statusConfig.Inactive
  const sizeClasses = size === 'sm' ? 'px-2 py-1 text-xs' : 'px-3 py-1.5 text-sm'

  return (
    <div className={`inline-flex items-center ${sizeClasses} rounded-full border ${config.bg} ${config.border}`}>
      <div className={`w-2 h-2 rounded-full ${config.dot} mr-2 animate-pulse`} />
      <span className={`font-medium ${config.text}`}>{status}</span>
    </div>
  )
}

export default StatusBadge
