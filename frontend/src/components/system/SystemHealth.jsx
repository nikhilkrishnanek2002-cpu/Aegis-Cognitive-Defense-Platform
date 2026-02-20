import { Card } from '../common/Card'
import { StatusBadge } from '../common/StatusBadge'
import { useSystemHealth } from '../../hooks/useSystemMetrics'

export function SystemHealth() {
  const { health, isHealthy } = useSystemHealth()

  if (!health) {
    return (
      <Card title="System Health" subtitle="Loading...">
        <div className="h-20 animate-pulse bg-slate-700 rounded" />
      </Card>
    )
  }

  const metrics = [
    { label: 'Database', value: health.db_connected ? 'Connected' : 'Disconnected' },
    { label: 'Kafka', value: health.kafka_available ? 'Available' : 'Unavailable' },
    { label: 'Status', value: health.status || 'Unknown' },
  ]

  return (
    <Card
      title="System Health"
      action={<StatusBadge status={isHealthy ? 'Connected' : 'Disconnected'} size="sm" />}
    >
      <div className="space-y-3">
        {metrics.map((metric) => (
          <div key={metric.label} className="flex items-center justify-between text-sm">
            <span className="text-slate-400">{metric.label}</span>
            <StatusBadge status={metric.value} size="sm" />
          </div>
        ))}
      </div>
    </Card>
  )
}

export default SystemHealth
