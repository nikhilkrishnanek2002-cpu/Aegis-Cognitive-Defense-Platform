import { useRadarStore } from '../../store/radarStore'
import { useState, useEffect } from 'react'

interface LogEvent {
    timestamp: string
    level: 'INFO' | 'WARNING' | 'ERROR'
    module: string
    message: string
    details?: string
}

export default function LogsTab() {
    const { trackHistory, frame } = useRadarStore()
    const [filter, setFilter] = useState('')
    const [logs, setLogs] = useState<LogEvent[]>([])

    // Generate realistic logs from frame data
    useEffect(() => {
        const newLogs: LogEvent[] = []

        if (frame) {
            newLogs.push({
                timestamp: new Date().toLocaleTimeString(),
                level: 'INFO',
                module: 'RADAR',
                message: `Scan complete: ${frame.num_detections} detections`,
                details: `Threshold: ${(frame.threshold * 100).toFixed(1)}%`
            })

            if (frame.is_alert) {
                newLogs.push({
                    timestamp: new Date().toLocaleTimeString(),
                    level: 'WARNING',
                    module: 'THREAT',
                    message: `Alert: Potential ${frame.detected} detected`,
                    details: `Confidence: ${(frame.confidence * 100).toFixed(1)}%`
                })
            }

            const activeTracks = Object.keys(frame.active_tracks ?? {}).length
            if (activeTracks > 0) {
                newLogs.push({
                    timestamp: new Date().toLocaleTimeString(),
                    level: 'INFO',
                    module: 'TRACKING',
                    message: `${activeTracks} active track${activeTracks !== 1 ? 's' : ''}`,
                    details: `EW: ${frame.ew?.active ? 'ACTIVE' : 'CLEAR'}`
                })
            }
        }

        // Combine with history
        setLogs(prev => [...newLogs, ...prev].slice(0, 100))
    }, [frame])

    const filtered = filter
        ? logs.filter((e) =>
            e.message.toLowerCase().includes(filter.toLowerCase()) ||
            e.module.toLowerCase().includes(filter.toLowerCase()) ||
            e.level.includes(filter.toUpperCase())
        )
        : logs

    const events = (trackHistory || []).map((entry, i) => ({
        time: new Date((entry?.time || 0) * 1000).toLocaleTimeString(),
        tracks: Object.keys(entry?.tracks || {}).length,
        confirmed: Object.values(entry?.tracks || {}).filter((t: any) => t.state === 'confirmed').length,
        frame: i + 1,
    })).reverse()

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* System Logs */}
            <div style={styles.section}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                    <h3 style={styles.title}>📊 System Logs ({logs.length} events)</h3>
                    <input
                        placeholder="Filter by message/module/level..."
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        style={styles.filterInput}
                    />
                </div>

                {filtered.length === 0 ? (
                    <p style={{ color: '#64748b', fontSize: 13 }}>No logs. System events will populate automatically.</p>
                ) : (
                    <div style={{ overflowY: 'auto', maxHeight: 400 }}>
                        <table style={styles.table}>
                            <thead>
                                <tr>
                                    {['Time', 'Level', 'Module', 'Message', 'Details'].map(h => (
                                        <th key={h} style={styles.th}>{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {filtered.slice(0, 50).map((log, i) => {
                                    const levelColor = log.level === 'ERROR' ? '#fca5a5' : log.level === 'WARNING' ? '#fbbf24' : '#86efac'
                                    return (
                                        <tr key={i} style={{ background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)' }}>
                                            <td style={styles.td}><span style={{ fontSize: 11 }}>{log.timestamp}</span></td>
                                            <td style={{ ...styles.td, color: levelColor, fontWeight: 600 }}>{log.level}</td>
                                            <td style={{ ...styles.td, color: '#60a5fa' }}>{log.module}</td>
                                            <td style={styles.td}>{log.message}</td>
                                            <td style={{ ...styles.td, color: '#94a3b8', fontSize: 12 }}>{log.details || '—'}</td>
                                        </tr>
                                    )
                                })}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {/* Detection History */}
            <div style={styles.section}>
                <h3 style={styles.title}>📡 Detection History ({events.length} frames)</h3>
                {events.length === 0 ? (
                    <p style={{ color: '#64748b', fontSize: 13 }}>No detection history. Radar data will populate automatically.</p>
                ) : (
                    <div style={{ overflowY: 'auto', maxHeight: 300 }}>
                        <table style={styles.table}>
                            <thead>
                                <tr>
                                    {['Frame', 'Time', 'Total Tracks', 'Confirmed'].map(h => (
                                        <th key={h} style={styles.th}>{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {events.slice(0, 30).map((e, i) => (
                                    <tr key={i} style={{ background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)' }}>
                                        <td style={styles.td}>#{e.frame}</td>
                                        <td style={styles.td}>{e.time}</td>
                                        <td style={styles.td}>
                                            <span style={{ color: e.tracks > 0 ? '#60a5fa' : '#64748b' }}>{e.tracks}</span>
                                        </td>
                                        <td style={styles.td}>
                                            <span style={{ color: e.confirmed > 0 ? '#22c55e' : '#64748b' }}>{e.confirmed}</span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    )
}

const styles: Record<string, React.CSSProperties> = {
    section: { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: 20 },
    title: { margin: 0, color: '#e2e8f0', fontSize: 15, fontWeight: 600, marginBottom: 12 },
    table: { width: '100%', borderCollapse: 'collapse', fontSize: 12 },
    th: { textAlign: 'left', padding: '10px 12px', color: '#64748b', borderBottom: '1px solid rgba(255,255,255,0.08)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 600 },
    td: { padding: '10px 12px', color: '#e2e8f0', borderBottom: '1px solid rgba(255,255,255,0.05)' },
    filterInput: { padding: '8px 12px', background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#e2e8f0', fontSize: 13, width: 280 },
}
