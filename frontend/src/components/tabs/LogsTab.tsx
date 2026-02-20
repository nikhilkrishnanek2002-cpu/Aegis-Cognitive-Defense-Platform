import { useRadarStore } from '../../store/radarStore'
import { useState } from 'react'

export default function LogsTab() {
    const { trackHistory } = useRadarStore()
    const [filter, setFilter] = useState('')

    const events = trackHistory.map((entry, i) => ({
        time: new Date(entry.time * 1000).toLocaleTimeString(),
        tracks: Object.keys(entry.tracks).length,
        confirmed: Object.values(entry.tracks).filter((t) => t.state === 'confirmed').length,
        frame: i + 1,
    })).reverse()

    const filtered = filter
        ? events.filter((e) => e.time.includes(filter) || String(e.tracks).includes(filter))
        : events

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={styles.section}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                    <h3 style={styles.title}>📋 Detection History ({trackHistory.length} events)</h3>
                    <input
                        placeholder="Filter..."
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        style={styles.filterInput}
                    />
                </div>

                {filtered.length === 0 ? (
                    <p style={{ color: '#64748b', fontSize: 13 }}>No detection history yet. Live data will populate automatically.</p>
                ) : (
                    <div style={{ overflowY: 'auto', maxHeight: 400 }}>
                        <table style={styles.table}>
                            <thead>
                                <tr>
                                    {['Frame', 'Time', 'Total Tracks', 'Confirmed Tracks'].map(h => (
                                        <th key={h} style={styles.th}>{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {filtered.slice(0, 50).map((e, i) => (
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
    title: { margin: 0, color: '#e2e8f0', fontSize: 15, fontWeight: 600 },
    table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
    th: { textAlign: 'left', padding: '8px 10px', color: '#64748b', borderBottom: '1px solid rgba(255,255,255,0.08)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5 },
    td: { padding: '8px 10px', color: '#e2e8f0', borderBottom: '1px solid rgba(255,255,255,0.05)' },
    filterInput: { padding: '6px 12px', background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#e2e8f0', fontSize: 13, width: 160 },
}
