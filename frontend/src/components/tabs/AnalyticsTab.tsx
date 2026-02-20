import { useRadarStore } from '../../store/radarStore'

export default function AnalyticsTab() {
    const { frame, trackHistory } = useRadarStore()

    if (!frame) return <p style={{ color: '#94a3b8' }}>⏳ Waiting for radar data...</p>

    const priorityColor: Record<string, string> = { Critical: '#ef4444', High: '#f97316', Medium: '#eab308', Low: '#22c55e' }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* Top Metrics Row */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
                {[
                    { label: 'Detected Target', value: frame.detected, color: priorityColor[frame.priority] || '#60a5fa' },
                    { label: 'Confidence', value: `${(frame.confidence * 100).toFixed(1)}%`, color: '#60a5fa' },
                    { label: 'Priority', value: frame.priority, color: priorityColor[frame.priority] || '#94a3b8' },
                    { label: 'Active Tracks', value: Object.keys(frame.active_tracks).length, color: '#a78bfa' },
                ].map((m) => (
                    <div key={m.label} style={styles.metricCard}>
                        <div style={styles.metricLabel}>{m.label}</div>
                        <div style={{ ...styles.metricValue, color: m.color }}>{String(m.value)}</div>
                    </div>
                ))}
            </div>

            {/* Alert Banner */}
            {frame.is_alert && (
                <div style={styles.alertBanner}>
                    🚨 <strong>THREAT ALERT</strong> — {frame.detected} detected with {(frame.confidence * 100).toFixed(1)}% confidence
                </div>
            )}

            {/* Track History Table */}
            <div style={styles.section}>
                <h3 style={styles.sectionTitle}>🎯 Active Tracks (Kalman Filter)</h3>
                {Object.keys(frame.active_tracks).length === 0 ? (
                    <p style={{ color: '#64748b', fontSize: 13 }}>No active tracks — run more scans to build track history.</p>
                ) : (
                    <table style={styles.table}>
                        <thead>
                            <tr>
                                {['Track ID', 'Position (R, D)', 'Velocity', 'State', 'Confidence'].map(h => (
                                    <th key={h} style={styles.th}>{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {Object.entries(frame.active_tracks).map(([tid, t]) => (
                                <tr key={tid}>
                                    <td style={styles.td}><code style={{ color: '#60a5fa' }}>{tid.slice(0, 8)}</code></td>
                                    <td style={styles.td}>{t.position[0].toFixed(1)}, {t.position[1].toFixed(1)}</td>
                                    <td style={styles.td}>{t.velocity[0].toFixed(2)}, {t.velocity[1].toFixed(2)}</td>
                                    <td style={styles.td}>
                                        <span style={{ color: t.state === 'confirmed' ? '#22c55e' : '#f59e0b', fontWeight: 600 }}>
                                            {t.state}
                                        </span>
                                    </td>
                                    <td style={styles.td}>{(t.confidence * 100).toFixed(1)}%</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>

            {/* Phase Stats */}
            <div style={styles.section}>
                <h3 style={styles.sectionTitle}>📐 Phase Statistics</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                    {['Mean Phase', 'Variance', 'Coherence'].map((label, i) => (
                        <div key={label} style={styles.metricCard}>
                            <div style={styles.metricLabel}>{label}</div>
                            <div style={styles.metricValue}>{frame.meta[i] !== undefined ? frame.meta[i].toFixed(4) : '—'}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* EW Status */}
            <div style={styles.section}>
                <h3 style={styles.sectionTitle}>🛡️ Electronic Warfare Status</h3>
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                    <div style={styles.metricCard}>
                        <div style={styles.metricLabel}>EW Active</div>
                        <div style={{ ...styles.metricValue, color: frame.ew.active ? '#ef4444' : '#22c55e' }}>
                            {frame.ew.active ? '🔴 YES' : '🟢 CLEAR'}
                        </div>
                    </div>
                    <div style={styles.metricCard}>
                        <div style={styles.metricLabel}>Threat Level</div>
                        <div style={styles.metricValue}>{frame.ew.threat_level.toUpperCase()}</div>
                    </div>
                    <div style={styles.metricCard}>
                        <div style={styles.metricLabel}>Active Threats</div>
                        <div style={styles.metricValue}>{frame.ew.num_threats}</div>
                    </div>
                    <div style={styles.metricCard}>
                        <div style={styles.metricLabel}>Cognitive Adaptation</div>
                        <div style={{ ...styles.metricValue, color: frame.cognitive.is_adaptive ? '#a78bfa' : '#64748b' }}>
                            {frame.cognitive.is_adaptive ? `🔄 ${frame.cognitive.suggested_gain_db} dB` : 'Passive'}
                        </div>
                    </div>
                </div>
            </div>

            {/* Track History Summary */}
            {trackHistory.length > 0 && (
                <div style={styles.section}>
                    <h3 style={styles.sectionTitle}>📈 Track History ({trackHistory.length} frames buffered)</h3>
                    <div style={{ overflow: 'auto', maxHeight: 200 }}>
                        <table style={styles.table}>
                            <thead>
                                <tr>
                                    <th style={styles.th}>Frame</th>
                                    <th style={styles.th}>Tracks Active</th>
                                    <th style={styles.th}>Timestamp</th>
                                </tr>
                            </thead>
                            <tbody>
                                {trackHistory.slice(-10).reverse().map((h, i) => (
                                    <tr key={i}>
                                        <td style={styles.td}>#{trackHistory.length - i}</td>
                                        <td style={styles.td}>{Object.keys(h.tracks).length}</td>
                                        <td style={styles.td}>{new Date(h.time * 1000).toLocaleTimeString()}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    )
}

const styles: Record<string, React.CSSProperties> = {
    metricCard: { background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10, padding: '16px', flex: 1 },
    metricLabel: { fontSize: 12, color: '#64748b', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 },
    metricValue: { fontSize: 22, fontWeight: 700, color: '#f1f5f9' },
    alertBanner: { background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.4)', borderRadius: 10, padding: '14px 18px', color: '#fca5a5', fontSize: 14 },
    section: { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: 20 },
    sectionTitle: { margin: '0 0 14px', color: '#e2e8f0', fontSize: 15, fontWeight: 600 },
    table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
    th: { textAlign: 'left', padding: '8px 10px', color: '#64748b', borderBottom: '1px solid rgba(255,255,255,0.08)', fontSize: 12, textTransform: 'uppercase', letterSpacing: 0.5 },
    td: { padding: '8px 10px', color: '#e2e8f0', borderBottom: '1px solid rgba(255,255,255,0.05)' },
}
