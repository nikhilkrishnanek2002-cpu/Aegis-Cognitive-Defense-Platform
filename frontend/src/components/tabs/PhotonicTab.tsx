import { useRadarStore } from '../../store/radarStore'

export default function PhotonicTab() {
    const { frame } = useRadarStore()
    if (!frame) return <p style={{ color: '#94a3b8' }}>⏳ Waiting for radar data...</p>
    const { photonic } = frame

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
                {[
                    { label: 'Instantaneous Bandwidth', value: `${photonic.bandwidth_mhz.toFixed(2)} MHz` },
                    { label: 'Chirp Slope', value: `${photonic.chirp_slope_thz.toFixed(2)} THz/s` },
                    { label: 'Pulse Width', value: `${photonic.pulse_width_us.toFixed(2)} μs` },
                    { label: 'Noise Power', value: photonic.noise_power.toExponential(3) },
                    { label: 'Clutter Power', value: photonic.clutter_power.toExponential(3) },
                    { label: 'TTD Elements', value: photonic.ttd_vector.length },
                ].map((m) => (
                    <div key={m.label} style={styles.card}>
                        <div style={styles.label}>{m.label}</div>
                        <div style={styles.value}>{String(m.value)}</div>
                    </div>
                ))}
            </div>

            <div style={styles.section}>
                <h3 style={styles.sectionTitle}>📡 True Time Delay (TTD) Beamforming Vector</h3>
                <div style={styles.sciFiContainer}>
                    <div style={{ height: 100, display: 'flex', alignItems: 'flex-end', gap: 2, padding: '16px 8px', position: 'relative' }}>
                        {photonic.ttd_vector.map((v, i) => {
                            // Fix: Use Math.abs to handle negative phase delays, and prevent height: NaN or < 0
                            const max = Math.max(...photonic.ttd_vector.map(Math.abs), 0.0001)
                            const heightPercentage = (Math.abs(v) / max) * 100
                            const isNegative = v < 0

                            return (
                                <div
                                    key={i}
                                    style={{
                                        flex: 1,
                                        height: `${heightPercentage}%`,
                                        background: isNegative ? 'linear-gradient(to top, rgba(239, 68, 68, 0.4), #ef4444)' : 'linear-gradient(to top, rgba(45, 212, 191, 0.4), #2dd4bf)',
                                        borderRadius: '2px 2px 0 0',
                                        boxShadow: isNegative ? '0 0 8px rgba(239, 68, 68, 0.6)' : '0 0 8px rgba(45, 212, 191, 0.6)',
                                        minHeight: '2px',
                                        transition: 'height 0.2s ease-out'
                                    }}
                                    title={`Element ${i}: ${v.toFixed(4)}`}
                                />
                            )
                        })}
                        {/* Center Zero Line */}
                        <div style={{ position: 'absolute', bottom: '16px', left: 0, right: 0, borderBottom: '1px dashed rgba(255,255,255,0.2)' }} />
                    </div>
                </div>
                <p style={{ color: '#64748b', fontSize: 12, margin: '8px 0 0 0', textAlign: 'right', fontFamily: 'monospace' }}>
                    [{photonic.ttd_vector.length} PHASE ELEMENTS]
                </p>
            </div>
        </div>
    )
}

const styles: Record<string, React.CSSProperties> = {
    card: { background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10, padding: 16 },
    label: { fontSize: 12, color: '#64748b', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 },
    value: { fontSize: 20, fontWeight: 700, color: '#60a5fa' },
    section: { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: 20 },
    sectionTitle: { margin: '0 0 14px', color: '#e2e8f0', fontSize: 15, fontWeight: 600 },
    sciFiContainer: {
        background: 'linear-gradient(180deg, rgba(8, 20, 35, 0.9) 0%, rgba(5, 12, 22, 0.9) 100%)',
        border: '1px solid #38bdf8',
        borderRadius: '8px',
        boxShadow: '0 0 10px rgba(56, 189, 248, 0.1), inset 0 0 10px rgba(56, 189, 248, 0.05)',
        position: 'relative',
        overflow: 'hidden',
    },
}
