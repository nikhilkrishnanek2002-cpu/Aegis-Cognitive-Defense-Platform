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
                <div style={{ height: 80, display: 'flex', alignItems: 'flex-end', gap: 1, padding: '8px 0' }}>
                    {photonic.ttd_vector.map((v, i) => {
                        const max = Math.max(...photonic.ttd_vector)
                        const height = max > 0 ? (v / max) * 60 : 0
                        return <div key={i} style={{ flex: 1, height: height + 'px', background: 'linear-gradient(to top, #3b82f6, #60a5fa)', borderRadius: 2 }} title={`Element ${i}: ${v.toFixed(4)}`} />
                    })}
                </div>
                <p style={{ color: '#64748b', fontSize: 12, margin: 0 }}>{photonic.ttd_vector.length} antenna elements</p>
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
}
