import { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import { useRadarStore } from '../../store/radarStore'

// @ts-ignore - apiClient is a JS file
import { radar } from '../../services/apiClient'

interface GradCAMData {
    scan_id: string
    heatmap: number[][]
    heatmap_shape: [number, number]
    target_class: string
    confidence: number
    image_path: string
}

export default function XAITab() {
    const { frame } = useRadarStore()
    const [gradcamData, setGradcamData] = useState<GradCAMData | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    useEffect(() => {
        if (frame?.xai) {
            setGradcamData(frame.xai as GradCAMData)
            setError('')
        }
    }, [frame])

    const handleGenerateGradCAM = async () => {
        if (!frame?.detected) {
            setError('No radar frame available. Run a scan first.')
            return
        }
        setLoading(true)
        setError('')
        try {
            // Bypass any frontend Axios interceptors or mocks by using raw fetch
            const token = localStorage.getItem('aegis_token')
            const res = await fetch('/api/radar/scan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                body: JSON.stringify({
                    target: frame.detected,
                    distance: 200,
                    gain_db: 15,
                })
            })

            if (!res.ok) {
                throw new Error(`HTTP Error ${res.status}: ${res.statusText}`)
            }

            const data = await res.json()
            if (data && data.xai) {
                setGradcamData(data.xai)
                setError('')
            } else {
                const keys = Object.keys(data).join(', ')
                throw new Error(`No XAI data! Received keys: [${keys}]`)
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Unknown error'
            setError(`Failed to generate Grad-CAM: ${message}`)
            console.error('Error generating Grad-CAM:', err)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div style={{ color: '#94a3b8', display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={styles.section}>
                <h3 style={styles.title}>🧠 Explainable AI — Grad-CAM Heatmaps</h3>
                <p style={styles.description}>
                    Grad-CAM visualizations highlight which regions of the radar map influenced the AI classification decision.
                    Red areas = high influence, Blue areas = low influence.
                </p>

                {
                    error && (
                        <div style={styles.error}>
                            ⚠️ {error}
                        </div>
                    )
                }

                {
                    !gradcamData ? (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 16 }}>
                            {['RD Map Analysis', 'Target Influence'].map((label) => (
                                <div key={label} style={styles.placeholder}>
                                    <div style={styles.icon}>🎨</div>
                                    <div style={styles.placeholderText}>{label}</div>
                                    <p style={styles.placeholderSub}>
                                        Grad-CAM visualization will appear here after generating a scan.
                                        {frame?.detected ? ' Click "Generate" to create visualization.' : ' No radar data available.'}
                                    </p>
                                    <button
                                        onClick={handleGenerateGradCAM}
                                        disabled={loading || !frame?.detected}
                                        style={{
                                            ...styles.button,
                                            opacity: (loading || !frame?.detected) ? 0.5 : 1,
                                            cursor: (loading || !frame?.detected) ? 'not-allowed' : 'pointer',
                                        }}
                                    >
                                        {loading ? '⏳ Generating...' : '▶ Generate Grad-CAM'}
                                    </button>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div style={{ marginTop: 20 }}>
                            <div style={styles.dataHeader}>
                                <span style={{ color: '#60a5fa' }}>🎯 Target: {gradcamData.target_class}</span>
                                <span style={{ color: '#22c55e' }}>✓ Confidence: {(gradcamData.confidence * 100).toFixed(1)}%</span>
                                <span style={{ color: '#a78bfa' }}>📍 Scan: {gradcamData.scan_id}</span>
                            </div>

                            <div style={{ marginTop: 16 }}>
                                <h4 style={styles.chartTitle}>Grad-CAM Activation Heatmap</h4>
                                {gradcamData.heatmap && (
                                    <Plot
                                        data={[{
                                            z: gradcamData.heatmap,
                                            type: 'heatmap' as const,
                                            colorscale: [[0, '#000000'], [0.5, '#FF6600'], [1, '#FFFF00']],
                                            showscale: true,
                                            colorbar: { title: { text: 'Activation Strength' } },
                                        } as any]}
                                        layout={{
                                            title: { text: `Grad-CAM: ${gradcamData.target_class}` },
                                            xaxis: { title: { text: 'Range (bins)' } },
                                            yaxis: { title: { text: 'Doppler (bins)' } },
                                            plot_bgcolor: 'rgba(15, 23, 42, 0.5)',
                                            paper_bgcolor: 'rgba(15, 23, 42, 0.3)',
                                            font: { color: '#e2e8f0' },
                                            margin: { t: 40, r: 100, b: 60, l: 60 }
                                        } as any}
                                        style={{ width: '100%', height: 400 }}
                                        config={{ responsive: true }}
                                    />
                                )}
                            </div>

                            <div style={{ marginTop: 20, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                                <div style={styles.infoCard}>
                                    <div style={styles.infoLabel}>Classification Confidence</div>
                                    <div style={styles.infoValue}>{(gradcamData.confidence * 100).toFixed(1)}%</div>
                                    <div style={{ width: '100%', background: 'rgba(255,255,255,0.1)', borderRadius: 4, height: 6, marginTop: 8, overflow: 'hidden' }}>
                                        <div style={{ background: '#60a5fa', height: '100%', width: `${gradcamData.confidence * 100}%`, transition: 'width 0.3s' }} />
                                    </div>
                                </div>
                                <div style={styles.infoCard}>
                                    <div style={styles.infoLabel}>Heatmap Resolution</div>
                                    <div style={styles.infoValue}>{gradcamData.heatmap_shape[0]}×{gradcamData.heatmap_shape[1]}</div>
                                </div>
                            </div>

                            <button
                                onClick={handleGenerateGradCAM}
                                disabled={loading}
                                style={{
                                    ...styles.regenerateButton,
                                    opacity: loading ? 0.6 : 1,
                                    marginTop: 20
                                }}
                            >
                                {loading ? '⏳ Regenerating...' : '🔄 Regenerate Grad-CAM'}
                            </button>
                        </div>
                    )
                }
            </div >

            <div style={styles.infoSection}>
                <h4 style={styles.infoTitle}>📖 How Grad-CAM Works</h4>
                <ul style={styles.infoList}>
                    <li>Red regions show areas that strongly indicate the predicted target class</li>
                    <li>Blue regions show areas that oppose the prediction</li>
                    <li>Green regions show neutral areas</li>
                    <li>Use this to understand which radar features the AI model relies on</li>
                </ul>
            </div>
        </div >
    )
}

const styles: Record<string, React.CSSProperties> = {
    section: { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: 20 },
    title: { margin: '0 0 8px', color: '#e2e8f0', fontSize: 16, fontWeight: 700 },
    description: { color: '#94a3b8', fontSize: 13, lineHeight: 1.5, margin: '0 0 12px' },
    error: { background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 8, padding: 12, color: '#fca5a5', fontSize: 13, marginBottom: 16 },
    placeholder: {
        background: 'rgba(96, 165, 250, 0.05)',
        border: '2px dashed rgba(96, 165, 250, 0.3)',
        borderRadius: 12,
        padding: 32,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 12,
        textAlign: 'center',
    },
    icon: { fontSize: 48 },
    placeholderText: { color: '#e2e8f0', fontWeight: 600, fontSize: 14 },
    placeholderSub: { color: '#64748b', fontSize: 12, margin: 0 },
    button: {
        marginTop: 12,
        padding: '10px 20px',
        background: 'rgba(59, 130, 246, 0.8)',
        border: 'none',
        borderRadius: 8,
        color: '#fff',
        fontWeight: 600,
        cursor: 'pointer',
        fontSize: 13,
    },
    dataHeader: { display: 'flex', gap: 20, flexWrap: 'wrap', padding: 12, background: 'rgba(255,255,255,0.02)', borderRadius: 8, fontSize: 12 },
    chartTitle: { margin: 0, color: '#e2e8f0', fontSize: 14, fontWeight: 600, marginBottom: 12 },
    infoCard: { background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, padding: 12 },
    infoLabel: { fontSize: 11, color: '#64748b', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 },
    infoValue: { fontSize: 18, fontWeight: 700, color: '#60a5fa' },
    regenerateButton: { padding: '10px 16px', background: 'rgba(96, 165, 250, 0.2)', border: '1px solid rgba(96, 165, 250, 0.4)', borderRadius: 8, color: '#60a5fa', fontWeight: 600, cursor: 'pointer', fontSize: 13, width: '100%' },
    infoSection: { background: 'rgba(139, 92, 246, 0.05)', border: '1px solid rgba(139, 92, 246, 0.2)', borderRadius: 12, padding: 16 },
    infoTitle: { margin: '0 0 12px', color: '#c4b5fd', fontSize: 13, fontWeight: 600 },
    infoList: { margin: 0, paddingLeft: 20, color: '#cbd5e1', fontSize: 12, lineHeight: 1.6 },
}
