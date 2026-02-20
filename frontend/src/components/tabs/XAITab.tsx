import { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import { useRadarStore } from '../../store/radarStore'
import { API_BASE } from '../../api/client'

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

    useEffect(() => {
        if (frame && frame.xai) {
            setGradcamData(frame.xai as GradCAMData)
        }
    }, [frame])

    const handleGenerateGradCAM = async () => {
        if (!frame) {
            alert('No radar frame available')
            return
        }
        setLoading(true)
        try {
            const response = await fetch(`${API_BASE}/radar/scan`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`,
                },
                body: JSON.stringify({
                    target: frame.detected,
                    distance: 200,
                    gain_db: 15,
                }),
            })
            if (response.ok) {
                const data = await response.json()
                if (data.xai) {
                    setGradcamData(data.xai)
                }
            }
        } catch (err) {
            console.error('Error generating Grad-CAM:', err)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div style={{ color: '#94a3b8', display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={styles.section}>
                <h3 style={styles.title}>🧠 Explainable AI — Grad-CAM Heatmaps</h3>
                <p>Grad-CAM visualizations highlight exactly which regions of the Range-Doppler map and Micro-Doppler spectrogram influenced the AI classification decision.</p>

                {!gradcamData ? (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 16 }}>
                        {['RD Map Heatmap (Grad-CAM)', 'Spectrogram Heatmap (Grad-CAM)'].map((label) => (
                            <div key={label} style={styles.placeholder}>
                                <div style={styles.icon}>🎨</div>
                                <div style={styles.placeholderText}>{label}</div>
                                <p style={styles.placeholderSub}>Grad-CAM heatmaps require a triggered scan with AI inference. Run a scan from the Analytics tab or click "Generate Grad-CAM" to create visualization.</p>
                                <button
                                    onClick={handleGenerateGradCAM}
                                    disabled={loading}
                                    style={{
                                        ...styles.button,
                                        opacity: loading ? 0.6 : 1,
                                        cursor: loading ? 'not-allowed' : 'pointer',
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
                            <span style={{ color: '#60a5fa' }}>Target: {gradcamData.target_class}</span>
                            <span style={{ color: '#22c55e' }}>Confidence: {(gradcamData.confidence * 100).toFixed(1)}%</span>
                            <span style={{ color: '#a78bfa' }}>Scan ID: {gradcamData.scan_id}</span>
                        </div>

                        <div style={{ marginTop: 16 }}>
                            <h4 style={styles.chartTitle}>Grad-CAM Heatmap Visualization</h4>
                            <Plot
                                data={[{
                                    z: gradcamData.heatmap,
                                    type: 'heatmap',
                                    colorscale: 'Hot',
                                    showscale: true,
                                    colorbar: { title: 'Influence' },
                                }]}
                                layout={{
                                    title: `Grad-CAM: ${gradcamData.target_class}`,
                                    xaxis: { title: 'X Pixels' },
                                    yaxis: { title: 'Y Pixels' },
                                    plot_bgcolor: 'rgba(0,0,0,0)',
                                    paper_bgcolor: 'rgba(0,0,0,0)',
                                    font: { color: '#94a3b8' },
                                }}
                                style={{ width: '100%', height: 500 }}
                            />
                        </div>

                        {gradcamData.image_path && (
                            <div style={{ marginTop: 20 }}>
                                <h4 style={styles.chartTitle}>Grad-CAM Overlay (PNG)</h4>
                                <img
                                    src={`${API_BASE}${gradcamData.image_path}`}
                                    alt="Grad-CAM"
                                    style={{
                                        width: '100%',
                                        maxWidth: 600,
                                        borderRadius: 8,
                                        border: '1px solid rgba(255,255,255,0.08)',
                                    }}
                                />
                            </div>
                        )}

                        <button
                            onClick={handleGenerateGradCAM}
                            disabled={loading}
                            style={{
                                ...styles.button,
                                marginTop: 16,
                                opacity: loading ? 0.6 : 1,
                                cursor: loading ? 'not-allowed' : 'pointer',
                            }}
                        >
                            {loading ? '⏳ Generating...' : '🔄 Generate New Grad-CAM'}
                        </button>
                    </div>
                )}
            </div>

            <div style={styles.section}>
                <h3 style={styles.title}>ℹ️ How Grad-CAM Works</h3>
                <ul style={{ paddingLeft: 20, lineHeight: 2, color: '#94a3b8', fontSize: 14 }}>
                    <li>Computes gradients of the target class output w.r.t. convolutional feature maps</li>
                    <li>Weights feature maps by their contribution to the classification</li>
                    <li>Produces a coarse localization heatmap — <strong style={{ color: '#ef4444' }}>red</strong> = high influence, <strong style={{ color: '#3b82f6' }}>blue</strong> = low influence</li>
                    <li>Confirms the model is reasoning on physical target features (e.g., rotor micro-Doppler)</li>
                </ul>
            </div>
        </div>
    )
}

const styles: Record<string, React.CSSProperties> = {
    section: { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: 20 },
    title: { margin: '0 0 12px', color: '#e2e8f0', fontSize: 15, fontWeight: 600 },
    placeholder: { background: 'rgba(255,255,255,0.03)', border: '1px dashed rgba(255,255,255,0.15)', borderRadius: 10, padding: 24, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 },
    icon: { fontSize: 36 },
    placeholderText: { color: '#60a5fa', fontWeight: 600, fontSize: 14 },
    placeholderSub: { color: '#64748b', fontSize: 12, maxWidth: 240 },
    button: {
        background: 'linear-gradient(135deg, #60a5fa, #3b82f6)',
        color: 'white',
        border: 'none',
        borderRadius: 6,
        padding: '8px 16px',
        fontSize: 13,
        fontWeight: 600,
        cursor: 'pointer',
        marginTop: 12,
    },
    dataHeader: {
        display: 'flex',
        gap: 20,
        padding: 12,
        background: 'rgba(96, 165, 250, 0.1)',
        borderRadius: 8,
        fontSize: 13,
        fontWeight: 600,
    },
    chartTitle: {
        color: '#e2e8f0',
        fontSize: 14,
        fontWeight: 600,
        margin: '0 0 12px',
    },
}
