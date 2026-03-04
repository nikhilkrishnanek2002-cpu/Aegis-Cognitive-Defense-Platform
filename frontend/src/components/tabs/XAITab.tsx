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

            // Validate XAI data structure
            if (data && data.xai && typeof data.xai === 'object') {
                const xai = data.xai

                // Ensure required fields exist
                if (!xai.heatmap || !Array.isArray(xai.heatmap) || xai.heatmap.length === 0) {
                    console.warn('XAI heatmap is invalid, received:', xai)
                    throw new Error('Invalid Grad-CAM heatmap data')
                }

                if (!xai.heatmap_shape || !Array.isArray(xai.heatmap_shape)) {
                    throw new Error('Invalid heatmap shape')
                }

                setGradcamData(xai as GradCAMData)
                setError('')
            } else {
                const keys = data ? Object.keys(data).join(', ') : 'empty response'
                console.warn('Response data:', data)
                throw new Error(`No valid XAI data! Received keys: [${keys}]`)
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Unknown error'
            setError(`Failed to generate Grad-CAM: ${message}`)
            console.error('Error generating Grad-CAM:', err)
        } finally {
            setLoading(false)
        }
    }

    const rd_map = frame?.rd_map
    const rd_z_vals = rd_map ? rd_map.map((row) => row.map((v) => Math.abs(v))) : null

    // Real-world Radar Colorscales
    // Jet colorscale — standard for Range-Doppler maps in radar systems
    const radarJetColorscale = [
        [0.0, '#000080'], [0.1, '#0000ff'], [0.2, '#0066ff'],
        [0.3, '#00ccff'], [0.4, '#00ffcc'], [0.5, '#66ff66'],
        [0.6, '#ccff00'], [0.7, '#ffcc00'], [0.8, '#ff6600'],
        [0.9, '#ff0000'], [1.0, '#800000'],
    ]

    // Inferno colorscale — standard for Grad-CAM / activation heatmaps
    const gradcamInfernoColorscale = [
        [0.0, '#000004'], [0.1, '#160b39'], [0.2, '#420a68'],
        [0.3, '#6a176e'], [0.4, '#932667'], [0.5, '#bc3754'],
        [0.6, '#dd513a'], [0.7, '#f37819'], [0.8, '#fca50a'],
        [0.9, '#f6d746'], [1.0, '#fcffa4'],
    ]

    // Convert heatmap values to dB scale for display
    const toDBScale = (data: number[][]) => {
        return data.map(row => row.map(v => {
            const clamped = Math.max(v, 1e-6)
            return 10 * Math.log10(clamped)
        }))
    }

    const LayoutBase = {
        plot_bgcolor: '#0a101d',
        paper_bgcolor: 'transparent',
        font: { color: '#2dd4bf', family: "'Orbitron', 'Courier New', monospace" },
        margin: { t: 40, r: 60, b: 60, l: 80 },
        xaxis: {
            gridcolor: 'rgba(45, 212, 191, 0.2)',
            zerolinecolor: 'rgba(45, 212, 191, 0.5)',
            tickfont: { color: '#8da9c4' },
            showline: true, linecolor: '#2dd4bf', linewidth: 1, mirror: true
        },
        yaxis: {
            gridcolor: 'rgba(45, 212, 191, 0.2)',
            zerolinecolor: 'rgba(45, 212, 191, 0.5)',
            tickfont: { color: '#8da9c4' },
            showline: true, linecolor: '#2dd4bf', linewidth: 1, mirror: true
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

                {error && <div style={styles.error}>⚠️ {error}</div>}

                {!gradcamData ? (
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

                        <div style={{ marginTop: 24, display: 'flex', flexDirection: 'column', gap: 24 }}>
                            {/* RANGE DOPPLER MAP */}
                            {rd_z_vals && (
                                <div style={styles.sciFiContainer}>
                                    <div style={styles.sciFiHeader}>
                                        <h4 style={styles.sciFiTitle}>RANGE-DOPPLER MAP</h4>
                                        <div style={styles.cornerDecoration}></div>
                                    </div>
                                    <Plot
                                        data={[{
                                            z: toDBScale(rd_z_vals),
                                            x0: 0, dx: 500 / (rd_z_vals[0]?.length || 1),
                                            y0: -50, dy: 100 / (rd_z_vals.length || 1),
                                            type: 'heatmap',
                                            colorscale: radarJetColorscale as any,
                                            showscale: true,
                                            zsmooth: 'best',
                                            colorbar: {
                                                thickness: 15,
                                                tickfont: { color: '#cbd5e1', size: 10 },
                                                outlinecolor: '#475569',
                                                bgcolor: 'rgba(0,0,0,0.6)',
                                                title: { text: 'Power (dB)', side: 'right', font: { color: '#94a3b8', size: 11 } },
                                                ticksuffix: ' dB'
                                            }
                                        }]}
                                        layout={{
                                            ...LayoutBase,
                                            xaxis: { ...LayoutBase.xaxis, title: { text: "Range (m)", font: { color: '#94a3b8' } } },
                                            yaxis: { ...LayoutBase.yaxis, title: { text: "Doppler Velocity (m/s)", font: { color: '#94a3b8' } } },
                                        } as any}
                                        style={{ width: '100%', height: 380 }}
                                        config={{ responsive: true, displayModeBar: false }}
                                    />
                                </div>
                            )}

                            {/* GRAD-CAM SPECTROGRAM (ACTUAL GRAD-CAM HEATMAP WITH PROPER STYLE) */}
                            {gradcamData.heatmap && (
                                <div style={styles.sciFiContainer}>
                                    <div style={styles.sciFiHeader}>
                                        <h4 style={styles.sciFiTitle}>GRAD-CAM INFLUENCE</h4>
                                        <div style={styles.cornerDecoration}></div>
                                    </div>
                                    <Plot
                                        data={[{
                                            z: gradcamData.heatmap,
                                            type: 'heatmap' as const,
                                            colorscale: gradcamInfernoColorscale as any,
                                            showscale: true,
                                            zsmooth: 'best',
                                            zmin: 0,
                                            zmax: 1,
                                            colorbar: {
                                                thickness: 15,
                                                tickfont: { color: '#cbd5e1', size: 10 },
                                                outlinecolor: '#475569',
                                                bgcolor: 'rgba(0,0,0,0.6)',
                                                title: { text: 'Activation', side: 'right', font: { color: '#94a3b8', size: 11 } },
                                            }
                                        }]}
                                        layout={{
                                            ...LayoutBase,
                                            xaxis: { ...LayoutBase.xaxis, title: { text: "Range Bin", font: { color: '#94a3b8' } } },
                                            yaxis: { ...LayoutBase.yaxis, title: { text: "Doppler Bin", font: { color: '#94a3b8' } } },
                                        } as any}
                                        style={{ width: '100%', height: 380 }}
                                        config={{ responsive: true, displayModeBar: false }}
                                    />
                                </div>
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
                )}
            </div>

            <div style={styles.infoSection}>
                <h4 style={styles.infoTitle}>📖 How Grad-CAM Works</h4>
                <ul style={styles.infoList}>
                    <li>Red and yellow regions show areas that strongly indicate the predicted target class</li>
                    <li>Dark/blue regions show areas that oppose the prediction or have no influence</li>
                    <li>Compare the raw Range-Doppler map above to the Grad-CAM influence map to see what the AI "saw"</li>
                </ul>
            </div>
        </div>
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

    // SCI-FI STYLES ADDED
    sciFiContainer: {
        background: 'linear-gradient(180deg, rgba(8, 20, 35, 0.9) 0%, rgba(5, 12, 22, 0.9) 100%)',
        border: '2px solid #38bdf8',
        borderRadius: '8px',
        boxShadow: '0 0 15px rgba(56, 189, 248, 0.2), inset 0 0 20px rgba(56, 189, 248, 0.1)',
        padding: '2px', // Minimal padding to allow plot to fill
        position: 'relative',
        overflow: 'hidden',
    },
    sciFiHeader: {
        display: 'flex',
        justifyContent: 'center',
        padding: '12px 20px',
        borderBottom: '1px solid rgba(56, 189, 248, 0.3)',
        position: 'relative',
        background: 'rgba(14, 165, 233, 0.1)',
    },
    sciFiTitle: {
        margin: 0,
        color: '#bae6fd',
        fontSize: '22px',
        fontWeight: 400,
        letterSpacing: '4px',
        fontFamily: "'Orbitron', 'Courier New', monospace",
        textShadow: '0 0 10px rgba(186, 230, 253, 0.5)',
    },
    cornerDecoration: {
        position: 'absolute',
        top: 0,
        left: 0,
        width: '20px',
        height: '20px',
        borderTop: '2px solid #e0f2fe',
        borderLeft: '2px solid #e0f2fe',
        opacity: 0.8,
    }
}
