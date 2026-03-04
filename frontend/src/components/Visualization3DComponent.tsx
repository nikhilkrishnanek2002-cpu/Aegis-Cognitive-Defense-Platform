import { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import { useRadarStore } from '../store/radarStore'

interface Visualization3D {
    rd_map?: number[][]
    spec?: number[][]
    meta?: number[]
}

export default function Visualization3DComponent() {
    const { frame } = useRadarStore()
    const [data3D, setData3D] = useState<Visualization3D | null>(null)

    useEffect(() => {
        if (frame && frame.rd_map && frame.spec) {
            setData3D({
                rd_map: frame.rd_map,
                spec: frame.spec,
                meta: frame.meta,
            })
        }
    }, [frame])

    if (!data3D || !data3D.rd_map || !data3D.spec) {
        return <p style={{ color: '#0ff', fontFamily: 'monospace' }}>[ SYSTEM ] ⏳ Waiting for radar telemetry...</p>
    }

    // Range-Doppler Map Processing
    const rd_map = data3D.rd_map!
    const rd_z_vals = rd_map.map((row) => row.map((v) => Math.abs(v)))

    // Spectrogram Processing
    const spec = data3D.spec!
    const spec_vals = spec.map((row) => row.map((v) => Math.abs(v)))

    // Real-world Radar Colorscales
    // Jet colorscale — industry standard for Range-Doppler maps
    const radarJetColorscale = [
        [0.0, '#000080'], [0.1, '#0000ff'], [0.2, '#0066ff'],
        [0.3, '#00ccff'], [0.4, '#00ffcc'], [0.5, '#66ff66'],
        [0.6, '#ccff00'], [0.7, '#ffcc00'], [0.8, '#ff6600'],
        [0.9, '#ff0000'], [1.0, '#800000'],
    ]

    // Viridis colorscale — perceptually uniform, standard for spectrograms
    const spectrogramViridisColorscale = [
        [0.0, '#440154'], [0.1, '#482878'], [0.2, '#3e4989'],
        [0.3, '#31688e'], [0.4, '#26828e'], [0.5, '#1f9e89'],
        [0.6, '#35b779'], [0.7, '#6ece58'], [0.8, '#b5de2b'],
        [0.9, '#e5e419'], [1.0, '#fde725'],
    ]

    // Convert linear values to dB for Range-Doppler display
    const toDBScale = (data: number[][]) => {
        return data.map(row => row.map(v => {
            const clamped = Math.max(v, 1e-6)
            return 10 * Math.log10(clamped)
        }))
    }

    const rd_db = toDBScale(rd_z_vals)
    const spec_db = toDBScale(spec_vals)

    const LayoutBase = {
        plot_bgcolor: '#0a101d', // Slightly transparent blue/gray to match border
        paper_bgcolor: 'transparent',
        font: { color: '#2dd4bf', family: "'Orbitron', 'Courier New', monospace" },
        margin: { t: 40, r: 60, b: 60, l: 80 },
        xaxis: {
            gridcolor: 'rgba(45, 212, 191, 0.2)', // Cyan grid
            zerolinecolor: 'rgba(45, 212, 191, 0.5)',
            tickfont: { color: '#8da9c4' },
            showline: true,
            linecolor: '#2dd4bf',
            linewidth: 1,
            mirror: true
        },
        yaxis: {
            gridcolor: 'rgba(45, 212, 191, 0.2)',
            zerolinecolor: 'rgba(45, 212, 191, 0.5)',
            tickfont: { color: '#8da9c4' },
            showline: true,
            linecolor: '#2dd4bf',
            linewidth: 1,
            mirror: true
        }
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 30, padding: 10 }}>
            {/* RANGE-DOPPLER MAP */}
            <div style={styles.sciFiContainer}>
                <div style={styles.sciFiHeader}>
                    <h4 style={styles.sciFiTitle}>RANGE-DOPPLER MAP</h4>
                    <div style={styles.cornerDecoration}></div>
                </div>
                <Plot
                    data={[{
                        z: rd_db,
                        x0: 0, dx: 500 / (rd_db[0]?.length || 1),
                        y0: -50, dy: 100 / (rd_db.length || 1),
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

            {/* SPECTROGRAM */}
            <div style={styles.sciFiContainer}>
                <div style={styles.sciFiHeader}>
                    <h4 style={styles.sciFiTitle}>SPECTROGRAM</h4>
                    <div style={styles.cornerDecoration}></div>
                </div>
                <Plot
                    data={[{
                        z: spec_db,
                        x0: 0, dx: 10 / (spec_db[0]?.length || 1),
                        y0: 0, dy: 5000 / (spec_db.length || 1),
                        type: 'heatmap',
                        colorscale: spectrogramViridisColorscale as any,
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
                        xaxis: { ...LayoutBase.xaxis, title: { text: "Time (s)", font: { color: '#94a3b8' } } },
                        yaxis: { ...LayoutBase.yaxis, title: { text: "Frequency (Hz)", font: { color: '#94a3b8' } } },
                    } as any}
                    style={{ width: '100%', height: 380 }}
                    config={{ responsive: true, displayModeBar: false }}
                />
            </div>
        </div>
    )
}

const styles: Record<string, React.CSSProperties> = {
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
