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

    // Custom Sci-Fi Colorscales
    const neonCyanColorscale = [
        [0.0, '#0f2038'], // Deep navy/black background
        [0.2, '#184c78'], // Dark blue
        [0.4, '#1b80a6'], // Bright blue
        [0.6, '#26c4cc'], // Cyan
        [0.8, '#69eec2'], // Light cyan/green
        [1.0, '#e2ffda'], // White/cyan core
    ]

    const neonPurpleColorscale = [
        [0.0, '#1a0b2e'], // Deep purple/black background
        [0.2, '#3b1257'], // Dark purple
        [0.4, '#6a1a7a'], // Magenta
        [0.6, '#a4369e'], // Bright pink/purple
        [0.8, '#d96a84'], // Orange-pink
        [1.0, '#ffe89b'], // Yellow/White core
    ]

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
                        z: rd_z_vals,
                        type: 'heatmap',
                        colorscale: neonCyanColorscale as any,
                        showscale: true,
                        colorbar: {
                            thickness: 15,
                            tickfont: { color: '#2dd4bf' },
                            outlinecolor: '#2dd4bf',
                            bgcolor: 'rgba(0,0,0,0.5)',
                        }
                    }]}
                    layout={{
                        ...LayoutBase,
                        xaxis: { ...LayoutBase.xaxis, title: { text: "RANGE (m)", font: { color: '#4fd1c5' } } },
                        yaxis: { ...LayoutBase.yaxis, title: { text: "DOPPLER (m/s)", font: { color: '#4fd1c5' } } },
                    } as any}
                    style={{ width: '100%', height: 350 }}
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
                        z: spec_vals,
                        type: 'heatmap',
                        colorscale: neonPurpleColorscale as any,
                        showscale: true,
                        colorbar: {
                            thickness: 15,
                            tickfont: { color: '#fca5a5' },
                            outlinecolor: '#fca5a5',
                            bgcolor: 'rgba(0,0,0,0.5)',
                        }
                    }]}
                    layout={{
                        ...LayoutBase,
                        xaxis: { ...LayoutBase.xaxis, title: { text: "TIME (s)", font: { color: '#9ca3af' } } },
                        yaxis: { ...LayoutBase.yaxis, title: { text: "FREQUENCY (Hz)", font: { color: '#9ca3af' } } },
                    } as any}
                    style={{ width: '100%', height: 350 }}
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
