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
        return <p style={{ color: '#94a3b8' }}>⏳ Waiting for radar data...</p>
    }

    // Prepare 3D surface plot for Range-Doppler map
    const rd_map = data3D.rd_map!
    const z_vals = rd_map.map((row) => row.map((v) => Math.abs(v)))

    // Prepare Range-Doppler 3D scatter
    const scatter3d_points: { r: number[], d: number[], p: number[] } = { r: [], d: [], p: [] }
    rd_map.forEach((row, i) => {
        row.forEach((val, j) => {
            if (Math.abs(val) > 0.1) {  // Threshold to reduce clutter
                scatter3d_points.r.push(i)
                scatter3d_points.d.push(j)
                scatter3d_points.p.push(Math.abs(val))
            }
        })
    })

    // Prepare 3D surface for spectrogram
    const spec = data3D.spec!
    const spec_vals = spec.map((row) => row.map((v) => Math.abs(v)))

    return (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 20 }}>
            {/* 3D Surface: Range-Doppler Map */}
            <div style={styles.container}>
                <h4 style={styles.title}>🗺️ Range-Doppler 3D Surface</h4>
                <Plot
                    data={[{
                        z: z_vals,
                        type: 'surface',
                        colorscale: 'Viridis',
                        showscale: true,
                    }]}
                    layout={{
                        title: { text: 'RD Map 3D' },
                        scene: {
                            xaxis: { title: 'Doppler Bins' },
                            yaxis: { title: 'Range Bins' },
                            zaxis: { title: 'Magnitude' },
                            camera: { eye: { x: 1.5, y: 1.5, z: 1.3 } },
                        },
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#94a3b8' },
                    }}
                    style={{ width: '100%', height: 500 }}
                />
            </div>

            {/* 3D Scatter: Detections */}
            <div style={styles.container}>
                <h4 style={styles.title}>🎯 Detection Points (3D Scatter)</h4>
                <Plot
                    data={[{
                        x: scatter3d_points.d,
                        y: scatter3d_points.r,
                        z: scatter3d_points.p,
                        mode: 'markers',
                        marker: {
                            size: 5,
                            color: scatter3d_points.p,
                            colorscale: 'Hot',
                            showscale: true,
                        },
                        type: 'scatter3d',
                    }]}
                    layout={{
                        title: { text: 'Detection Scatter 3D' },
                        scene: {
                            xaxis: { title: 'Doppler' },
                            yaxis: { title: 'Range' },
                            zaxis: { title: 'Power' },
                        },
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#94a3b8' },
                    }}
                    style={{ width: '100%', height: 500 }}
                />
            </div>

            {/* 3D Surface: Spectrogram */}
            <div style={styles.container}>
                <h4 style={styles.title}>📡 Spectrogram 3D Surface</h4>
                <Plot
                    data={[{
                        z: spec_vals,
                        type: 'surface',
                        colorscale: 'Plasma',
                        showscale: true,
                    }]}
                    layout={{
                        title: { text: 'Spectrogram 3D' },
                        scene: {
                            xaxis: { title: 'Frequency' },
                            yaxis: { title: 'Time' },
                            zaxis: { title: 'Magnitude' },
                            camera: { eye: { x: 1.5, y: 1.5, z: 1.3 } },
                        },
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#94a3b8' },
                    }}
                    style={{ width: '100%', height: 500 }}
                />
            </div>

            {/* Heatmap: Range-Doppler 2D */}
            <div style={styles.container}>
                <h4 style={styles.title}>🔴 Range-Doppler Heatmap (Top View)</h4>
                <Plot
                    data={[{
                        z: z_vals,
                        type: 'heatmap',
                        colorscale: 'Viridis',
                        showscale: true,
                    }]}
                    layout={{
                        title: { text: 'RD Map 2D Heatmap' },
                        xaxis: { title: 'Doppler Bins' },
                        yaxis: { title: 'Range Bins' },
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#94a3b8' },
                    }}
                    style={{ width: '100%', height: 400 }}
                />
            </div>
        </div>
    )
}

const styles: Record<string, React.CSSProperties> = {
    container: {
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: 12,
        padding: 12,
    },
    title: {
        margin: '0 0 10px',
        color: '#e2e8f0',
        fontSize: 13,
        fontWeight: 600,
    },
}
