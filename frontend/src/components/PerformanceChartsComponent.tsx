import { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import { API_BASE } from '../api/client'

interface ChartData {
    confusion_matrix?: number[][]
    roc_curve?: { fpr: number[], tpr: number[], auc: number }
    precision_recall?: { precision: number[], recall: number[], f1: number[] }
    training_history?: { loss: number[], val_loss: number[], accuracy: number[], val_accuracy: number[] }
}

export default function PerformanceChartsComponent() {
    const [chartData, setChartData] = useState<ChartData | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')

    useEffect(() => {
        const fetchCharts = async () => {
            try {
                // Try to fetch performance chart data
                const response = await fetch(`${API_BASE}/visualizations/performance-charts`, {
                    headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
                })
                if (response.ok) {
                    const data = await response.json()
                    if (data.status === 'ok') {
                        setChartData(data.metrics)
                    } else {
                        setError('No performance data available yet')
                    }
                }
            } catch (err) {
                setError(String(err))
            } finally {
                setLoading(false)
            }
        }

        fetchCharts()
    }, [])

    if (loading) return <p style={{ color: '#94a3b8' }}>Loading performance charts...</p>
    if (error) return <p style={{ color: '#ef4444' }}>⚠️ {error}</p>
    if (!chartData) return <p style={{ color: '#94a3b8' }}>No chart data available</p>

    return (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 20 }}>
            {/* Confusion Matrix */}
            {chartData.confusion_matrix && (
                <div style={styles.chartContainer}>
                    <h4 style={styles.chartTitle}>Confusion Matrix</h4>
                    <Plot
                        data={[{
                            z: chartData.confusion_matrix,
                            type: 'heatmap',
                            colorscale: 'Blues',
                        }]}
                        layout={{
                            title: { text: 'Confusion Matrix' },
                            xaxis: { title: 'Predicted' },
                            yaxis: { title: 'Actual' },
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            font: { color: '#94a3b8' },
                        }}
                        style={{ width: '100%', height: 400 }}
                    />
                </div>
            )}

            {/* ROC Curve */}
            {chartData.roc_curve && (
                <div style={styles.chartContainer}>
                    <h4 style={styles.chartTitle}>ROC Curve (AUC: {chartData.roc_curve.auc.toFixed(3)})</h4>
                    <Plot
                        data={[
                            {
                                x: chartData.roc_curve.fpr,
                                y: chartData.roc_curve.tpr,
                                type: 'scatter',
                                mode: 'lines',
                                name: 'ROC',
                                line: { color: '#60a5fa', width: 2 },
                            },
                            {
                                x: [0, 1],
                                y: [0, 1],
                                type: 'scatter',
                                mode: 'lines',
                                name: 'Chance',
                                line: { color: '#64748b', width: 1, dash: 'dash' },
                            }
                        ]}
                        layout={{
                            title: { text: 'ROC Curve' },
                            xaxis: { title: 'False Positive Rate' },
                            yaxis: { title: 'True Positive Rate' },
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            font: { color: '#94a3b8' },
                        }}
                        style={{ width: '100%', height: 400 }}
                    />
                </div>
            )}

            {/* Precision-Recall */}
            {chartData.precision_recall && (
                <div style={styles.chartContainer}>
                    <h4 style={styles.chartTitle}>Precision-Recall Curve</h4>
                    <Plot
                        data={[
                            {
                                x: chartData.precision_recall.recall,
                                y: chartData.precision_recall.precision,
                                type: 'scatter',
                                mode: 'lines',
                                name: 'PR Curve',
                                line: { color: '#a78bfa', width: 2 },
                            }
                        ]}
                        layout={{
                            title: { text: 'Precision-Recall Curve' },
                            xaxis: { title: 'Recall' },
                            yaxis: { title: 'Precision' },
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            font: { color: '#94a3b8' },
                        }}
                        style={{ width: '100%', height: 400 }}
                    />
                </div>
            )}

            {/* Training History */}
            {chartData.training_history && (
                <div style={styles.chartContainer}>
                    <h4 style={styles.chartTitle}>Training Progress</h4>
                    <Plot
                        data={[
                            {
                                y: chartData.training_history.loss,
                                name: 'Train Loss',
                                type: 'scatter',
                                mode: 'lines',
                                line: { color: '#ef4444' },
                            },
                            {
                                y: chartData.training_history.val_loss,
                                name: 'Val Loss',
                                type: 'scatter',
                                mode: 'lines',
                                line: { color: '#fbbf24' },
                            }
                        ]}
                        layout={{
                            title: { text: 'Loss Over Epochs' },
                            xaxis: { title: 'Epoch' },
                            yaxis: { title: 'Loss' },
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            font: { color: '#94a3b8' },
                        }}
                        style={{ width: '100%', height: 400 }}
                    />
                </div>
            )}
        </div>
    )
}

const styles: Record<string, React.CSSProperties> = {
    chartContainer: {
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: 12,
        padding: 16,
    },
    chartTitle: {
        margin: '0 0 12px',
        color: '#e2e8f0',
        fontSize: 14,
        fontWeight: 600,
    },
}
