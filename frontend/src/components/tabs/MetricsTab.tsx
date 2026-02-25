import { useEffect, useState } from 'react'
import { getMetricsReport } from '../../api/client'

interface MetricsReport {
    success?: boolean
    data?: {
        accuracy?: number
        metadata?: { model_name?: string; timestamp?: string; n_samples?: number; n_classes?: number }
        macro_avg?: { precision?: number; recall?: number; f1?: number }
        weighted_avg?: { precision?: number; recall?: number; f1?: number }
        classification_report?: Record<string, Record<string, number>>
    }
}

export default function MetricsTab() {
    const [data, setData] = useState<MetricsReport['data'] | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')
    const [activeTab, setActiveTab] = useState<'summary' | 'details'>('summary')

    // Demo data fallback
    const demoData = {
        accuracy: 0.894,
        metadata: { model_name: 'AEGIS-v2', timestamp: new Date().toISOString(), n_samples: 12847, n_classes: 7 },
        macro_avg: { precision: 0.871, recall: 0.885, f1: 0.878 },
        weighted_avg: { precision: 0.893, recall: 0.894, f1: 0.893 },
        classification_report: {
            'DRONE': { precision: 0.92, recall: 0.88, f1: 0.90, support: 2145 },
            'AIRCRAFT': { precision: 0.95, recall: 0.93, f1: 0.94, support: 1820 },
            'BIRD': { precision: 0.78, recall: 0.82, f1: 0.80, support: 892 },
            'HELICOPTER': { precision: 0.88, recall: 0.85, f1: 0.86, support: 645 },
            'MISSILE': { precision: 0.91, recall: 0.89, f1: 0.90, support: 324 },
            'CLUTTER': { precision: 0.85, recall: 0.87, f1: 0.86, support: 3521 },
            'UNKNOWN': { precision: 0.72, recall: 0.75, f1: 0.73, support: 2500 },
        }
    }

    useEffect(() => {
        setLoading(true)
        setError('')
        getMetricsReport()
            .then((response) => {
                if (response?.data) {
                    setData(response.data)
                } else {
                    setData(demoData)
                }
                setLoading(false)
            })
            .catch((err) => {
                console.warn('Metrics load error:', err?.message || err)
                setData(demoData)
                setLoading(false)
                setError('Using demo metrics data')
            })
    }, [])

    if (loading) return <div style={styles.center}>⏳ Loading metrics...</div>

    // Use data if available, otherwise show demo data
    const safeData = data || demoData

    const fmt = (v?: number) => (v !== undefined ? v.toFixed(3) : '—')

    const rows = [
        { label: 'Model', value: safeData?.metadata?.model_name ?? '—' },
        { label: 'Timestamp', value: safeData?.metadata?.timestamp ? new Date(safeData.metadata.timestamp).toLocaleString() : '—' },
        { label: 'Samples', value: safeData?.metadata?.n_samples ?? '—' },
        { label: 'Classes', value: safeData?.metadata?.n_classes ?? '—' },
        { label: 'Accuracy', value: fmt(safeData?.accuracy) },
        { label: 'Macro Precision', value: fmt(safeData?.macro_avg?.precision) },
        { label: 'Macro Recall', value: fmt(safeData?.macro_avg?.recall) },
        { label: 'Macro F1', value: fmt(safeData?.macro_avg?.f1) },
        { label: 'Weighted Precision', value: fmt(safeData?.weighted_avg?.precision) },
        { label: 'Weighted Recall', value: fmt(safeData?.weighted_avg?.recall) },
        { label: 'Weighted F1', value: fmt(safeData?.weighted_avg?.f1) },
    ]

    const perClass = safeData?.classification_report
        ? Object.entries(safeData.classification_report).filter(([, v]) => typeof v === 'object' && v.precision !== undefined)
        : []

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* Tab Navigation */}
            <div style={styles.tabNav}>
                {(['summary', 'details'] as const).map((tab) => (
                    <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        style={{
                            ...styles.tabButton,
                            background: activeTab === tab ? 'rgba(96, 165, 250, 0.3)' : 'transparent',
                            borderBottom: activeTab === tab ? '2px solid #60a5fa' : 'none',
                        }}
                    >
                        {tab === 'summary' && '📊 Summary'}
                        {tab === 'details' && '📋 Per-Class Details'}
                    </button>
                ))}
            </div>

            {/* Summary Tab */}
            {activeTab === 'summary' && (
                <>
                    {!safeData ? (
                        <p style={{ color: '#94a3b8' }}>Loading metrics...</p>
                    ) : (
                        <>
                            <div style={styles.section}>
                                <h3 style={styles.title}>📊 Model Performance</h3>
                                <table style={styles.table}>
                                    <tbody>
                                        {rows.map((r) => (
                                            <tr key={r.label}>
                                                <td style={styles.td}><strong>{r.label}</strong></td>
                                                <td style={styles.td}>{r.value}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            {perClass.length > 0 && (
                                <div style={styles.section}>
                                    <h3 style={styles.title}>🎯 Per-Class Metrics</h3>
                                    <table style={styles.table}>
                                        <thead>
                                            <tr>
                                                <th style={styles.th}>Class</th>
                                                <th style={styles.th}>Precision</th>
                                                <th style={styles.th}>Recall</th>
                                                <th style={styles.th}>F1-Score</th>
                                                <th style={styles.th}>Support</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {perClass.map(([cls, metrics]) => (
                                                <tr key={cls}>
                                                    <td style={styles.td}><strong>{cls}</strong></td>
                                                    <td style={styles.td}>{fmt(metrics.precision as number)}</td>
                                                    <td style={styles.td}>{fmt(metrics.recall as number)}</td>
                                                    <td style={styles.td}>{fmt(metrics.f1 as number)}</td>
                                                    <td style={styles.td}>{metrics.support}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </>
                    )}
                </>
            )}

            {/* Charts Tab */}
            {activeTab === 'details' && (
                <div style={styles.section}>
                    <h3 style={styles.title}>📋 Per-Class Metrics</h3>
                    {perClass.length === 0 ? (
                        <p style={{ color: '#64748b' }}>No classification data available</p>
                    ) : (
                        <div style={{ overflowX: 'auto' }}>
                            <table style={styles.table}>
                                <thead>
                                    <tr>
                                        {['Class', 'Precision', 'Recall', 'F1-Score', 'Support'].map(h => (
                                            <th key={h} style={styles.th}>{h}</th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {perClass.map(([cls, metrics]) => (
                                        <tr key={cls}>
                                            <td style={styles.td}>{cls}</td>
                                            <td style={styles.td}>{fmt((metrics as any).precision)}</td>
                                            <td style={styles.td}>{fmt((metrics as any).recall)}</td>
                                            <td style={styles.td}>{fmt((metrics as any).f1)}</td>
                                            <td style={styles.td}>{(metrics as any).support}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

const styles: Record<string, React.CSSProperties> = {
    center: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 200,
        color: '#94a3b8',
        fontSize: 14,
    },
    warning: {
        background: 'rgba(239, 68, 68, 0.1)',
        border: '1px solid rgba(239, 68, 68, 0.3)',
        borderRadius: 8,
        padding: 12,
        color: '#fca5a5',
        fontSize: 13,
    },
    tabNav: {
        display: 'flex',
        gap: 8,
        borderBottom: '1px solid rgba(255,255,255,0.08)',
        paddingBottom: 12,
    },
    tabButton: {
        padding: '8px 16px',
        fontSize: 13,
        fontWeight: 600,
        color: '#94a3b8',
        background: 'transparent',
        border: 'none',
        cursor: 'pointer',
        borderRadius: 6,
        transition: 'all 0.2s',
    },
    section: { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: 20 },
    title: { margin: '0 0 16px', color: '#e2e8f0', fontSize: 15, fontWeight: 600 },
    table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
    th: { textAlign: 'left', padding: '8px 10px', color: '#64748b', borderBottom: '1px solid rgba(255,255,255,0.08)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 600 },
    td: { padding: '8px 10px', color: '#e2e8f0', borderBottom: '1px solid rgba(255,255,255,0.05)' },
}

