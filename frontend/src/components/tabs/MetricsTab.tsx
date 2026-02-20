import { useEffect, useState } from 'react'
import { getMetricsReport } from '../../api/client'

interface MetricsReport {
    accuracy?: number
    metadata?: { model_name?: string; timestamp?: string; n_samples?: number; n_classes?: number }
    macro_avg?: { precision?: number; recall?: number; f1?: number }
    weighted_avg?: { precision?: number; recall?: number; f1?: number }
    classification_report?: Record<string, Record<string, number>>
}

export default function MetricsTab() {
    const [data, setData] = useState<MetricsReport | null>(null)
    const [error, setError] = useState('')

    useEffect(() => {
        getMetricsReport()
            .then((r) => setData(r.data))
            .catch(() => setError('Metrics report not available yet. Train the model to generate metrics.'))
    }, [])

    if (error) return <div style={styles.warning}>⚠️ {error}</div>
    if (!data) return <p style={{ color: '#94a3b8' }}>Loading metrics...</p>

    const fmt = (v?: number) => (v !== undefined ? v.toFixed(3) : '—')

    const rows = [
        { label: 'Model', value: data.metadata?.model_name ?? '—' },
        { label: 'Timestamp', value: data.metadata?.timestamp ?? '—' },
        { label: 'Samples', value: data.metadata?.n_samples ?? '—' },
        { label: 'Classes', value: data.metadata?.n_classes ?? '—' },
        { label: 'Accuracy', value: fmt(data.accuracy) },
        { label: 'Macro Precision', value: fmt(data.macro_avg?.precision) },
        { label: 'Macro Recall', value: fmt(data.macro_avg?.recall) },
        { label: 'Macro F1', value: fmt(data.macro_avg?.f1) },
        { label: 'Weighted Precision', value: fmt(data.weighted_avg?.precision) },
        { label: 'Weighted Recall', value: fmt(data.weighted_avg?.recall) },
        { label: 'Weighted F1', value: fmt(data.weighted_avg?.f1) },
    ]

    const perClass = data.classification_report
        ? Object.entries(data.classification_report).filter(([, v]) => typeof v === 'object')
        : []

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div style={styles.section}>
                <h3 style={styles.title}>📊 Experiment Summary</h3>
                <table style={styles.table}>
                    <tbody>
                        {rows.map((r) => (
                            <tr key={r.label}>
                                <td style={{ ...styles.td, color: '#64748b', width: 200 }}>{r.label}</td>
                                <td style={{ ...styles.td, color: '#60a5fa', fontWeight: 600 }}>{String(r.value)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {perClass.length > 0 && (
                <div style={styles.section}>
                    <h3 style={styles.title}>📋 Per-Class Classification Report</h3>
                    <table style={styles.table}>
                        <thead>
                            <tr>{['Class', 'Precision', 'Recall', 'F1', 'Support'].map(h => <th key={h} style={styles.th}>{h}</th>)}</tr>
                        </thead>
                        <tbody>
                            {perClass.map(([cls, vals]) => (
                                <tr key={cls}>
                                    <td style={{ ...styles.td, color: '#a78bfa', fontWeight: 600 }}>{cls}</td>
                                    <td style={styles.td}>{fmt(vals.precision)}</td>
                                    <td style={styles.td}>{fmt(vals.recall)}</td>
                                    <td style={styles.td}>{fmt(vals['f1-score'])}</td>
                                    <td style={styles.td}>{vals.support ?? '—'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            <div style={styles.section}>
                <h3 style={styles.title}>🖼️ Performance Charts</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                    {['confusion_matrix', 'roc_curve', 'precision_recall', 'training_curves'].map((name) => (
                        <div key={name} style={{ textAlign: 'center' }}>
                            <p style={{ color: '#64748b', fontSize: 12, textTransform: 'uppercase', marginBottom: 8 }}>{name.replace(/_/g, ' ')}</p>
                            <img
                                src={`/api/metrics/images/${name}`}
                                alt={name}
                                style={{ width: '100%', borderRadius: 8, border: '1px solid rgba(255,255,255,0.08)' }}
                                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                            />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

const styles: Record<string, React.CSSProperties> = {
    section: { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: 20 },
    title: { margin: '0 0 14px', color: '#e2e8f0', fontSize: 15, fontWeight: 600 },
    table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
    th: { textAlign: 'left', padding: '8px 10px', color: '#64748b', borderBottom: '1px solid rgba(255,255,255,0.08)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5 },
    td: { padding: '8px 10px', color: '#e2e8f0', borderBottom: '1px solid rgba(255,255,255,0.05)' },
    warning: { padding: 16, background: 'rgba(251,191,36,0.1)', border: '1px solid rgba(251,191,36,0.3)', borderRadius: 10, color: '#fcd34d', fontSize: 14 },
}
