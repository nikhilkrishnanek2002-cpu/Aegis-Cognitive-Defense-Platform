export default function XAITab() {
    return (
        <div style={{ color: '#94a3b8', display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={styles.section}>
                <h3 style={styles.title}>🧠 Explainable AI — Grad-CAM Heatmaps</h3>
                <p>Grad-CAM visualizations highlight exactly which regions of the Range-Doppler map and Micro-Doppler spectrogram influenced the AI classification decision.</p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 16 }}>
                    {['RD Map Heatmap (Grad-CAM)', 'Spectrogram Heatmap (Grad-CAM)'].map((label) => (
                        <div key={label} style={styles.placeholder}>
                            <div style={styles.icon}>🎨</div>
                            <div style={styles.placeholderText}>{label}</div>
                            <p style={styles.placeholderSub}>Grad-CAM heatmaps require a triggered scan with AI inference. Run a scan from the Analytics tab to generate a classification and view the XAI overlay here.</p>
                        </div>
                    ))}
                </div>
            </div>
            <div style={styles.section}>
                <h3 style={styles.title}>ℹ️ How Grad-CAM Works</h3>
                <ul style={{ paddingLeft: 20, lineHeight: 2, color: '#94a3b8', fontSize: 14 }}>
                    <li>Computes gradients of the target class output w.r.t. convolutional feature maps</li>
                    <li>Weights feature maps by their contribution to the classification</li>
                    <li>Produces a coarse localization heatmap — <strong style={{ color: '#60a5fa' }}>red</strong> = high influence, <strong style={{ color: '#3b82f6' }}>blue</strong> = low influence</li>
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
}
