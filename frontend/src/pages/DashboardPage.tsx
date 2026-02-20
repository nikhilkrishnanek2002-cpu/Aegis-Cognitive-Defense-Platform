import { useState } from 'react'
import { useAuthStore } from '../store/authStore'
import { useRadarStore } from '../store/radarStore'
import { useRadarWebSocket } from '../api/websocket'
import type { RadarFrame } from '../store/radarStore'

// Tab imports
import AnalyticsTab from '../components/tabs/AnalyticsTab'
import XAITab from '../components/tabs/XAITab'
import PhotonicTab from '../components/tabs/PhotonicTab'
import MetricsTab from '../components/tabs/MetricsTab'
import LogsTab from '../components/tabs/LogsTab'
import AdminTab from '../components/tabs/AdminTab'

const TABS = ['📡 Real-Time Analytics', '🧠 Explainable AI', '💡 Photonic Params', '📊 Research Metrics', '📋 System Logs', '⚙️ Admin Panel']

export default function DashboardPage() {
    const [activeTab, setActiveTab] = useState(0)
    const { username, role, logout } = useAuthStore()
    const { setFrame, frame } = useRadarStore()

    // Connect WebSocket and funnel frames into Zustand store
    useRadarWebSocket((data) => {
        if (data.detected !== undefined) setFrame(data as unknown as RadarFrame)
    })

    const tabContent = [
        <AnalyticsTab />,
        <XAITab />,
        <PhotonicTab />,
        <MetricsTab />,
        <LogsTab />,
        <AdminTab />,
    ]

    return (
        <div style={styles.root}>
            {/* ─── Sidebar ─── */}
            <aside style={styles.sidebar}>
                <div style={styles.sidebarLogo}>📡</div>
                <h1 style={styles.sidebarTitle}>AEGIS</h1>
                <hr style={styles.divider} />
                <div style={styles.userInfo}>
                    <span style={styles.userLabel}>👤 {username}</span>
                    <span style={styles.roleLabel}>{(role ?? '').toUpperCase()}</span>
                </div>

                {frame && (
                    <div style={{ ...styles.statusBadge, background: frame.is_alert ? 'rgba(239,68,68,0.2)' : 'rgba(34,197,94,0.15)', borderColor: frame.is_alert ? '#ef4444' : '#22c55e' }}>
                        <span style={{ color: frame.is_alert ? '#fca5a5' : '#86efac', fontSize: 11, fontWeight: 700 }}>
                            {frame.is_alert ? '🚨 ALERT' : '✅ CLEAR'} — {frame.detected}
                        </span>
                    </div>
                )}

                <nav style={styles.nav}>
                    {TABS.map((t, i) => (
                        <button key={i} onClick={() => setActiveTab(i)} style={{ ...styles.navBtn, ...(activeTab === i ? styles.navBtnActive : {}) }}>
                            {t}
                        </button>
                    ))}
                </nav>

                <button onClick={logout} style={styles.logoutBtn}>🚪 Logout</button>
            </aside>

            {/* ─── Main Panel ─── */}
            <main style={styles.main}>
                <header style={styles.header}>
                    <h2 style={styles.tabTitle}>{TABS[activeTab]}</h2>
                    {frame && (
                        <div style={styles.headerMeta}>
                            <span style={styles.badge}>⚡ {frame.num_detections} detection{frame.num_detections !== 1 ? 's' : ''}</span>
                            <span style={styles.badge}>🎯 {Object.keys(frame.active_tracks).length} tracks</span>
                            <span style={{ ...styles.badge, color: frame.ew.active ? '#fca5a5' : '#86efac' }}>
                                EW {frame.ew.active ? '🔴 ACTIVE' : '🟢 CLEAR'}
                            </span>
                        </div>
                    )}
                </header>
                <div style={styles.content}>{tabContent[activeTab]}</div>
            </main>
        </div>
    )
}

const styles: Record<string, React.CSSProperties> = {
    root: { display: 'flex', minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: "'Inter', sans-serif" },
    sidebar: { width: 240, background: 'rgba(15,23,42,0.95)', borderRight: '1px solid rgba(255,255,255,0.08)', display: 'flex', flexDirection: 'column', padding: '24px 16px', gap: 8, flexShrink: 0 },
    sidebarLogo: { fontSize: 40, textAlign: 'center' },
    sidebarTitle: { color: '#60a5fa', textAlign: 'center', margin: '4px 0', fontSize: 20, fontWeight: 800, letterSpacing: 4 },
    divider: { border: 'none', borderTop: '1px solid rgba(255,255,255,0.1)', margin: '8px 0' },
    userInfo: { display: 'flex', flexDirection: 'column', gap: 2, padding: '8px 0' },
    userLabel: { fontSize: 13, color: '#94a3b8' },
    roleLabel: { fontSize: 10, color: '#60a5fa', fontWeight: 700, letterSpacing: 2 },
    statusBadge: { border: '1px solid', borderRadius: 8, padding: '6px 10px', margin: '4px 0', textAlign: 'center' },
    nav: { display: 'flex', flexDirection: 'column', gap: 2, flex: 1, marginTop: 8 },
    navBtn: { background: 'transparent', border: 'none', color: '#94a3b8', padding: '10px 12px', borderRadius: 8, textAlign: 'left', cursor: 'pointer', fontSize: 13, transition: 'all 0.15s' },
    navBtnActive: { background: 'rgba(96,165,250,0.15)', color: '#60a5fa', fontWeight: 600 },
    logoutBtn: { background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5', borderRadius: 8, padding: '10px', cursor: 'pointer', fontSize: 13 },
    main: { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' },
    header: { padding: '20px 24px', borderBottom: '1px solid rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
    tabTitle: { margin: 0, fontSize: 18, fontWeight: 700, color: '#f1f5f9' },
    headerMeta: { display: 'flex', gap: 12 },
    badge: { fontSize: 12, padding: '4px 10px', background: 'rgba(255,255,255,0.06)', borderRadius: 20, color: '#94a3b8' },
    content: { flex: 1, padding: 24, overflowY: 'auto' },
}
