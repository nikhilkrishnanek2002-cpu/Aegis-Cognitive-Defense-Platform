import { useEffect, useState, FormEvent } from 'react'
import { useAuthStore } from '../../store/authStore'
import { getUsers, createUser, deleteUser, updateRole, getHealth } from '../../api/client'

interface User { username: string; role: string }
interface Health { cpu_percent: number | null; memory_percent: number | null; db_connected: boolean; rtlsdr_available: boolean; kafka_available: boolean }

export default function AdminTab() {
    const { role } = useAuthStore()
    const [users, setUsers] = useState<User[]>([])
    const [health, setHealth] = useState<Health | null>(null)
    const [newU, setNewU] = useState(''); const [newP, setNewP] = useState(''); const [newR, setNewR] = useState('viewer')
    const [selectedUser, setSelectedUser] = useState('')
    const [newRoleVal, setNewRoleVal] = useState('viewer')
    const [msg, setMsg] = useState('')

    const refresh = () => {
        getUsers().then((r) => { setUsers(r.data); setSelectedUser(r.data[0]?.username ?? '') }).catch(() => { })
        getHealth().then((r) => setHealth(r.data)).catch(() => { })
    }

    useEffect(() => { if (role === 'admin') refresh() }, [role])

    if (role !== 'admin') return (
        <div style={{ padding: 24, background: 'rgba(251,191,36,0.1)', border: '1px solid rgba(251,191,36,0.3)', borderRadius: 10, color: '#fcd34d' }}>
            ⚠️ Admin privileges required.
        </div>
    )

    const handleCreate = async (e: FormEvent) => {
        e.preventDefault()
        if (!newU || !newP) { setMsg('Fields cannot be empty'); return }
        try { await createUser(newU, newP, newR); setMsg('✅ User created'); setNewU(''); setNewP(''); refresh() }
        catch { setMsg('❌ Failed to create user') }
    }

    const handleDelete = async () => {
        if (!selectedUser) return
        try { await deleteUser(selectedUser); setMsg(`✅ Deleted ${selectedUser}`); refresh() }
        catch { setMsg('❌ Failed to delete user') }
    }

    const handleRoleUpdate = async () => {
        if (!selectedUser) return
        try { await updateRole(selectedUser, newRoleVal); setMsg(`✅ Role updated for ${selectedUser}`); refresh() }
        catch { setMsg('❌ Failed to update role') }
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* System Health */}
            {health && (
                <div style={styles.section}>
                    <h3 style={styles.title}>🖥️ System Health</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
                        {[
                            { label: 'CPU', value: health.cpu_percent != null ? `${health.cpu_percent}%` : 'N/A', color: '#60a5fa' },
                            { label: 'Memory', value: health.memory_percent != null ? `${health.memory_percent}%` : 'N/A', color: '#60a5fa' },
                            { label: 'Database', value: health.db_connected ? '✅ OK' : '❌ Error', color: health.db_connected ? '#22c55e' : '#ef4444' },
                            { label: 'RTL-SDR', value: health.rtlsdr_available ? '✅ Ready' : '❌ Missing', color: health.rtlsdr_available ? '#22c55e' : '#ef4444' },
                            { label: 'Kafka', value: health.kafka_available ? '✅ Active' : '⚠️ Off', color: health.kafka_available ? '#22c55e' : '#f59e0b' },
                        ].map((m) => (
                            <div key={m.label} style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, padding: 12, textAlign: 'center' }}>
                                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{m.label}</div>
                                <div style={{ fontWeight: 700, color: m.color, fontSize: 14 }}>{m.value}</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* User Table */}
            <div style={styles.section}>
                <h3 style={styles.title}>👥 Registered Operators ({users.length})</h3>
                <table style={styles.table}>
                    <thead><tr><th style={styles.th}>Username</th><th style={styles.th}>Role</th></tr></thead>
                    <tbody>
                        {users.map((u) => (
                            <tr key={u.username}>
                                <td style={styles.td}>{u.username}</td>
                                <td style={styles.td}><span style={{ color: u.role === 'admin' ? '#f97316' : u.role === 'analyst' ? '#a78bfa' : '#60a5fa' }}>{u.role}</span></td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Create User */}
            <div style={styles.section}>
                <h3 style={styles.title}>➕ Add Operator</h3>
                <form onSubmit={handleCreate} style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end' }}>
                    {[['Username', newU, setNewU, 'text'], ['Password', newP, setNewP, 'password']].map(([ph, val, setter, type]) => (
                        <input key={String(ph)} placeholder={String(ph)} type={String(type)} value={String(val)} onChange={(e) => (setter as (v: string) => void)(e.target.value)} style={styles.input} />
                    ))}
                    <select value={newR} onChange={(e) => setNewR(e.target.value)} style={styles.input}>
                        {['viewer', 'analyst', 'admin'].map((r) => <option key={r} value={r}>{r}</option>)}
                    </select>
                    <button type="submit" style={styles.btn}>Create</button>
                </form>
            </div>

            {/* Modify User */}
            <div style={styles.section}>
                <h3 style={styles.title}>✏️ Modify Operator</h3>
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-end' }}>
                    <select value={selectedUser} onChange={(e) => setSelectedUser(e.target.value)} style={styles.input}>
                        {users.map((u) => <option key={u.username} value={u.username}>{u.username}</option>)}
                    </select>
                    <select value={newRoleVal} onChange={(e) => setNewRoleVal(e.target.value)} style={styles.input}>
                        {['viewer', 'analyst', 'admin'].map((r) => <option key={r} value={r}>{r}</option>)}
                    </select>
                    <button onClick={handleRoleUpdate} style={styles.btn}>Update Role</button>
                    <button onClick={handleDelete} style={{ ...styles.btn, background: 'rgba(239,68,68,0.8)' }}>Delete</button>
                </div>
            </div>

            {msg && <div style={{ padding: 10, background: 'rgba(255,255,255,0.04)', borderRadius: 8, color: '#94a3b8', fontSize: 13 }}>{msg}</div>}
        </div>
    )
}

const styles: Record<string, React.CSSProperties> = {
    section: { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: 20 },
    title: { margin: '0 0 14px', color: '#e2e8f0', fontSize: 15, fontWeight: 600 },
    table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
    th: { textAlign: 'left', padding: '8px 10px', color: '#64748b', borderBottom: '1px solid rgba(255,255,255,0.08)', fontSize: 11, textTransform: 'uppercase' },
    td: { padding: '8px 10px', color: '#e2e8f0', borderBottom: '1px solid rgba(255,255,255,0.05)' },
    input: { padding: '8px 12px', background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#e2e8f0', fontSize: 13 },
    btn: { padding: '8px 16px', background: 'rgba(59,130,246,0.8)', border: 'none', borderRadius: 8, color: '#fff', fontWeight: 600, cursor: 'pointer', fontSize: 13 },
}
