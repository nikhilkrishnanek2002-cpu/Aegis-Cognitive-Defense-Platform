import { useState, FormEvent } from 'react'
import { useAuthStore } from '../store/authStore'
import { login, register } from '../api/client'

export default function LoginPage() {
    const [mode, setMode] = useState<'login' | 'register'>('login')
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [confirm, setConfirm] = useState('')
    const [role, setRole] = useState('viewer')
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)
    const { login: storeLogin } = useAuthStore()

    const handleLogin = async (e: FormEvent) => {
        e.preventDefault()
        setLoading(true)
        setError('')
        try {
            const res = await login(username, password)
            storeLogin(res.data.access_token, res.data.username, res.data.role)
        } catch {
            setError('Invalid credentials. Try again.')
        } finally {
            setLoading(false)
        }
    }

    const handleRegister = async (e: FormEvent) => {
        e.preventDefault()
        if (password !== confirm) { setError('Passwords do not match'); return }
        setLoading(true)
        setError('')
        try {
            await register(username, password, role)
            setMode('login')
            setError('')
        } catch (err: unknown) {
            const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
            setError(detail || 'Registration failed')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div style={styles.bg}>
            <div style={styles.card}>
                <div style={styles.logo}>📡</div>
                <h1 style={styles.title}>AEGIS</h1>
                <p style={styles.subtitle}>Cognitive Defense Platform</p>

                {mode === 'login' ? (
                    <form onSubmit={handleLogin} style={styles.form}>
                        <h2 style={styles.formTitle}>🔐 Operator Login</h2>
                        <input style={styles.input} placeholder="Operator ID" value={username} onChange={e => setUsername(e.target.value)} required autoComplete="username" />
                        <input style={styles.input} placeholder="Access Code" type="password" value={password} onChange={e => setPassword(e.target.value)} required autoComplete="current-password" />
                        {error && <p style={styles.error}>{error}</p>}
                        <button style={styles.btn} type="submit" disabled={loading}>{loading ? 'Authorizing...' : 'AUTHORIZE'}</button>
                        <button type="button" style={styles.link} onClick={() => { setMode('register'); setError('') }}>New Operator? Register Here</button>
                    </form>
                ) : (
                    <form onSubmit={handleRegister} style={styles.form}>
                        <h2 style={styles.formTitle}>📝 Register Operator</h2>
                        <input style={styles.input} placeholder="Operator ID" value={username} onChange={e => setUsername(e.target.value)} required />
                        <input style={styles.input} placeholder="Access Code" type="password" value={password} onChange={e => setPassword(e.target.value)} required />
                        <input style={styles.input} placeholder="Confirm Access Code" type="password" value={confirm} onChange={e => setConfirm(e.target.value)} required />
                        <select style={styles.input} value={role} onChange={e => setRole(e.target.value)}>
                            <option value="viewer">Viewer</option>
                            <option value="analyst">Analyst</option>
                        </select>
                        {error && <p style={styles.error}>{error}</p>}
                        <button style={styles.btn} type="submit" disabled={loading}>{loading ? 'Registering...' : 'REGISTER'}</button>
                        <button type="button" style={styles.link} onClick={() => { setMode('login'); setError('') }}>Back to Login</button>
                    </form>
                )}
            </div>
        </div>
    )
}

const styles: Record<string, React.CSSProperties> = {
    bg: { minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%)' },
    card: { background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 16, padding: '40px 48px', width: 380, backdropFilter: 'blur(12px)', boxShadow: '0 25px 50px rgba(0,0,0,0.5)', textAlign: 'center' },
    logo: { fontSize: 56, marginBottom: 8 },
    title: { margin: 0, color: '#60a5fa', fontSize: 32, fontWeight: 800, letterSpacing: 6 },
    subtitle: { color: '#94a3b8', fontSize: 13, marginBottom: 28, marginTop: 4, letterSpacing: 2 },
    form: { display: 'flex', flexDirection: 'column', gap: 12 },
    formTitle: { color: '#e2e8f0', fontSize: 16, marginBottom: 8 },
    input: { padding: '10px 14px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.15)', background: 'rgba(255,255,255,0.07)', color: '#f1f5f9', fontSize: 14, outline: 'none' },
    btn: { padding: '12px', background: 'linear-gradient(90deg, #3b82f6, #2563eb)', color: '#fff', border: 'none', borderRadius: 8, fontWeight: 700, fontSize: 14, cursor: 'pointer', letterSpacing: 1, marginTop: 4 },
    link: { background: 'none', border: 'none', color: '#60a5fa', cursor: 'pointer', fontSize: 13, textDecoration: 'underline', marginTop: 4 },
    error: { color: '#f87171', fontSize: 13, margin: 0 },
}
