import { useState } from 'react'
import { auth } from '../services/apiClient'

export function LoginPage() {
  const [mode, setMode] = useState('login') // 'login' or 'register'
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      if (mode === 'login') {
        const response = await auth.login(username, password)
        localStorage.setItem('aegis_token', response.data.access_token)
        window.location.href = '/'
      } else {
        if (password !== confirm) {
          setError('Passwords do not match')
          setLoading(false)
          return
        }
        await auth.register(username, password)
        setError('')
        setMode('login')
        setPassword('')
        setConfirm('')
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Authentication failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-cyan-400 mb-2">⚔️ AEGIS</h1>
          <p className="text-slate-400">Defense Monitoring System</p>
        </div>

        {/* Card */}
        <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-8 backdrop-blur">
          {/* Mode Selector */}
          <div className="flex space-x-4 mb-6 bg-slate-900 p-1 rounded">
            <button
              onClick={() => {
                setMode('login')
                setError('')
              }}
              className={`flex-1 px-4 py-2 rounded font-medium transition-colors ${
                mode === 'login'
                  ? 'bg-cyan-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Login
            </button>
            <button
              onClick={() => {
                setMode('register')
                setError('')
              }}
              className={`flex-1 px-4 py-2 rounded font-medium transition-colors ${
                mode === 'register'
                  ? 'bg-cyan-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Register
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-4 p-3 bg-red-900/30 border border-red-700 rounded text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                {mode === 'login' ? 'Operator ID' : 'Create Operator ID'}
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your ID"
                required
                className="w-full bg-slate-700/50 border border-slate-600 rounded px-4 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                {mode === 'login' ? 'Access Code' : 'Create Access Code'}
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                required
                className="w-full bg-slate-700/50 border border-slate-600 rounded px-4 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition"
              />
            </div>

            {mode === 'register' && (
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Confirm Access Code
                </label>
                <input
                  type="password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  placeholder="Confirm your password"
                  required
                  className="w-full bg-slate-700/50 border border-slate-600 rounded px-4 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition"
                />
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-white font-semibold py-2 px-4 rounded transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Authorizing...' : mode === 'login' ? 'AUTHORIZE' : 'REGISTER'}
            </button>
          </form>

          {/* Footer */}
          <div className="mt-6 text-center text-xs text-slate-500">
            <p>© 2026 Aegis Defense System</p>
            <p>v2.0.0 - Classified</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LoginPage
