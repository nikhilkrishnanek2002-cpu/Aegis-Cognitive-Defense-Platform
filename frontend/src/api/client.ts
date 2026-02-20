import axios from 'axios'

export const API_BASE = 'http://localhost:8000'

const api = axios.create({ baseURL: '/api' })

// Attach JWT token from localStorage to every request
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('aegis_token')
    if (token) config.headers.Authorization = `Bearer ${token}`
    return config
})

// ─── Auth ─────────────────────────────────────────────────────────────────────
export const login = (username: string, password: string) =>
    api.post<{ access_token: string; role: string; username: string }>('/auth/login', { username, password })

export const register = (username: string, password: string, role: string) =>
    api.post('/auth/register', { username, password, role })

// ─── Radar ────────────────────────────────────────────────────────────────────
export const scanRadar = (target: string, distance: number, gain_db: number) =>
    api.post('/radar/scan', { target, distance, gain_db })

export const getLabels = () => api.get('/radar/labels')

// ─── Tracks ───────────────────────────────────────────────────────────────────
export const getTracks = () => api.get('/tracks')
export const resetTracks = () => api.delete('/tracks/reset')

// ─── Admin ────────────────────────────────────────────────────────────────────
export const getUsers = () => api.get('/admin/users')
export const createUser = (username: string, password: string, role: string) =>
    api.post('/admin/users', { username, password, role })
export const deleteUser = (username: string) => api.delete(`/admin/users/${username}`)
export const updateRole = (username: string, role: string) =>
    api.patch(`/admin/users/${username}/role`, { role })
export const getHealth = () => api.get('/admin/health')

// ─── Metrics ──────────────────────────────────────────────────────────────────
export const getMetricsReport = () => api.get('/metrics/report')

export default api
