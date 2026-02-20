import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const API_PATH = `${API_BASE}/api`

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_PATH,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Attach JWT token to every request
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('aegis_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle response errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('aegis_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// ─── Auth ─────────────────────────────────────────────────────────────────────
export const auth = {
  login: (username, password) =>
    apiClient.post('/auth/login', { username, password }),
  register: (username, password, role = 'operator') =>
    apiClient.post('/auth/register', { username, password, role }),
  refresh: () => apiClient.post('/auth/refresh'),
}

// ─── Radar ────────────────────────────────────────────────────────────────────
export const radar = {
  scan: (params = {}) =>
    apiClient.post('/radar/scan', { signal_source: 'generated', ...params }),
  getLabels: () => apiClient.get('/radar/labels'),
  getHistory: () => apiClient.get('/radar/history'),
  getTargets: () => apiClient.get('/radar/targets'),
}

// ─── Threats ──────────────────────────────────────────────────────────────────
export const threats = {
  getAll: () => apiClient.get('/threats'),
  getActive: () => apiClient.get('/threats?status=active'),
  getById: (id) => apiClient.get(`/threats/${id}`),
}

// ─── EW ───────────────────────────────────────────────────────────────────────
export const ew = {
  getStatus: () => apiClient.get('/ew/status'),
  getSignals: () => apiClient.get('/ew/signals'),
  analyze: (signal) => apiClient.post('/ew/analyze', signal),
}

// ─── Visualizations ───────────────────────────────────────────────────────────
export const visualizations = {
  performance: () => apiClient.get('/visualizations/performance-charts'),
  confusion: () => apiClient.get('/visualizations/confusion-matrix'),
  roc: () => apiClient.get('/visualizations/roc-curve'),
  precisionRecall: () => apiClient.get('/visualizations/precision-recall'),
  training: () => apiClient.get('/visualizations/training-history'),
  surface3d: () => apiClient.get('/visualizations/3d-surface-plot'),
  gradcam: (scanId) => apiClient.get(`/visualizations/xai-gradcam/${scanId}`),
}

// ─── Admin ────────────────────────────────────────────────────────────────────
export const admin = {
  health: () => apiClient.get('/admin/health'),
  metrics: () => apiClient.get('/admin/metrics'),
  users: () => apiClient.get('/admin/users'),
  createUser: (user) => apiClient.post('/admin/users', user),
  deleteUser: (username) => apiClient.delete(`/admin/users/${username}`),
}

// ─── Metrics ──────────────────────────────────────────────────────────────────
export const metrics = {
  report: () => apiClient.get('/metrics/report'),
  performance: () => apiClient.get('/metrics/performance'),
  accuracy: () => apiClient.get('/metrics/accuracy'),
}

export default apiClient
