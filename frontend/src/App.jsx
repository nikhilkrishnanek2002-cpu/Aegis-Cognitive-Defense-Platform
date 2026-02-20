import { useEffect, lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { DashboardLayout } from './layout/DashboardLayout'
import { Loader } from './components/common/Loader'
import { useSystemMetrics } from './hooks/useSystemMetrics'
import Dashboard from './pages/Dashboard'
import RadarLive from './pages/RadarLive'
import ThreatAnalysis from './pages/ThreatAnalysis'
import EWControl from './pages/EWControl'
import ModelMonitor from './pages/ModelMonitor'
import Settings from './pages/Settings'
import LoginPage from './pages/LoginPage'

// Lazy load pages for better performance
const LazyDashboard = lazy(() => import('./pages/Dashboard'))
const LazyRadarLive = lazy(() => import('./pages/RadarLive'))
const LazyThreatAnalysis = lazy(() => import('./pages/ThreatAnalysis'))
const LazyEWControl = lazy(() => import('./pages/EWControl'))
const LazyModelMonitor = lazy(() => import('./pages/ModelMonitor'))
const LazySettings = lazy(() => import('./pages/Settings'))

const PageLoader = () => <Loader text="Loading page..." />

function ProtectedRoute({ children }) {
  const isAuthenticated = !!localStorage.getItem('aegis_token')
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }
  return children
}

export function App() {
  // Initialize system metrics
  useSystemMetrics(5000)

  return (
    <BrowserRouter>
      <Routes>
        {/* Public Routes */}
        <Route path="/login" element={<LoginPage />} />

        {/* Protected Routes */}
        <Route
          element={
            <ProtectedRoute>
              <DashboardLayout />
            </ProtectedRoute>
          }
        >
          <Route
            index
            element={
              <Suspense fallback={<PageLoader />}>
                <Dashboard />
              </Suspense>
            }
          />
          <Route
            path="radar"
            element={
              <Suspense fallback={<PageLoader />}>
                <RadarLive />
              </Suspense>
            }
          />
          <Route
            path="threats"
            element={
              <Suspense fallback={<PageLoader />}>
                <ThreatAnalysis />
              </Suspense>
            }
          />
          <Route
            path="ew"
            element={
              <Suspense fallback={<PageLoader />}>
                <EWControl />
              </Suspense>
            }
          />
          <Route
            path="monitor"
            element={
              <Suspense fallback={<PageLoader />}>
                <ModelMonitor />
              </Suspense>
            }
          />
          <Route
            path="settings"
            element={
              <Suspense fallback={<PageLoader />}>
                <Settings />
              </Suspense>
            }
          />
        </Route>

        {/* Catch all */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
