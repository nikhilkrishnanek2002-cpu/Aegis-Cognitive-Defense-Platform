import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { DashboardLayout } from '../layout/DashboardLayout'
import Dashboard from '../pages/Dashboard'
import RadarLive from '../pages/RadarLive'
import ThreatAnalysis from '../pages/ThreatAnalysis'
import EWControl from '../pages/EWControl'
import ModelMonitor from '../pages/ModelMonitor'
import Settings from '../pages/Settings'

export function AppRouter() {
  const isAuthenticated = !!localStorage.getItem('aegis_token')

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Navigate to="/" replace />} />
        
        <Route element={<DashboardLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="radar" element={<RadarLive />} />
          <Route path="threats" element={<ThreatAnalysis />} />
          <Route path="ew" element={<EWControl />} />
          <Route path="monitor" element={<ModelMonitor />} />
          <Route path="settings" element={<Settings />} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default AppRouter
