import { useEffect } from 'react'
import { useSystemMetrics } from '../hooks/useSystemMetrics'

export function Providers({ children }) {
  // Initialize system metrics polling on app load
  useSystemMetrics(5000) // Poll every 5 seconds

  return <>{children}</>
}

export default Providers
