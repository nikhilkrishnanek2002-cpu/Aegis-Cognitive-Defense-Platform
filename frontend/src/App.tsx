import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './store/authStore'
import LoginPage from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import { Component, ReactNode } from 'react'

interface ErrorBoundaryProps {
    children: ReactNode
}

interface ErrorBoundaryState {
    hasError: boolean
    error?: Error
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
    constructor(props: ErrorBoundaryProps) {
        super(props)
        this.state = { hasError: false }
    }

    static getDerivedStateFromError(error: Error) {
        return { hasError: true, error }
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
        console.error('🚨 [Error Boundary] Caught error:', error, errorInfo)
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    minHeight: '100vh',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: '#0f172a',
                    color: '#e2e8f0',
                    flexDirection: 'column',
                    gap: 16,
                    fontFamily: "'Inter', sans-serif",
                    padding: 20
                }}>
                    <div style={{ fontSize: 48 }}>⚠️</div>
                    <h1 style={{ fontSize: 24, fontWeight: 700, margin: 0 }}>Component Error</h1>
                    <p style={{ color: '#94a3b8', fontSize: 14, maxWidth: 400, textAlign: 'center', margin: 0 }}>
                        {this.state.error?.message || 'An unexpected error occurred in a component. Refreshing the page may help.'}
                    </p>
                    <button
                        onClick={() => {
                            this.setState({ hasError: false })
                            window.location.href = '/'
                        }}
                        style={{
                            background: '#60a5fa',
                            border: 'none',
                            color: '#fff',
                            padding: '10px 20px',
                            borderRadius: 6,
                            cursor: 'pointer',
                            fontWeight: 600,
                            marginTop: 16
                        }}
                    >
                        🔄 Go to Home
                    </button>
                </div>
            )
        }

        return this.props.children
    }
}

function PrivateRoute({ children }: { children: JSX.Element }) {
    const { token } = useAuthStore()
    return token ? children : <Navigate to="/login" replace />
}

export default function App() {
    return (
        <ErrorBoundary>
            <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
                <Routes>
                    <Route path="/login" element={<LoginPage />} />
                    <Route
                        path="/*"
                        element={
                            <PrivateRoute>
                                <DashboardPage />
                            </PrivateRoute>
                        }
                    />
                </Routes>
            </BrowserRouter>
        </ErrorBoundary>
    )
}
