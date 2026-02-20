# 🛡️ Aegis Defense Monitoring UI - Refactored Architecture

Professional real-time defense monitoring dashboard built with React 18 + Vite + Zustand.

## 📐 Architecture Overview

```
src/
├── app/                          # Application setup
│   ├── router.jsx               # React Router v6 configuration
│   └── providers.jsx            # Global providers
│
├── layout/                       # Layout components
│   ├── DashboardLayout.jsx      # Main layout wrapper
│   ├── Sidebar.jsx              # Navigation sidebar
│   └── Topbar.jsx               # Top navigation bar
│
├── pages/                        # Page components
│   ├── Dashboard.jsx            # Overview dashboard
│   ├── RadarLive.jsx            # Real-time radar monitoring
│   ├── ThreatAnalysis.jsx       # Threat detailed view
│   ├── EWControl.jsx            # Electronic Warfare control
│   ├── ModelMonitor.jsx         # AI/ML model performance
│   └── Settings.jsx             # System settings
│
├── components/                   # Reusable components
│   ├── radar/
│   │   ├── RadarCanvas.jsx      # SVG radar visualization
│   │   └── TargetOverlay.jsx    # Target list display
│   ├── threat/
│   │   ├── ThreatCard.jsx       # Individual threat card
│   │   └── ThreatTable.jsx      # Threat table view
│   ├── system/
│   │   ├── SystemHealth.jsx     # System status display
│   │   └── StatusBadge.jsx      # Status indicator
│   └── common/
│       ├── Card.jsx             # Reusable card component
│       └── Loader.jsx           # Loading spinner
│
├── store/                        # Zustand stores (global state)
│   ├── radarStore.js            # Radar state management
│   ├── threatStore.js           # Threat state management
│   └── systemStore.js           # System state management
│
├── services/                     # API & WebSocket clients
│   ├── apiClient.js             # Axios REST client
│   └── websocketClient.js       # WebSocket with auto-reconnect
│
├── hooks/                        # Custom React hooks
│   ├── useRadarStream.js        # Radar data streaming hook
│   └── useSystemMetrics.js      # System metrics polling hook
│
├── styles/                       # Global styles
│   └── theme.css                # Design system & animations
│
├── App.jsx                       # Root component with routing
├── main.jsx                      # Vite entry point
└── index.css                     # Base styles
```

## 🚀 Key Features

### Real-Time Data Streaming
- **WebSocket Connection**: Automatic reconnection with exponential backoff
- **Event Emitter Pattern**: Flexible event subscribing
- **Auto-Reconnect**: Max 10 retry attempts with increasing delays

### State Management
- **Zustand Store**: Minimal, performant global state
- **DevTools Integration**: Redux DevTools support for debugging
- **Computed Selectors**: Derived state and filtering

### API Integration
- **Centralized Client**: All API calls through `services/apiClient.js`
- **Automatic Token Injection**: JWT token management
- **Error Handling**: 401 redirect on auth failure
- **Request Interceptors**: Consistent header/auth setup

### Component Architecture
- **Functional Components**: React 18 hooks only
- **Custom Hooks**: Extract reusable logic
- **Lazy Loading**: Code splitting for pages
- **Suspense Boundaries**: Graceful loading states

### UI/UX
- **Professional Dark Theme**: Slate 900/800 base colors
- **Cyan Accent Colors**: Modern, defense-focused palette
- **Responsive Grid**: Adapts to all screen sizes
- **Real-Time Updates**: Sub-second UI refresh
- **Smooth Animations**: CSS transitions and keyframes

## 📦 Dependencies

```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "react-router-dom": "^6.x.x",
  "zustand": "^4.x.x",
  "axios": "^1.x.x",
  "eventemitter3": "^4.x.x",
  "plotly.js": "^2.x.x",
  "tailwindcss": "^3.x.x"
}
```

## 🔄 Data Flow

### REST API Flow
```
Component → Hook (useRadarStream) → apiClient.js → Zustand Store → Component Re-render
```

### WebSocket Flow
```
WebSocket Data → WebSocketClient (EventEmitter) → Hook → Zustand Store → Component
```

### State Update Pattern
```
User Action → Hook Function → API/Service Call → Store Update → Auto Re-render
```

## 🏗️ Component Usage Examples

### Using Radar Stream
```jsx
import { useRadarStream, useTriggerScan } from '@/hooks/useRadarStream'
import { useRadarStore } from '@/store/radarStore'

function RadarPage() {
  useRadarStream() // Connect to WebSocket
  const { targets, isConnected } = useRadarStore()
  const triggerScan = useTriggerScan()

  return (
    <div>
      {targets.map(target => <TargetCard key={target.id} {...target} />)}
    </div>
  )
}
```

### Using System Metrics
```jsx
import { useSystemMetrics, useSystemHealth } from '@/hooks/useSystemMetrics'
import { useSystemStore } from '@/store/systemStore'

function HealthPanel() {
  useSystemMetrics(5000) // Poll every 5 seconds
  const { health, isHealthy } = useSystemHealth()
  const { events } = useSystemStore()

  return (
    // Display health data
  )
}
```

### Creating Stores
```jsx
import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

export const useRadarStore = create(
  devtools((set) => ({
    targets: [],
    setTargets: (targets) => set({ targets }),
  }))
)
```

## 🔐 Security

- **JWT Token Management**: Auto-inject auth headers
- **Protected Routes**: Redirect unauthenticated users to login
- **Token Expiration**: Handle 401 responses
- **CORS**: Properly configured for frontend domain

## 🚄 Performance Optimizations

- **Code Splitting**: Lazy load pages with React.lazy
- **Suspense Boundaries**: Show loader during code load
- **Event Emitter**: Efficient WebSocket event handling
- **Zustand DevTools**: Debug state without overhead
- **Memoization**: Component optimization with React.memo (optional)

## 🛠️ Development

### Start Development Server
```bash
npm run dev
# Runs on http://localhost:3000
```

### Build for Production
```bash
npm run build
# Creates dist/ folder
```

### Environment Variables
```bash
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## 📊 Store Structure

### radarStore
- `targets`: Detected targets
- `frame`: Current radar frame
- `scanHistory`: Previous scans
- `isConnected`: WebSocket status
- `isScanning`: Scan in progress

### threatStore
- `threats`: All threats
- `activeThreats`: Currently active
- `threatHistory`: Historical data
- `ewThreats`: Electronic warfare threats
- `selectedThreat`: Currently selected

### systemStore
- `health`: System status
- `metrics`: Performance metrics
- `status`: Operational status
- `events`: System events log
- `alerts`: Active alerts

## 🎯 Refactoring Highlights

✅ **Before**
- Direct fetch() calls scattered everywhere
- React Context prop drilling
- Inline component styles
- Mixed concerns in pages
- State management inconsistent

✅ **After**
- Centralized API client with Axios
- Zustand global store
- Tailwind CSS + CSS module
- Separated concerns (pages → components → hooks → services)
- Consistent error handling
- Type-safe (ready for TypeScript)
- Production-ready code quality

## 🚀 Next Steps

1. **Add TypeScript**: Convert `.js` to `.ts`
2. **Implement Plots**: Use Plotly.js for 3D/2D charts
3. **Add Unit Tests**: Jest + React Testing Library
4. **Setup CI/CD**: GitHub Actions pipeline
5. **Add E2E Tests**: Cypress automation
6. **Performance Monitoring**: Sentry integration
7. **PWA Support**: Service workers & offline mode

## 📝 File Size Reference

- `App.jsx`: ~3KB
- `radarStore.js`: ~1.5KB
- `apiClient.js`: ~2KB
- `websocketClient.js`: ~2.5KB
- All CSS: ~5KB
- **Total Gzipped**: ~15KB (excluding React/dependencies)

---

**Architecture Design**: Production-ready, scalable, maintainable  
**Build Tool**: Vite (near-instant HMR)  
**State**: Zustand (minimal boilerplate)  
**Styling**: Tailwind CSS (utility-first)  
**Performance**: Lazy loading + code splitting

🎉 **Ready for deployment!**
