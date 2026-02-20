# 🚀 React Refactor - Quick Reference Guide

## 📂 Folder Structure at a Glance

```
frontend/src/
├── app/                    # Entry & routing
├── layout/                 # Structural components
├── pages/                  # Full page components (7)
├── components/             # Reusable UI (8)
├── store/                  # Zustand stores (3)
├── services/               # API & WebSocket (2)
├── hooks/                  # Custom hooks (2)
├── styles/                 # Theming & animations
└── App.jsx                 # Root with routes
```

---

## 🎯 What Gets Created in Each Folder

### `pages/` - Full Page Views (7 Components)
| Page | Purpose | Route |
|------|---------|-------|
| Dashboard | Overview & system status | `/dashboard` |
| RadarLive | Real-time radar streaming | `/radar` |
| ThreatAnalysis | Threat details & history | `/threats` |
| EWControl | Electronic Warfare panel | `/ew` |
| ModelMonitor | AI/ML performance | `/monitor` |
| Settings | System configuration | `/settings` |
| LoginPage | Authentication | `/login` |

### `components/` - Reusable UI Parts (8 Components)
| Component | Used In | Purpose |
|-----------|---------|---------|
| Card | Dashboard, all pages | Container with title/action |
| Loader | Suspense fallback | Loading spinner |
| StatusBadge | Topbar, pages | Status indicator |
| RadarCanvas | RadarLive | SVG radar visualization |
| TargetOverlay | RadarLive | Target list display |
| ThreatCard | ThreatAnalysis | Individual threat card |
| ThreatTable | Dashboard, pages | Threats list table |
| SystemHealth | Dashboard | System status display |

### `store/` - Global State (3 Stores)
| Store | Manages | Key Actions |
|-------|---------|-------------|
| radarStore | Targets, frames, scanning | setTargets, setScan |
| threatStore | Active threats, history | addThreat, setSelected |
| systemStore | Health, metrics, events | addEvent, addAlert |

### `services/` - API & WebSocket (2 Services)
| Service | Handles | Methods |
|---------|---------|---------|
| apiClient | REST calls | radar.scan(), threats.getActive() |
| websocketClient | Live streaming | subscribe(), emit(), on() |

### `hooks/` - Custom Logic (2 Hooks)
| Hook | Purpose | Usage |
|------|---------|-------|
| useRadarStream | WebSocket connection | useRadarStream() in RadarLive |
| useSystemMetrics | Polling system status | useSystemMetrics(5000) in Dashboard |

---

## 🔄 Data Flow Pattern

```
User Action (click scan button)
  ↓
API Call via apiClient.js
  ↓
Backend Response
  ↓
Update Zustand Store (radarStore.setTargets)
  ↓
Component Re-render (reads from store)
  ↓
UI Updates
```

## 🔗 WebSocket Flow

```
useRadarStream hook
  ↓
wsClient.connect() (auto-reconnect if fails)
  ↓
wsClient.subscribe('data-stream')
  ↓
Receive radar frames
  ↓
radarStore.setTargets()
  ↓
RadarCanvas component re-renders
```

---

## 💻 Common Developer Tasks

### Add a New Page
1. Create `pages/NewPage.jsx`
2. Add import to `App.jsx`
3. Add route in App.jsx
4. Add link in `layout/Sidebar.jsx`

### Add a New Component
1. Create `components/domain/NewComponent.jsx`
2. Export from parent folder
3. Import in page and use

### Add a New API Endpoint
1. Add method to `services/apiClient.js`
2. Organize under correct group (auth, radar, threats, etc)
3. Import and call from page/hook

### Add New Global State
1. Create new store in `store/`
2. Define state and actions
3. Import `useStore` in component
4. Read and update in component

### Add New Hook
1. Create `hooks/useNewHook.js`
2. Import services/stores needed
3. Manage side effects
4. Export hook function
5. Use in component

---

## 🎨 Styling Guide

### Use Tailwind Classes
```jsx
<div className="bg-slate-900 text-slate-100 p-4 rounded-lg">
  <h1 className="text-xl font-bold text-cyan-400">Title</h1>
</div>
```

### Color Palette
| Element | Color | Tailwind Class |
|---------|-------|----------------|
| Background | Slate-900 | `bg-slate-900` |
| Surface | Slate-800 | `bg-slate-800` |
| Text | Slate-100 | `text-slate-100` |
| Accent | Cyan-400 | `text-cyan-400` |
| DangerOK | Red-500 | `text-red-500` |
| Success | Green-500 | `text-green-500` |

### Animation Classes
```jsx
// In theme.css
.pulse-glow { animation: pulse-glow 2s infinite; }
.slide-in { animation: slide-in 0.3s ease; }
.fade-in { animation: fade-in 0.5s ease; }
```

---

## 🔔 State Management Quick Reference

### Reading State
```jsx
import { useRadarStore } from '@/store/radarStore'

const { targets, isConnected } = useRadarStore()
```

### Updating State
```jsx
import { useRadarStore } from '@/store/radarStore'

const setTargets = useRadarStore(state => state.setTargets)
setTargets(newTargets)
```

### Watching Changes
```jsx
import { useEffect } from 'react'
import { useRadarStore } from '@/store/radarStore'

useEffect(() => {
  const unsub = useRadarStore.subscribe(
    state => state.targets,
    targets => console.log('Targets updated:', targets)
  )
  return unsub
}, [])
```

---

## 📡 API Client Quick Reference

### Making Requests
```jsx
import { radar, threats, admin } from '@/services/apiClient'

// GET
const data = await radar.getStatus()

// POST
const result = await threats.create({ name: 'Threat1' })

// Token auto-injected in header
```

### Organized Endpoints
```javascript
radar.*              // All radar endpoints
threats.*            // All threat endpoints  
auth.*               // Login/register
ew.*                 // Electronic warfare
visualizations.*     // Chart/visualization data
admin.*              // Admin operations
metrics.*            // Performance metrics
```

---

## 🔌 WebSocket Quick Reference

### Connect & Listen
```jsx
import { useRadarStream } from '@/hooks/useRadarStream'

// In component
useRadarStream() // Auto-manages connection

// Listen for data
wsClient.on('data', frame => {
  console.log('New frame:', frame)
})
```

### Broadcasting
```jsx
wsClient.emit('command', { action: 'scan' })
```

---

## 🧪 Testing Checklist

- ✅ Dev server starts: `npm run dev`
- ✅ All pages load
- ✅ Sidebar navigation works
- ✅ Login redirects properly
- ✅ Dashboard shows metrics
- ✅ Radar canvas renders
- ✅ WebSocket connects
- ✅ Threat table updates
- ✅ System health displays
- ✅ Logout clears token

---

## 🐛 Troubleshooting

### Page not loading?
1. Check route in App.jsx
2. Check lazy import syntax
3. Check component export
4. See browser console for errors

### WebSocket not connecting?
1. Check backend is running
2. Check address in websocketClient.js
3. Check firewall/ports
4. See browser Network tab

### State not updating?
1. Check store action called
2. Check component subscribed to store
3. Check no console errors
4. Use Zustand devtools

### API returning 401?
1. Check token in localStorage
2. Check login worked
3. Check token not expired
4. Check API has endpoint

---

## 📊 Performance Tips

| Optimization | Where | Impact |
|--------------|-------|--------|
| Lazy load pages | App.jsx | Smaller initial bundle |
| Zustand over Redux | store/ | Simpler, less re-renders |
| Custom hooks | hooks/ | Extract logic, reuse |
| Event Emitter | websocketClient | Efficient pub-sub |
| CSS utilities | theme.css | Optimized CSS output |
| React.lazy | routes | Code splitting |
| Suspense | App.jsx | Better UX on load |

---

## 🔐 Security Notes

- ✅ JWT token auto-injected in requests
- ✅ 401 responses redirect to login
- ✅ Protected routes check auth
- ✅ WebSocket message validation needed
- ✅ CORS configured for backend
- ✅ No sensitive data in localStorage (only JWT)

---

## 📚 File Templates

### New Page Template
```jsx
import { useEffect } from 'react'
import { useRadarStore } from '@/store/radarStore'

export default function NewPage() {
  const { data } = useRadarStore()
  
  useEffect(() => {
    // Side effects here
  }, [])
  
  return (
    <div className="p-4">
      <h1>New Page</h1>
      {/* Content */}
    </div>
  )
}
```

### New Component Template
```jsx
export default function NewComponent({ title, data }) {
  return (
    <div className="p-4 bg-slate-800 rounded-lg">
      <h2>{title}</h2>
      {/* Render data */}
    </div>
  )
}
```

### New Store Template
```javascript
import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

const useMyStore = create(
  devtools(set => ({
    data: [],
    setData: data => set({ data }),
    addItem: item => set(state => ({ 
      data: [...state.data, item] 
    })),
  }), { name: 'myStore' })
)

export default useMyStore
```

---

## 🎯 Key Principles

1. **One Responsibility** - Each file does one thing
2. **No Prop Drilling** - Use stores for shared state
3. **Reusable Components** - Extract to components/
4. **Centralized API** - All calls through apiClient
5. **Custom Hooks** - Extract complex logic
6. **Consistent Styling** - Use Tailwind + theme.css
7. **Error Handling** - Handle failures gracefully
8. **Performance First** - Optimize from start

---

## 🚀 Launch Commands

```bash
# Start development
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Type check (when TypeScript added)
npm run type-check

# Lint (when ESLint added)
npm run lint

# Test (when Vitest added)
npm run test
```

---

**Remember**: This structure is designed for growth. Add files following existing patterns, and the app scales smoothly! 🚀
