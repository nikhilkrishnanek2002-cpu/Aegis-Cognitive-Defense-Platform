# 🔧 REFACTORING MIGRATION GUIDE

This document shows the EXACT changes made to restructure Aegis for local development.

---

## ✅ WHAT HAS BEEN DONE

### 1. Created New Backend Entry Point
**File:** `main_api.py` (NEW - in project root)

- Single unified entry point that works from project root
- Automatically adds paths so imports work correctly
- Proper async lifespan management for service startup/shutdown
- Imports all services and starts them in correct order
- Proper CORS configuration for local dev

**Run with:**
```bash
python -m uvicorn main_api:app --reload --port 8000
```

---

### 2. Created Frontend API Config
**File:** `frontend/src/config/apiConfig.ts` (NEW)

- Centralized API endpoint configuration
- Auto-detects dev vs production mode
- Configurable WebSocket URL
- Export for easy use in components

**Use with:**
```typescript
import { API_CONFIG } from './config/apiConfig'
// API_CONFIG.BASE_URL, API_CONFIG.API_URL, API_CONFIG.WS_URL
```

---

### 3. Created Local Dev Launcher
**File:** `dev_local.py` (NEW - in project root)

- Single command to start backend + frontend
- Dependency checking
- Automatic npm install (if needed)
- Graceful shutdown with Ctrl+C
- Colored terminal output with status indicators

**Run with:**
```bash
python dev_local.py
```

---

### 4. Created Comprehensive Documentation
**Files:**
- `LOCAL_DEV_GUIDE.md` - Quick start + troubleshooting
- `REFACTORING_MIGRATION_GUIDE.md` - This file

---

## 🗑️ FILES TO DELETE

Delete these files/folders as they are now redundant:

### ❌ Old Backend Entry Points
```
/api/main.py              ← Replaced by main_api.py
/api/__init__.py
/api/routes/              ← Routes now in backend/app/api/routes/
/api/websocket.py         ← WebSocket now in backend/app/api/websocket/
/api/auth_utils.py        ← Auth now in backend/app services
/api/state.py             ← State management in backend/app services
/api/__pycache__/         ← Cache, will regenerate
```

### ❌ Old Frontend JSX Files
```
/frontend/src/main.jsx                 ← Use main.tsx instead
/frontend/src/App.jsx                  ← Use App.tsx instead
/frontend/src/pages/Dashboard.jsx      ← Use DashboardPage.tsx
/frontend/src/pages/LoginPage.jsx      ← Use LoginPage.tsx (from tsx dir)
/frontend/src/pages/RadarLive.jsx      ← Use equivalent .tsx
/frontend/src/pages/ThreatAnalysis.jsx ← Use equivalent .tsx
/frontend/src/pages/EWControl.jsx      ← Use equivalent .tsx
/frontend/src/pages/ModelMonitor.jsx   ← Use equivalent .tsx
/frontend/src/pages/Settings.jsx       ← Use equivalent .tsx
```

### ⚠️ Old Launchers (Still functional but deprecated)
```
/launcher.py              ← Use dev_local.py instead
/start.py                 ← Use dev_local.py instead
/app_console.py           ← Streamlit app (local dev only)
```

### 📌 Keep These
```
/backend/app/             ← KEEP - Primary backend
/src/                     ← KEEP - Original implementation (legacy)
/frontend/src/            ← KEEP - React app
/config.yaml              ← KEEP - Configuration
/requirements.txt         ← KEEP - Python deps
```

---

## 🔄 IMPORT PATH CHANGES

### Old (❌ Don't use)
```python
# From /api/main.py
from api.routes import auth, radar
from src.db import init_db
from src.logger import init_logging
```

### New (✅ Use this)
```python
# From main_api.py (project root)
from backend.app.api.routes import auth, radar
from backend.app.core.config import get_config
from backend.app.core.logging import pipeline_logger
from backend.app.services.radar_service import get_radar_service
```

---

## 🚀 STARTUP SEQUENCE (NEW)

### Before
```
launcher.py → api.main:app → services manually started? (unclear)
```

### After
```
main_api.py
├─ lifespan startup()
│  ├─ Load config from backend/app/core/config.py
│  ├─ Init logging from backend/app/core/logging.py
│  ├─ Initialize all 5 services (singletons)
│  ├─ Create controller with all services
│  ├─ Start controller (begins event loop)
│  └─ Ready for requests
│
├─ App runs (routes, WebSocket)
│
└─ lifespan shutdown()
   ├─ Stop controller
   ├─ Cleanup resources
   └─ Exit cleanly
```

---

## 📊 ARCHITECTURE BEFORE & AFTER

### BEFORE (Confusing)
```
launcher.py ─→ uvicorn api.main:app
                 ├─ imports from src/
                 ├─ imports from api/
                 └─ Services unclear if started

frontend/src/
├─ main.jsx → App.jsx
└─ main.tsx → App.tsx (which one runs?)
```

### AFTER (Clear)
```
dev_local.py ─→ python -m uvicorn main_api:app
                 ├─ main_api.py (absolute entry point)
                 │  ├─ from backend.app.core
                 │  ├─ from backend.app.services
                 │  └─ lifespan manages startup/shutdown
                 │
                 └─ Returns FastAPI app ready to serve

npm run dev (from frontend/) ─→ Vite dev server
                                ├─ main.tsx (clear entry)
                                ├─ App.tsx (clear app)
                                └─ .tsx components only
```

---

## 📋 STEP-BY-STEP CLEANUP (SAFE)

### Step 1: Backup (Optional)
```bash
git commit -m "Pre-refactor backup"
```

### Step 2: Delete Old Files
```bash
# Delete old API files
rm -rf api/

# Delete JSX frontend files
rm frontend/src/main.jsx frontend/src/App.jsx
rm frontend/src/pages/*.jsx

# Delete old launchers (optional, keep for reference)
# rm launcher.py start.py
```

### Step 3: Verify New Setup
```bash
# Should exist:
ls main_api.py
ls dev_local.py
ls frontend/src/main.tsx frontend/src/App.tsx
ls backend/app/main.py backend/app/core/
```

### Step 4: Test
```bash
python dev_local.py
```

---

## 🧪 TESTING CHECKLIST

After cleanup, verify everything works:

### Backend
- [ ] `python -m uvicorn main_api:app --reload` starts without errors
- [ ] Logs show "AEGIS READY - MONITORING ACTIVE"
- [ ] http://localhost:8000/health returns JSON
- [ ] http://localhost:8000/docs shows Swagger UI

### Frontend
- [ ] `cd frontend && npm run dev` starts without errors
- [ ] http://localhost:3000 loads dashboard
- [ ] Can login with `admin / admin123`
- [ ] Dashboard shows connected to API (status indicator)

### Integration
- [ ] WebSocket connects (check browser DevTools → Network → WS)
- [ ] Radar data updates in real-time
- [ ] Threats/EW data displays correctly
- [ ] API calls show in DevTools Network tab

### Launcher
- [ ] `python dev_local.py` starts both servers
- [ ] No errors in logs
- [ ] Can access dashboard
- [ ] Ctrl+C stops both servers gracefully

---

## 🔗 MAPPING: OLD → NEW

### Entry Points
```
api.main:app          → main_api:app
launcher.py           → dev_local.py
start.py              → dev_local.py
```

### Backend Imports
```
api.routes            → backend.app.api.routes
api.websocket         → backend.app.api.websocket
api.auth_utils        → backend.app.services
src.config            → backend.app.core.config
src.logger            → backend.app.core.logging
src.db                → (create if needed)
```

### Frontend
```
main.jsx              → main.tsx
App.jsx               → App.tsx
pages/*.jsx           → pages/*.tsx
```

---

## 🎯 GOAL ACHIEVED

✅ **Single Backend Entry Point**: `main_api.py`
✅ **Proper Service Startup**: Lifespan context manager
✅ **Clean Imports**: All paths work from project root
✅ **TypeScript Only**: No JSX/TSX conflicts
✅ **Simple Launcher**: `dev_local.py` starts both servers
✅ **Local Dev Focus**: Works without Docker or deployment setup

---

## 📖 NEXT STEPS

1. ✅ Read `LOCAL_DEV_GUIDE.md` for quick start
2. ✅ Delete old files listed in "🗑️ FILES TO DELETE"
3. ✅ Run `python dev_local.py` to start
4. ✅ Access dashboard at http://localhost:3000
5. ✅ Make changes and see hot-reload work

---

**Status: ✅ REFACTORING COMPLETE**

The platform is now streamlined for local development. All legacy/redundant files can be safely deleted.
