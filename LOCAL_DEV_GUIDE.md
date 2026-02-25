# AEGIS LOCAL DEVELOPMENT SETUP (NO DOCKER)

This guide shows how to run the entire Aegis Cognitive Defense Platform locally in development mode.

---

## 🚀 QUICK START (5 minutes)

### Prerequisites
- **Python 3.8+** with venv
- **Node.js 16+** with npm

### Option 1: Automated One-Command Start

```bash
python dev_local.py
```

This:
- ✓ Checks dependencies
- ✓ Installs Node modules (if needed)
- ✓ Starts Backend API on `http://localhost:8000`
- ✓ Starts React frontend on `http://localhost:3000`

Then open: **http://localhost:3000** and login with `admin / admin123`

Press `Ctrl+C` to stop both servers.

---

### Option 2: Manual Start (Two Terminals)

**Terminal 1 - Backend API:**
```bash
pip install -r requirements.txt
python -m uvicorn main_api:app --reload --port 8000
```

**Terminal 2 - Frontend (in `frontend/` folder):**
```bash
cd frontend
npm install  # (first time only)
npm run dev
```

Open: **http://localhost:3000**

---

## 📁 PROJECT STRUCTURE (CLEANED UP)

```
Aegis-Cognitive-Defense-Platform/
│
├── main_api.py                 ← MAIN BACKEND ENTRY POINT (NEW)
├── dev_local.py                ← LOCAL DEV LAUNCHER (NEW)
├── requirements.txt            ← Python dependencies
│
├── backend/
│   └── app/                    ← PRIMARY BACKEND (OFFICIAL)
│       ├── main.py
│       ├── core/               ← Config, logging
│       ├── services/           ← Radar, detection, tracking, threat, EW
│       ├── engine/             ← Controller, event bus
│       ├── api/
│       │   ├── routes/         ← Auth, radar, threats, metrics
│       │   └── websocket/      ← Real-time stream
│       └── models/             ← Data schemas
│
├── frontend/
│   ├── src/
│   │   ├── main.tsx            ← Entry point (TypeScript)
│   │   ├── App.tsx             ← App component (TypeScript)
│   │   ├── config/
│   │   │   └── apiConfig.ts    ← API/WS URL config (NEW)
│   │   ├── api/                ← API clients (TypeScript)
│   │   ├── services/           ← API calls (JavaScript)
│   │   ├── pages/              ← All .tsx pages (TypeScript only)
│   │   ├── components/         ← Reusable UI
│   │   └── ...
│   ├── index.html              ← Loads src/main.tsx
│   ├── vite.config.ts          ← Configured for /api proxy
│   └── package.json
│
├── src/                        ← LEGACY (Keep for now, not used by API)
│   ├── config.py
│   ├── logger.py
│   └── ... (original implementation)
│
└── docs/, experiments/, results/, scripts/, tests/, tools/
```

---

## 🔗 API ENDPOINTS

### Base URL
```
http://localhost:8000
```

### Authentication
```
POST   /api/auth/login                  # Login
POST   /api/auth/register               # Register
```

### Radar Operations
```
GET    /api/radar/scan                  # Get latest scan
POST   /api/radar/scan                  # Start scan
GET    /api/radar/targets               # Get detected targets
GET    /api/radar/labels                # Get class labels
```

### Threat Analysis
```
GET    /api/threats                     # Get all threats
GET    /api/threats/{threat_id}         # Get threat details
```

### Electronic Warfare
```
GET    /api/ew/status                   # EW defense status
GET    /api/ew/signals                  # Detected jamming signals
```

### Metrics & Health
```
GET    /api/health                      # Health check
GET    /api/metrics/report              # System metrics
```

### WebSocket (Real-time)
```
WS     /ws/stream                       # Live radar frames
```

---

## 🔐 LOGIN CREDENTIALS

Default admin account:
```
Username: admin
Password: admin123
```

---

## 📝 FRONTEND CONFIGURATION

The frontend automatically connects to:
- **Dev Mode** (`npm run dev`): `http://localhost:8000`
- **Proxy**: Vite proxies `/api` → backend

Edit `frontend/src/config/apiConfig.ts` to change URLs:
```typescript
const DEV_API_URL = 'http://localhost:8000'  // Change this
```

---

## 🐛 TROUBLESHOOTING

### "Address already in use" on port 8000
```bash
# Kill process on port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -i :8000
kill -9 <PID>
```

### Frontend can't reach API
1. Check backend is running: `http://localhost:8000/health` should return JSON
2. Check vite config has proxy: `frontend/vite.config.ts` should have `'/api': 'http://localhost:8000'`
3. Clear React DevTools cache: `npm cache clean --force`

### WebSocket connection fails
- WS URL should be: `ws://localhost:8000/ws/stream`
- Check browser console (DevTools → Console)
- Backend log should show WebSocket connections

### Python dependencies fail
```bash
# Create fresh virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 SERVICE STARTUP SEQUENCE

When backend starts (`python -m uvicorn main_api:app`):

1. ✓ Load config (`backend/app/core/config.py`)
2. ✓ Initialize logging
3. ✓ Initialize services:
   - RadarService (signal scanning)
   - DetectionService (target detection)
   - TrackingService (target tracking)
   - ThreatService (threat assessment)
   - EWService (electronic warfare)
4. ✓ Start event bus
5. ✓ Start pipeline controller (begins scanning loop)
6. ✓ Register routes
7. ✓ Listen on port 8000
8. ✓ Ready for WebSocket connections

---

## 🗑️ WHAT WAS REMOVED/DEPRECATED

Old entry points (replaced by `main_api.py`):
- ❌ `/api/main.py` (legacy FastAPI)
- ❌ `/api/routes/` (old route handlers)
- ❌ `/api/websocket.py` (old WS implementation)
- ❌ `/api/__init__.py`

Old frontend files (use TypeScript only):
- ❌ `frontend/src/main.jsx` (use `main.tsx`)
- ❌ `frontend/src/App.jsx` (use `App.tsx`)
- ❌ `frontend/src/pages/*.jsx` (use `.tsx` versions)

Launchers (use `dev_local.py`):
- ⚠️ `launcher.py` (old multi-component launcher)
- ⚠️ `start.py` (old simple launcher)
- ⚠️ `app_console.py` (Streamlit app)

---

## 🔄 DEVELOPMENT WORKFLOW

### Make backend changes
1. Edit files in `backend/app/`
2. Backend auto-reloads (Uvicorn `--reload` flag)
3. Frontend automatically reconnects via WebSocket

### Make frontend changes
1. Edit files in `frontend/src/`
2. Frontend hot-reloads automatically (Vite HMR)

### Testing endpoints
```bash
# Test API health
curl http://localhost:8000/health

# Test API docs (interactive)
open http://localhost:8000/docs

# Test WebSocket (with wscat)
npm install -g wscat
wscat -c ws://localhost:8000/ws/stream
```

---

## 📦 PRODUCTION (NOT COVERED HERE)

This setup is **LOCAL DEVELOPMENT ONLY**. For production:
- Use Docker Compose (see `docker-compose.yml` if it exists)
- Set environment variables (JWT_SECRET, API_HOST, etc.)
- Use external database (not SQLite)
- Set CORS properly
- Use HTTPS and WSS
- Deploy to cloud (AWS, Azure, GCP)

---

## 📞 SUPPORT

For issues:
1. Check logs: Backend logs print to console
2. Check browser DevTools: Frontend errors in Console tab
3. Check API docs: http://localhost:8000/docs
4. Review startup sequence above

---

**Happy developing! 🚀**
