# Local Development Quickstart

**Date:** February 24, 2026  
**Purpose:** Minimal setup to run Aegis frontend + backend locally  
**Estimated Time:** 5-10 minutes

---

## Prerequisites

- Python 3.10+ (backend)
- Node.js 18+ with npm (frontend)
- Two terminal windows (one for backend, one for frontend)

---

## Backend Setup & Run

### Install Dependencies
```bash
cd backend
pip install -r ../requirements.txt
```

### Start Backend Server
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
[1/5] Initializing services...
[2/5] Loading models...
[3/5] Creating pipeline...
...
✓ All services verified ready
```

**Port:** http://localhost:8000  
**Health Check:** `curl http://localhost:8000/health`

---

## Frontend Setup & Run

### Install Dependencies
```bash
cd frontend
npm install
```

### Start Dev Server
```bash
cd frontend
npm run dev
```

**Expected Output:**
```
➜  Local:   http://localhost:3000
➜  press h to show help
VITE v5.4 ready in 234 ms
```

**Port:** http://localhost:3000

---

## Access Application

### Browser
1. Open: **http://localhost:3000**
2. Login with credentials:
   - Username: `admin`
   - Password: `admin` (or register new account)

### Verify Connectivity
```bash
# Check backend is responding
curl http://localhost:8000/health

# Expected response:
# {"status":"running","cycle_count":123,"uptime_seconds":45.2,...}
```

---

## Quick Troubleshooting

### Dashboard Shows No Data

**Checklist:**

- [ ] **Backend running?**
  ```bash
  curl http://localhost:8000/health
  # Should return {"status":"running", ...}
  # If fails: Port might be in use
  kill -9 $(lsof -t -i:8000)  # Free port 8000
  ```

- [ ] **Frontend connected to backend?**
  - Open DevTools (F12) → Network tab
  - Refresh page
  - Look for requests to `localhost:8000/api`
  - Should see responses like `200 OK`

- [ ] **WebSocket connected?**
  - DevTools → Console tab
  - Should show: `[WS] Connected to ws://localhost:8000/ws/radar-stream`
  - If not: WebSocket connection failed

- [ ] **Models loaded?**
  ```bash
  curl http://localhost:8000/api/controller/status
  # Check: "initialization_status": "READY"
  # If "PARTIAL" or "FAILED": Check backend logs for errors
  ```

- [ ] **Radar data generating?**
  - Dashboard → Real-Time Analytics tab
  - Should see live radar frames updating every 0.5 seconds
  - If empty: Backend radar service not running

- [ ] **Port conflicts?**
  ```bash
  # Check if ports are in use
  lsof -i :8000  # Backend
  lsof -i :3000  # Frontend
  
  # If in use, kill process:
  kill -9 <PID>
  ```

- [ ] **Clear browser cache?**
  - Hard refresh: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
  - Or clear cache in DevTools → Application → Clear Storage

- [ ] **Check browser console for errors?**
  - DevTools → Console tab
  - Look for red error messages
  - Copy error and check logs

---

## Common Issues & Fixes

### Issue: "Address already in use" on port 8000
```bash
# Linux/Mac:
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Windows PowerShell:
Get-Process | Where-Object {$_.Name -eq "python"} | Stop-Process -Force
```

### Issue: Frontend shows "Cannot GET /api/..."
```
Problem: API calls returning 404
Solution: Backend not running or API routes not loaded
Fix:
1. Verify backend is on http://localhost:8000
2. Check backend terminal for error messages
3. Restart backend: Ctrl+C then rerun uvicorn command
```

### Issue: WebSocket connection fails
```
Problem: Console shows "Failed to connect WebSocket"
Solution: Backend WebSocket endpoint not listening
Fix:
1. Check backend logs for errors starting WebSocket
2. Verify firewall not blocking port 8000
3. Restart backend with: python -m uvicorn app.main:app --reload --port 8000
```

### Issue: "npm: command not found"
```bash
# Install Node.js from https://nodejs.org/
# Verify installation:
node --version   # Should show v18+
npm --version    # Should show 8+
```

### Issue: "No module named 'backend'"
```bash
# Make sure running from correct directory
cd "c:\Users\nikhil\Desktop\MCA Project\Aegis-Cognitive-Defense-Platform"

# Then:
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

---

## Development Workflow

### During Development

**Terminal 1 - Backend:**
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
# Code changes auto-reload (Ctrl+C to stop)
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# File changes auto-reload via HMR
```

### Making Backend Changes
1. Edit Python file in `backend/app/`
2. Backend auto-reloads (watch for "Application reloaded")
3. Frontend auto-reconnects

### Making Frontend Changes
1. Edit React component in `frontend/src/`
2. Frontend auto-updates in browser (HMR)
3. No refresh needed

### Testing API Endpoints
```bash
# Get controller status
curl http://localhost:8000/api/controller/status

# Get health
curl http://localhost:8000/health

# Get metrics (if available)
curl http://localhost:8000/api/metrics/report
```

---

## Performance Notes

- **Backend CPU:** ~5-10% (idle state)
- **Frontend Dev Server:** ~50-100MB RAM
- **Database:** In-memory (SQLite optional)
- **WebSocket:** Heartbeat every 30 seconds

---

## Stopping & Restarting

### Stop Backend
```bash
# Press Ctrl+C in backend terminal
```

### Stop Frontend
```bash
# Press Ctrl+C in frontend terminal
```

### Full Restart
```bash
# Terminal 1
cd backend
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2 (new)
cd frontend
npm run dev
```

---

## Next Steps

- [ ] Started backend on port 8000
- [ ] Started frontend on port 3000
- [ ] Opened http://localhost:3000 in browser
- [ ] Logged in with credentials
- [ ] Verified real-time data displays
- [ ] Checked browser console (F12) for errors
- [ ] Confirmed WebSocket connected

**Success:** Dashboard shows real-time radar data updating every 0.5 seconds

---

## For More Info

- **Backend Architecture:** See `backend/BACKEND_ARCHITECTURE.md`
- **Frontend Structure:** See `frontend/REFACTORED_ARCHITECTURE.md`
- **API Documentation:** Check `backend/app/api/routes/` for endpoint details
- **Environment Config:** See `.env.development` and `.env.production`

---

## Quick Commands Reference

```bash
# Backend start (from project root)
cd backend && python -m uvicorn app.main:app --reload --port 8000

# Frontend start (from project root)
cd frontend && npm install && npm run dev

# Check backend health
curl http://localhost:8000/health

# Check frontend (browser)
http://localhost:3000

# View backend logs (during runtime)
# Check terminal window running uvicorn

# View frontend logs (during runtime)
# Check DevTools Console (F12)
```

**Total setup time:** ~5 minutes  
**No Docker required**  
**No deployment needed**
