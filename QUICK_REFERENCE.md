# ⚡ QUICK REFERENCE CARD

## 🚀 START HERE (Copy & Paste)

```bash
cd "c:\Users\nikhil\Desktop\MCA Project\Aegis-Cognitive-Defense-Platform"
python dev_local.py
```

Then open: **http://localhost:3000**
Login: `admin` / `admin123`

---

## 📍 FILE LOCATIONS

| What | Where | Purpose |
|------|-------|---------|
| **Backend Entry** | `main_api.py` | Start backend API |
| **Local Launcher** | `dev_local.py` | Start everything |
| **Backend Code** | `backend/app/` | Actual API code |
| **Frontend Code** | `frontend/src/` | React app |
| **Config** | `config.yaml` | Settings |
| **Dependencies** | `requirements.txt` | Python packages |

---

## 💻 COMMANDS

### Start Everything
```bash
python dev_local.py
```

### Start Backend Only
```bash
pip install -r requirements.txt
python -m uvicorn main_api:app --reload --port 8000
```

### Start Frontend Only
```bash
cd frontend
npm install  # first time only
npm run dev
```

### Test API
```bash
curl http://localhost:8000/health
open http://localhost:8000/docs
```

### Test WebSocket
```bash
npm install -g wscat
wscat -c ws://localhost:8000/ws/stream
```

---

## 🌐 URLS

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:3000 |
| **API Docs** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |
| **WebSocket** | ws://localhost:8000/ws/stream |

---

## 📚 DOCUMENTATION

| File | Read For |
|------|----------|
| `00_START_HERE.md` | This summary |
| `EXECUTIVE_SUMMARY.md` | Full overview |
| `LOCAL_DEV_GUIDE.md` | Detailed guide + troubleshooting |
| `REFACTORING_MIGRATION_GUIDE.md` | What changed (technical) |
| `CLEANUP_CHECKLIST.md` | How to delete old files |

---

## 🔑 LOGIN

**Username:** `admin`
**Password:** `admin123`

---

## 🗑️ FILES TO DELETE (When Ready)

```bash
rm -rf api/
rm frontend/src/main.jsx frontend/src/App.jsx
rm frontend/src/pages/*.jsx
rm launcher.py start.py
```

See `CLEANUP_CHECKLIST.md` for safe step-by-step deletion.

---

## 🐛 Common Issues

| Issue | Fix |
|-------|-----|
| **ModuleNotFoundError** | Run from project root |
| **Port 8000 in use** | `taskkill /PID <PID> /F` |
| **CORS errors** | Check `vite.config.ts` proxy |
| **npm install fails** | `rm -rf node_modules && npm install` |
| **WebSocket fails** | Ensure backend running on :8000 |

More solutions: `LOCAL_DEV_GUIDE.md` → Troubleshooting

---

## 📊 Architecture

```
Browser (localhost:3000)
        ↓
React Frontend (Vite)
        ↓ /api proxy
FastAPI Backend (localhost:8000)
        ↓
Services (Radar, Detection, Tracking, Threat, EW)
```

---

## ✅ VERIFICATION

```bash
# All should return success
python -m uvicorn main_api:app --reload --port 8000 &
sleep 3
curl http://localhost:8000/health
# Should show: {"status": "ok", ...}
```

---

## 📖 LEARNING PATH

1. Read: `00_START_HERE.md` (this file)
2. Start: `python dev_local.py`
3. Explore: http://localhost:3000 & http://localhost:8000/docs
4. Learn: `EXECUTIVE_SUMMARY.md` → `LOCAL_DEV_GUIDE.md`
5. Code: Edit `backend/app/` or `frontend/src/`
6. Clean: Follow `CLEANUP_CHECKLIST.md` when ready

---

**🎉 You're ready! Start with: `python dev_local.py`**
