# ✅ CLEANUP CHECKLIST - STEP BY STEP

Complete this checklist to finalize the refactoring and remove all redundant files.

---

## 📋 PRE-CLEANUP VERIFICATION

Before deleting anything, verify the new setup is working:

- [ ] `python dev_local.py` starts without errors
- [ ] Backend logs show "AEGIS READY - MONITORING ACTIVE"
- [ ] http://localhost:8000/health returns `{"status": "ok"}`
- [ ] http://localhost:3000 loads (login page)
- [ ] Can login with `admin / admin123`
- [ ] Dashboard displays without console errors

**If all checked:** Safe to proceed with cleanup. Otherwise, fix issues first.

---

## 🗑️ PHASE 1: DELETE OLD BACKEND (SAFE)

The `/api/` directory has been completely replaced by `backend/app/` reachable via `main_api.py`.

### Delete entire `/api/` directory
```bash
rm -rf api/
```

Or manually delete these files:
```
api/__init__.py
api/auth_utils.py
api/main.py                    ← OLD API entry point
api/state.py
api/websocket.py               ← OLD WebSocket handler

api/routes/                    ← OLD route handlers
  __init__.py
  admin.py
  auth.py
  ew.py
  metrics.py
  radar.py
  tracks.py
  visualizations.py

api/__pycache__/               ← Cache, will regenerate
```

### Verify
```bash
# Should NOT exist anymore
ls api/          # Should fail

# Should exist
ls backend/app/api/routes/     # Should show: auth.py, radar.py, threats.py, etc.
ls main_api.py                 # New entry point
```

---

## 🗑️ PHASE 2: DELETE OLD FRONTEND JSX

The frontend now uses **TypeScript only** (`.tsx`). JavaScript (`.jsx`) versions are duplicates and cause conflicts.

### Delete JSX entry points
```bash
rm frontend/src/main.jsx
rm frontend/src/App.jsx
```

### Delete JSX pages
```bash
rm frontend/src/pages/Dashboard.jsx
rm frontend/src/pages/EWControl.jsx
rm frontend/src/pages/LoginPage.jsx
rm frontend/src/pages/ModelMonitor.jsx
rm frontend/src/pages/RadarLive.jsx
rm frontend/src/pages/Settings.jsx
rm frontend/src/pages/ThreatAnalysis.jsx
```

### Manual deletion (Windows cmd)
```cmd
cd frontend\src
del main.jsx App.jsx
cd pages
del Dashboard.jsx EWControl.jsx LoginPage.jsx ModelMonitor.jsx RadarLive.jsx Settings.jsx ThreatAnalysis.jsx
```

### Verify
```bash
# Should NOT exist anymore
ls frontend/src/*.jsx          # Should fail or show nothing

# Should exist
ls frontend/src/main.tsx
ls frontend/src/App.tsx
ls frontend/src/pages/*.tsx    # All TypeScript pages
```

**Check:** index.html should point to `src/main.tsx` (already correct)

---

## 🗑️ PHASE 3: DELETE OLD LAUNCHERS (OPTIONAL)

These old launcher scripts are now replaced by `dev_local.py`. Delete them if you want a clean workspace:

```bash
rm launcher.py
rm start.py
rm app_console.py              # Streamlit console app (not used in local dev)
```

**Optional because:** They still work, but are no longer needed. Keep them in git history, just remove from working directory.

### Manual deletion (Windows)
```cmd
del launcher.py
del start.py
del app_console.py
```

---

## 🗑️ PHASE 4: CLEAN PYTHON CACHE (RECOMMENDED)

Remove Python cache files to prevent stale imports:

```bash
# Remove all __pycache__ directories
find . -name __pycache__ -type d -exec rm -rf {} +

# Remove all *.pyc files
find . -name "*.pyc" -delete

# Remove .pytest_cache
rm -rf .pytest_cache

# Remove any old eggs
rm -rf *.egg-info
```

Or on Windows:
```cmd
# Use PowerShell
Get-ChildItem -Path . -Recurse -Filter __pycache__ -Directory | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Recurse -Filter *.pyc | Remove-Item -Force
```

---

## ✅ FINAL VERIFICATION

After cleanup, verify the project structure:

### Backend should have
```bash
✅ main_api.py                 # New entry point
✅ backend/app/core/
✅ backend/app/services/
✅ backend/app/engine/
✅ backend/app/api/routes/
✅ backend/app/api/websocket/
```

### Frontend should have
```bash
✅ frontend/src/main.tsx       # TypeScript entry
✅ frontend/src/App.tsx        # TypeScript app
✅ frontend/src/pages/         # All .tsx (NO JSX)
✅ frontend/src/config/apiConfig.ts
```

### Docs should have
```bash
✅ LOCAL_DEV_GUIDE.md
✅ REFACTORING_MIGRATION_GUIDE.md
✅ EXECUTIVE_SUMMARY.md
✅ CLEANUP_CHECKLIST.md (this file)
```

### Should NOT have
```bash
❌ api/                        # Deleted
❌ frontend/src/*.jsx          # Deleted
❌ launcher.py                 # Deleted (optional)
❌ start.py                    # Deleted (optional)
```

---

## 🧪 FINAL TEST

After cleanup, run full test:

```bash
# 1. Start system
python dev_local.py

# 2. In another terminal, test API
curl http://localhost:8000/health

# 3. Open dashboard
open http://localhost:3000

# 4. Verify all features work
# - Login with admin/admin123
# - Navigate to Radar page
# - Check threats display
# - Verify WebSocket connection (DevTools → Network → WS)
# - Check API calls work (DevTools → Network → XHR)
```

✅ Everything working? **Cleanup successful!**

---

## 📊 BEFORE & AFTER

### BEFORE (Confusing Structure)
```
.
├── api/                       ← Confusing: is this the API backend?
│   ├── main.py               ← Which entry point to use?
│   ├── routes/
│   └── websocket.py
├── backend/                  ← Or is THIS the backend?
│   └── app/
│       └── main.py           ← Two different main.py files!
├── launcher.py               ← Start this? Or start.py?
├── start.py
├── launcher.py
└── frontend/src/
    ├── main.jsx              ← Which main? JSX or TSX?
    ├── main.tsx
    ├── App.jsx
    ├── App.tsx
    └── pages/
        ├── Dashboard.jsx     ← Duplicate components everywhere
        ├── DashboardPage.tsx
        └── ...
```

### AFTER (Clean Structure)
```
.
├── main_api.py               ← Clear: This is THE entry point
├── dev_local.py              ← Clear: This launches everything
├── backend/
│   └── app/                  ← Official backend stays here
│       ├── core/
│       ├── services/
│       ├── engine/
│       └── api/
├── api/                      ← ❌ DELETED (was duplicate)
├── frontend/src/
│   ├── main.tsx              ← Clear: TypeScript only
│   ├── App.tsx
│   └── pages/                ← All .tsx (NO JSX)
├── launcher.py               ← ❌ OPTIONAL DELETE
├── start.py                  ← ❌ OPTIONAL DELETE
└── docs/
    ├── EXECUTIVE_SUMMARY.md
    ├── LOCAL_DEV_GUIDE.md
    ├── REFACTORING_MIGRATION_GUIDE.md
    └── CLEANUP_CHECKLIST.md   ← This file
```

---

## 🎯 CLEANUP SUCCESS CRITERIA

You're done when:

- [ ] ✅ `api/` directory deleted
- [ ] ✅ All `.jsx` files deleted (except in legacy archives)
- [ ] ✅ `frontend/src/main.tsx` exists and is the entry point
- [ ] ✅ All TypeScript pages (`.tsx`) exist
- [ ] ✅ `main_api.py` is the only backend entry point
- [ ] ✅ `dev_local.py` starts all servers successfully
- [ ] ✅ Dashboard works at http://localhost:3000
- [ ] ✅ API works at http://localhost:8000/docs
- [ ] ✅ WebSocket connection works (real-time data flows)
- [ ] ✅ No console errors in browser DevTools
- [ ] ✅ No import errors in Python console

**All checked? ✅ CLEANUP COMPLETE!**

---

## 🔄 RECOVERY (IF NEEDED)

If cleanup causes issues:

```bash
# Restore from git
git restore api/
git restore frontend/src/main.jsx frontend/src/App.jsx
git restore launcher.py start.py

# Or revert entire commit
git revert HEAD
```

Then fix the issue and try again.

---

## 📝 NOTES

- **`src/` directory:** Keep for reference. It contains original implementation.
- **Config files:** Keep (config.yaml, requirements.txt, etc.)
- **Test/docs directories:** Keep (tests/, docs/, experiments/, etc.)
- **Results/outputs:** Keep (results/, outputs/, etc.)

Only delete what's listed above as redundant.

---

## ✅ SIGN-OFF

Date cleanup completed: _______________
Person: _____________________________
Status: ☐ VERIFIED ☐ NOT VERIFIED

Use checklist:
```
- [ ] Phase 1 complete (api/ deleted)
- [ ] Phase 2 complete (JSX deleted)  
- [ ] Phase 3 complete (launchers deleted - optional)
- [ ] Phase 4 complete (cache cleaned)
- [ ] Final verification passed
- [ ] Test passed
```

**When all boxes checked: ✅ Ready to commit changes!**

```bash
git status           # Review changes
git add -A
git commit -m "Refactor: Clean up duplicate backends and JSX/TSX conflicts"
git log --oneline -n 5  # Verify commit
```

---

**Status: Ready for Production Local Development** 🚀
