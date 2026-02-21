# Aegis Cognitive Defense Platform - Quick Start Guide

## 🚀 One-Command Launch

```bash
python launcher.py
```

This will automatically:
1. ✅ Check and install required Python dependencies (FastAPI, uvicorn, etc.)
2. ✅ Check if Node.js packages are installed (frontend only)
3. ✅ Start FastAPI backend on http://localhost:8000
4. ✅ Start React frontend on http://localhost:3000 (if Node.js available)
5. ✅ Open your browser automatically

## 📋 Requirements

### Core Requirements (API Server)
- Python 3.8+
- FastAPI, uvicorn, numpy, scipy
- python-jose, passlib, python-multipart

These are automatically installed by the launcher if missing.

### Optional - AI Features
- PyTorch (for AI model inference)
- Install with: `pip install torch`

Without PyTorch, the system runs in **radar-only mode** with all detection features available, but AI classification will be disabled.

### Optional - Frontend Dashboard  
- Node.js 18+ and npm
- React dependencies (auto-installed on first run)

Without Node.js, the system runs in **API-only mode** accessible at http://localhost:8000/docs

## 🔧 Manual Installation

If the launcher fails to install dependencies due to disk space or permissions:

```bash
# Install core Python dependencies
pip install fastapi uvicorn numpy scipy python-jose[cryptography] passlib[bcrypt] python-multipart

# Optional: Install PyTorch for AI features
pip install torch

# Optional: Install frontend (requires Node.js 18+)
cd frontend
npm install
```

## 🎯 Running in Different Modes

### API Only (No Frontend)
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Access at: http://localhost:8000/docs

### Frontend Only (API must be running separately)
```bash
cd frontend
npm run dev
```

### Training Scripts (Requires PyTorch)
```bash
# Basic training
python main.py

# Experiment runner with config
python run_experiment.py --config radar_ai_experiment.yaml

# Streamlit app
streamlit run app.py
```

## 🐛 Troubleshooting

### "Disk quota exceeded"
- The launcher tried to install packages but ran out of disk space
- Manually install core dependencies: `pip install fastapi uvicorn numpy scipy python-jose[cryptography] passlib[bcrypt] python-multipart`

### "npm not found"
- Node.js is not installed - the system will run in API-only mode
- Install Node.js 18+ from https://nodejs.org if you want the dashboard

### "PyTorch not installed"
- AI features will be disabled
- Install with: `pip install torch`
- The radar detection still works without PyTorch

### Port already in use
- Kill existing processes: `lsof -ti:8000 | xargs kill` or `lsof -ti:3000 | xargs kill`
- Or change ports in launcher.py

## 📊 Default Credentials

- **Username**: admin
- **Password**: admin123

Change these in the admin panel after first login.

## 🔍 Logs

- API Server: `api_server.log`
- React Frontend: `react_dev.log`
- System: `results/system.log`

## ⚡ Quick Commands

```bash
# Check if API is running
curl http://localhost:8000/health

# Check dependencies
python -c "import fastapi, uvicorn, numpy, scipy; print('✅ Core deps OK')"

# Test PyTorch
python -c "import torch; print('✅ PyTorch available')"

# Stop all servers
pkill -f uvicorn
pkill -f "npm run dev"
```

