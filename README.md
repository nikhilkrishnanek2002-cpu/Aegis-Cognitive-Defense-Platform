# 🛰️ Aegis Cognitive Defense Platform

An AI-enabled photonic radar system with real-time cognitive electronic warfare capabilities.

## Architecture

```
FastAPI Backend  (port 8000)  ←── Python radar pipeline
React Frontend   (port 3000)  ←── Live WebSocket dashboard
```

## Quick Start

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
pip install fastapi uvicorn "python-jose[cryptography]" "passlib[bcrypt]" python-multipart
```

### 2. Install Node.js frontend dependencies (first time only)
```bash
cd frontend && npm install && cd ..
```

### 3. Launch everything
```bash
python launcher.py
```

This starts both:
- **FastAPI API** → [http://localhost:8000](http://localhost:8000)
- **React Dashboard** → [http://localhost:3000](http://localhost:3000)
- **API Docs (Swagger)** → [http://localhost:8000/docs](http://localhost:8000/docs)

## Default Login

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | Admin |

## Project Structure

```
api/                  FastAPI backend
  main.py             App entry point + CORS + WebSocket
  routes/             REST endpoints (auth, radar, tracks, ew, admin, metrics)
  websocket.py        Live radar broadcast loop
  auth_utils.py       JWT authentication
  state.py            Shared tracker/cognitive/EW singletons

frontend/             React + TypeScript frontend (Vite)
  src/
    pages/            LoginPage, DashboardPage
    components/tabs/  6 tab panels (Analytics, XAI, Photonic, Metrics, Logs, Admin)
    store/            Zustand auth + radar state stores
    api/              Axios REST client + WebSocket hook

src/                  Core radar processing modules (unchanged)
  signal_generator.py Synthetic radar signal with realistic physics
  detection.py        CA-CFAR and OS-CFAR detection
  tracker.py          Kalman multi-target tracker (Hungarian matching)
  ew_defense.py       Electronic warfare threat analysis
  cognitive_controller.py  Reinforcement learning waveform adaptation
  xai_pytorch.py      Grad-CAM explainability
  model_pytorch.py    Multi-input CNN for target classification
  feature_extractor.py     Range-Doppler + spectrogram features

tests/                167 unit tests (pytest)
config.yaml           System-wide configuration
```

## Running Tests
```bash
pytest tests/ -q
```

## Training the AI Model
```bash
python -m src.train_pytorch
```

## API Reference
Full Swagger UI available at [http://localhost:8000/docs](http://localhost:8000/docs) when the server is running.

| Endpoint | Method | Description |
|---|---|---|
| `/api/auth/login` | POST | Get JWT token |
| `/api/auth/register` | POST | Register new operator |
| `/api/radar/scan` | POST | Run full radar pipeline |
| `/api/tracks` | GET | Get current Kalman tracks |
| `/api/tracks/reset` | DELETE | Reset tracker |
| `/api/admin/users` | GET | List users (admin) |
| `/api/admin/health` | GET | System health (admin) |
| `/api/metrics/report` | GET | Classification metrics JSON |
| `/ws/stream` | WebSocket | Live radar frame stream |
