"""
FastAPI Application Entry Point — Aegis Cognitive Defense Platform API
Provides REST endpoints and a live WebSocket stream for the React frontend.
"""
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket

from api.routes import auth, radar, tracks, ew, admin, metrics, visualizations
from api.websocket import ws_endpoint, radar_broadcast_loop
from src.db import init_db, ensure_admin_exists
from src.logger import init_logging, log_event
from src.config import get_config

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Aegis Cognitive Defense API",
    description="Real-time AI-enabled photonic radar backend",
    version="2.0.0",
)

# ─── CORS: allow React dev server (port 3000) and any local IP ────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",   # Vite default
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register Routers ─────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(radar.router)
app.include_router(tracks.router)
app.include_router(ew.router)
app.include_router(admin.router)
app.include_router(metrics.router)
app.include_router(visualizations.router)


# ─── WebSocket Live Stream ────────────────────────────────────────────────────
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await ws_endpoint(websocket)


# ─── Startup / Shutdown ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    cfg = get_config()
    init_logging(cfg)
    init_db()
    ensure_admin_exists()
    log_event("Aegis FastAPI backend started", level="info")
    # Start the background radar broadcast loop
    asyncio.create_task(radar_broadcast_loop())


@app.on_event("shutdown")
async def shutdown():
    log_event("Aegis FastAPI backend shutting down", level="info")


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "Aegis Cognitive Defense API", "version": "2.0.0"}
