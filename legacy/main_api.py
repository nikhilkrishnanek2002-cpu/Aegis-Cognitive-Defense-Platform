"""
AEGIS Cognitive Defense Platform - FastAPI Backend Entry Point
Single production-ready entry point. Run from project root with:
    uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path so backend imports work
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add backend directory for internal app imports
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ─── Core Imports ──────────────────────────────────────────────────────────
try:
    from backend.app.core.config import get_config, Config
    from backend.app.core.logging import pipeline_logger
    
    # Import services (singletons)
    from backend.app.services.radar_service import get_radar_service
    from backend.app.services.detection_service import get_detection_service
    from backend.app.services.tracking_service import get_tracking_service
    from backend.app.services.threat_service import get_threat_service
    from backend.app.services.ew_service import get_ew_service
    
    # Import engine
    from backend.app.engine.controller import get_controller
    from backend.app.engine.event_bus import event_bus, Events
    
    # Import routes
    from backend.app.api.routes import health, metrics, radar, threats, auth
    from backend.app.api.websocket.radar_ws_optimized import ws_endpoint
    
except ImportError as e:
    print(f"❌ Failed to import backend modules: {e}")
    print(f"   BACKEND_DIR: {BACKEND_DIR}")
    print(f"   sys.path: {sys.path[:3]}")
    sys.exit(1)

# ─── Global State ──────────────────────────────────────────────────────────
config: Config = None
controller = None


# ─── Lifespan: Startup & Shutdown ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services, start pipeline controller, and cleanup on shutdown."""
    
    # STARTUP
    pipeline_logger.info("=" * 70)
    pipeline_logger.info("🚀 AEGIS COGNITIVE DEFENSE PLATFORM STARTUP")
    pipeline_logger.info("=" * 70)
    
    try:
        global config, controller
        config = get_config()
        
        # Initialize services (singletons)
        pipeline_logger.info("Initializing services...")
        radar_svc = get_radar_service()
        detection_svc = get_detection_service()
        tracking_svc = get_tracking_service()
        threat_svc = get_threat_service()
        ew_svc = get_ew_service()
        pipeline_logger.info("✓ All services initialized")
        
        # Create and start controller
        pipeline_logger.info("Starting pipeline controller...")
        controller = get_controller(
            radar_svc,
            detection_svc,
            tracking_svc,
            threat_svc,
            ew_svc
        )
        await controller.start()
        pipeline_logger.info("✓ Pipeline controller started")
        pipeline_logger.info(f"✓ Scan interval: {config.radar_scan_interval}s")
        
        # Brief delay to let pipeline settle
        await asyncio.sleep(0.5)
        
        pipeline_logger.info("=" * 70)
        pipeline_logger.info("✅ AEGIS READY - MONITORING ACTIVE")
        pipeline_logger.info("=" * 70)
        
    except Exception as e:
        pipeline_logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise
    
    yield  # App runs here
    
    # SHUTDOWN
    pipeline_logger.info("=" * 70)
    pipeline_logger.info("🛑 AEGIS SHUTDOWN")
    pipeline_logger.info("=" * 70)
    
    if controller:
        try:
            await controller.stop()
            pipeline_logger.info("✓ Pipeline stopped")
        except Exception as e:
            pipeline_logger.error(f"Error stopping pipeline: {e}")
    
    pipeline_logger.info("=" * 70)


# ─── FastAPI App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Aegis Cognitive Defense API",
    description="Real-time AI-enabled photonic radar backend",
    version="2.0.0",
    lifespan=lifespan
)

# ─── CORS Middleware ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",   # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:8000",   # Self (for testing)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register Routers ──────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(radar.router)
app.include_router(threats.router)
app.include_router(metrics.router)

# ─── WebSocket Endpoint ───────────────────────────────────────────────────
from fastapi import WebSocket

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Live radar stream WebSocket endpoint."""
    await ws_endpoint(websocket)


# ─── Health Check ─────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Aegis Cognitive Defense API",
        "version": "2.0.0"
    }


# ─── Run Server ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    
    cfg = config or get_config()
    uvicorn.run(
        "main_api:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=True,
        log_level="info"
    )
