"""
Aegis Cognitive Defense Platform - FastAPI Backend
Event-driven real-time AI radar system with async pipeline architecture
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_config
from app.core.logging import pipeline_logger

# Import services
from app.services.radar_service import get_radar_service
from app.services.detection_service import get_detection_service
from app.services.tracking_service import get_tracking_service
from app.services.threat_service import get_threat_service
from app.services.ew_service import get_ew_service

# Import engine
from app.engine.controller import get_controller
from app.engine.event_bus import event_bus, Events

# Import routes
from app.api.routes import health, metrics, radar, threats, auth
from app.api.websocket.radar_ws_optimized import ws_endpoint


# ─── Global state ──────────────────────────────────────────────────────────
config = get_config()
controller = None


# ─── Startup / Shutdown ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    
    # STARTUP
    pipeline_logger.info("=" * 60)
    pipeline_logger.info("AEGIS COGNITIVE DEFENSE PLATFORM STARTUP")
    pipeline_logger.info("=" * 60)
    
    try:
        # Initialize services (singletons)
        radar_svc = get_radar_service()
        detection_svc = get_detection_service()
        tracking_svc = get_tracking_service()
        threat_svc = get_threat_service()
        ew_svc = get_ew_service()
        
        pipeline_logger.info("✓ All services initialized")
        
        # Create and start controller
        global controller
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
        await asyncio.sleep(1)
        
        pipeline_logger.info("=" * 60)
        pipeline_logger.info("AEGIS READY - MONITORING ACTIVE")
        pipeline_logger.info("=" * 60)
        
    except Exception as e:
        pipeline_logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # SHUTDOWN
    pipeline_logger.info("=" * 60)
    pipeline_logger.info("AEGIS SHUTDOWN INITIATED")
    pipeline_logger.info("=" * 60)
    
    if controller:
        await controller.stop()
        pipeline_logger.info("✓ Pipeline controller stopped")
    
    pipeline_logger.info("=" * 60)


# ─── FastAPI App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Aegis Cognitive Defense API",
    description="Real-time AI-enabled photonic radar backend with event-driven architecture",
    version="2.0.0",
    lifespan=lifespan
)


# ─── CORS Middleware ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*"  # Development: allow all - restrict in production
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


# ─── WebSocket Endpoint ────────────────────────────────────────────────────
@app.websocket("/ws/radar-stream")
async def websocket_radar_stream(websocket):
    """
    WebSocket endpoint for real-time radar frame and threat streaming.
    
    Clients subscribe to:
    - radar_frame: Live radar frames with targets
    - threats: Current threat assessment
    - system_status: System health and metrics
    """
    await ws_endpoint(websocket)


# ─── Root Endpoint ────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Aegis Cognitive Defense API",
        "version": "2.0.0",
        "documentation": "/docs",
        "openapi": "/openapi.json"
    }


# ─── Controller Status (for debugging) ──────────────────────────────────
@app.get("/api/controller/status")
async def get_controller_status():
    """Get controller and pipeline status."""
    if controller:
        return await controller.get_status()
    return {"error": "Controller not initialized"}


@app.post("/api/controller/restart")
async def restart_controller():
    """Restart the pipeline controller (debugging)."""
    global controller
    
    if controller:
        await controller.stop()
    
    try:
        controller = get_controller(
            get_radar_service(),
            get_detection_service(),
            get_tracking_service(),
            get_threat_service(),
            get_ew_service()
        )
        await controller.start()
        return {"success": True, "message": "Controller restarted"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Run with: uvicorn app.main:app --reload
    import uvicorn
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level="info"
    )
