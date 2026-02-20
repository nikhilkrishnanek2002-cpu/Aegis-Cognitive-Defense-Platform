"""Radar endpoints."""

from fastapi import APIRouter
from app.services.radar_service import get_radar_service
from app.services.tracking_service import get_tracking_service
from app.engine.controller import _controller

router = APIRouter(prefix="/api/radar", tags=["radar"])


@router.get("/status")
async def get_radar_status():
    """Get current radar status."""
    radar_svc = get_radar_service()
    
    return {
        "operational": True,
        "scan_count": radar_svc.scan_count,
        "last_scan": radar_svc.last_scan_time.isoformat() if radar_svc.last_scan_time else None,
        "connected": True
    }


@router.get("/targets")
async def get_radar_targets():
    """Get current radar targets."""
    if not _controller:
        return {"error": "Pipeline not started", "targets": []}
    
    targets = _controller.pipeline.last_targets
    return {
        "count": len(targets),
        "targets": [t.dict() for t in targets]
    }


@router.get("/tracks")
async def get_tracked_targets():
    """Get currently tracked targets."""
    tracking_svc = get_tracking_service()
    tracks = await tracking_svc.get_active_tracks()
    
    return {
        "count": len(tracks),
        "tracks": [t.dict() for t in tracks]
    }


@router.post("/scan")
async def trigger_scan():
    """Manually trigger a radar scan."""
    radar_svc = get_radar_service()
    scan = await radar_svc.scan()
    
    return {
        "success": True,
        "scan": scan.dict()
    }


@router.get("/signal-quality")
async def get_signal_quality():
    """Get radar signal quality metrics."""
    radar_svc = get_radar_service()
    quality = await radar_svc.get_signal_quality()
    
    return {
        "quality": quality,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    }
