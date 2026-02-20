"""Metrics and analytics endpoints."""

from fastapi import APIRouter
from datetime import datetime, timedelta
from app.services.radar_service import get_radar_service
from app.services.detection_service import get_detection_service
from app.services.tracking_service import get_tracking_service
from app.services.threat_service import get_threat_service
from app.services.ew_service import get_ew_service
from app.engine.controller import _controller
from app.core.performance import timer
from app.api.websocket.radar_ws_optimized import get_websocket_stats

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/radar")
async def get_radar_metrics():
    """Get radar performance metrics."""
    radar_svc = get_radar_service()
    quality = await radar_svc.get_signal_quality()
    
    return {
        "scan_count": radar_svc.scan_count,
        "last_scan": radar_svc.last_scan_time.isoformat() if radar_svc.last_scan_time else None,
        "signal_quality": quality
    }


@router.get("/detection")
async def get_detection_metrics():
    """Get detection model metrics."""
    detection_svc = get_detection_service()
    model_info = await detection_svc.get_model_info()
    
    return model_info


@router.get("/tracking")
async def get_tracking_metrics():
    """Get tracking metrics."""
    tracking_svc = get_tracking_service()
    tracks = await tracking_svc.get_active_tracks()
    
    return {
        "active_tracks": len(tracks),
        "total_tracks_history": sum(1 for t in tracking_svc.tracks.values()),
        "tracks": [t.dict() for t in tracks]
    }


@router.get("/threats")
async def get_threat_metrics():
    """Get threat assessment metrics."""
    threat_svc = get_threat_service()
    critical = await threat_svc.get_critical_threats()
    
    return {
        "critical_threats": len(critical),
        "threat_history_count": len(threat_svc.threat_history),
        "critical_threat_ids": [t.track_id for t in critical]
    }


@router.get("/ew")
async def get_ew_metrics():
    """Get EW status metrics."""
    ew_svc = get_ew_service()
    status = await ew_svc.get_ew_status()
    
    return status


@router.get("/pipeline")
async def get_pipeline_metrics():
    """Get pipeline execution metrics."""
    if _controller:
        return await _controller.get_status()
    return {"error": "Controller not initialized"}


@router.get("/system")
async def get_system_metrics():
    """Get overall system metrics."""
    if not _controller:
        return {"error": "System not ready"}
    
    pipeline_status = await _controller.get_status()
    
    # Aggregate metrics from all services
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline": pipeline_status,
        "uptime_seconds": pipeline_status.get("uptime_seconds", 0)
    }


@router.get("/performance")
async def get_performance_metrics():
    """Get performance metrics for all pipeline stages."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "stages": timer.get_all_stats(),
        "websocket": get_websocket_stats()
    }


@router.get("/performance/summary")
async def get_performance_summary():
    """Get high-level performance summary."""
    stats = timer.get_all_stats()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "radar_scan_ms": stats["radar_scan"]["latest"],
        "detection_ms": stats["detection"]["latest"],
        "tracking_ms": stats["tracking"]["latest"],
        "threat_assessment_ms": stats["threat_assessment"]["latest"],
        "ew_response_ms": stats["ew_response"]["latest"],
        "websocket_send_ms": stats["websocket_send"]["latest"],
        "total_cycle_ms": stats["total_cycle"]["latest"],
        "avg_cycle_ms": stats["total_cycle"]["avg"]
    }


@router.get("/health/cpu-memory")
async def get_cpu_memory():
    """Get CPU and memory usage."""
    import psutil
    
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        
        return {
            "cpu_percent": cpu_percent,
            "memory_mb": memory_info.rss / (1024 * 1024),
            "memory_percent": process.memory_percent(),
            "num_threads": process.num_threads()
        }
    except Exception as e:
        return {"error": str(e)}
