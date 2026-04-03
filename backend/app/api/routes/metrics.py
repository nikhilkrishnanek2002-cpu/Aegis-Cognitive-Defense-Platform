"""Metrics and analytics endpoints."""

from fastapi import APIRouter
from datetime import datetime, timedelta
from app.services.radar_service import get_radar_service
from app.services.detection_service import get_detection_service
from app.services.tracking_service import get_tracking_service
from app.services.threat_service import get_threat_service
from app.services.ew_service import get_ew_service
from app.services.metrics_service import get_metrics_collector
from app.core.metrics_store import get_metrics_store
from app.engine import _controller
from app.core.performance import timer
from app.api.websocket.radar_ws_optimized import get_websocket_stats
import psutil

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/radar")
async def get_radar_metrics():
    """Get radar performance metrics."""
    radar_svc = get_radar_service()
    quality = await radar_svc.get_signal_quality()
    
    return {
        "scan_count": radar_svc.scan_count,
        "last_scan": radar_svc.last_scan_time.isoformat() if radar_svc.last_scan_time else None,
        "signal_quality": quality,
        "timestamp": datetime.utcnow().isoformat()
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
        "tracks": [t.dict() for t in tracks],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/threats")
async def get_threat_metrics():
    """Get threat assessment metrics."""
    threat_svc = get_threat_service()
    critical = await threat_svc.get_critical_threats()
    
    return {
        "critical_threats": len(critical),
        "threat_history_count": len(threat_svc.threat_history),
        "critical_threat_ids": [t.track_id for t in critical],
        "timestamp": datetime.utcnow().isoformat()
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
    return {
        "running": False,
        "error": "Controller not initialized",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/system")
async def get_system_metrics():
    """Get overall system metrics with guaranteed data."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        if _controller:
            pipeline_status = await _controller.get_status()
        else:
            pipeline_status = {"running": False, "cycle_count": 0}
        
        # Aggregate metrics from all services
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "operational",
            "cpu_percent": float(cpu_percent),
            "memory_percent": float(memory.percent),
            "memory_mb": float(memory.used / (1024 * 1024)),
            "pipeline": pipeline_status,
            "uptime_seconds": pipeline_status.get("uptime_seconds", 0)
        }
    except Exception as e:
        # Fallback response
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "unknown",
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_mb": 0.0,
            "pipeline": {"running": False},
            "uptime_seconds": 0,
            "error": str(e)
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
    
    # Ensure all stages have default values
    stages = {
        "radar_scan_ms": stats.get("radar_scan", {}).get("latest", 0),
        "detection_ms": stats.get("detection", {}).get("latest", 0),
        "tracking_ms": stats.get("tracking", {}).get("latest", 0),
        "threat_assessment_ms": stats.get("threat_assessment", {}).get("latest", 0),
        "ew_response_ms": stats.get("ew_response", {}).get("latest", 0),
        "websocket_send_ms": stats.get("websocket_send", {}).get("latest", 0),
        "total_cycle_ms": stats.get("total_cycle", {}).get("latest", 0),
        "avg_cycle_ms": stats.get("total_cycle", {}).get("avg", 0)
    }
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        **stages
    }


@router.get("/health/cpu-memory")
async def get_cpu_memory():
    """Get CPU and memory usage."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        
        return {
            "cpu_percent": float(cpu_percent),
            "memory_mb": float(memory_info.rss / (1024 * 1024)),
            "memory_percent": float(process.memory_percent()),
            "num_threads": process.num_threads(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "cpu_percent": 0.0,
            "memory_mb": 0.0,
            "memory_percent": 0.0,
            "num_threads": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/live")
async def get_live_metrics():
    """Get live rolling metrics from current pipeline cycle."""
    metrics_store = get_metrics_store()
    latest = metrics_store.get_latest_metrics()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "latest": latest,
        "metrics": latest
    }


@router.get("/live/history")
async def get_metrics_history(limit: int = 100):
    """Get metrics history for graphing."""
    metrics_store = get_metrics_store()
    history = metrics_store.get_metrics_history(limit)
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "count": len(history),
        "data": history
    }


@router.get("/live/summary")
async def get_metrics_summary():
    """Get metrics summary statistics."""
    metrics_store = get_metrics_store()
    summary = metrics_store.get_summary()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        **summary
    }

@router.get("/report")
async def get_metrics_report():
    """Get ML model metrics report with demo data."""
    controller = _controller
    
    if controller and controller.cycle_count > 0:
        # Return metrics based on actual pipeline runs
        cycle_count = controller.cycle_count
        uptime = (controller.startup_time and (datetime.utcnow() - controller.startup_time).total_seconds()) or 0
    else:
        cycle_count = 0
        uptime = 0
    
    # Generate realistic ML metrics report (demo data)
    return {
        "accuracy": 0.894 + (cycle_count % 100) * 0.0001,  # Vary slightly with cycle count
        "metadata": {
            "model_name": "AEGIS Detection V2.1",
            "timestamp": datetime.utcnow().isoformat(),
            "n_samples": 12847 + cycle_count,
            "n_classes": 7,
            "training_time_seconds": 3847,
            "uptime_seconds": int(uptime)
        },
        "macro_avg": {
            "precision": 0.871,
            "recall": 0.885,
            "f1": 0.878
        },
        "weighted_avg": {
            "precision": 0.893,
            "recall": 0.894,
            "f1": 0.893
        },
        "classification_report": {
            "DRONE": {"precision": 0.92, "recall": 0.89, "f1": 0.905, "support": 2048},
            "AIRCRAFT": {"precision": 0.87, "recall": 0.91, "f1": 0.89, "support": 1836},
            "BIRD": {"precision": 0.78, "recall": 0.82, "f1": 0.80, "support": 1624},
            "HELICOPTER": {"precision": 0.88, "recall": 0.85, "f1": 0.865, "support": 1456},
            "CLUTTER": {"precision": 0.85, "recall": 0.88, "f1": 0.865, "support": 2049},
            "UNKNOWN": {"precision": 0.72, "recall": 0.75, "f1": 0.735, "support": 1000},
            "accuracy": None,
            "macro avg": {"precision": 0.862, "recall": 0.862, "f1": 0.862, "support": 9965},
            "weighted avg": {"precision": 0.877, "recall": 0.878, "f1": 0.877, "support": 9965}
        }
    }