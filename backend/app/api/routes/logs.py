"""System logs retrieval endpoints."""

from fastapi import APIRouter, Query
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import os

router = APIRouter(prefix="/api/logs", tags=["logs"])

LOGS_DIR = "./logs"


@router.get("/pipeline")
async def get_pipeline_logs(limit: int = Query(100, ge=1, le=1000)):
    """Get recent pipeline logs."""
    log_file = os.path.join(LOGS_DIR, "pipeline.log")
    return _read_log_file(log_file, limit)


@router.get("/radar")
async def get_radar_logs(limit: int = Query(100, ge=1, le=1000)):
    """Get recent radar logs."""
    log_file = os.path.join(LOGS_DIR, "radar.log")
    return _read_log_file(log_file, limit)


@router.get("/detection")
async def get_detection_logs(limit: int = Query(100, ge=1, le=1000)):
    """Get recent detection logs."""
    log_file = os.path.join(LOGS_DIR, "detection.log")
    return _read_log_file(log_file, limit)


@router.get("/tracking")
async def get_tracking_logs(limit: int = Query(100, ge=1, le=1000)):
    """Get recent tracking logs."""
    log_file = os.path.join(LOGS_DIR, "tracking.log")
    return _read_log_file(log_file, limit)


@router.get("/threat")
async def get_threat_logs(limit: int = Query(100, ge=1, le=1000)):
    """Get recent threat assessment logs."""
    log_file = os.path.join(LOGS_DIR, "threat.log")
    return _read_log_file(log_file, limit)


@router.get("/websocket")
async def get_websocket_logs(limit: int = Query(100, ge=1, le=1000)):
    """Get recent websocket logs."""
    log_file = os.path.join(LOGS_DIR, "websocket.log")
    return _read_log_file(log_file, limit)


@router.get("/all")
async def get_all_logs(limit: int = Query(50, ge=1, le=500)):
    """Get recent logs from all modules."""
    all_logs = {}
    
    for log_file in os.listdir(LOGS_DIR):
        if log_file.endswith(".log"):
            log_name = log_file.replace(".log", "")
            log_path = os.path.join(LOGS_DIR, log_file)
            all_logs[log_name] = _read_log_file(log_path, limit)
    
    return all_logs


@router.get("/summary")
async def get_logs_summary():
    """Get summary of available logs."""
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "logs_directory": LOGS_DIR,
        "log_files": []
    }
    
    try:
        if os.path.exists(LOGS_DIR):
            for log_file in os.listdir(LOGS_DIR):
                if log_file.endswith(".log"):
                    log_path = os.path.join(LOGS_DIR, log_file)
                    size = os.path.getsize(log_path)
                    modified = datetime.fromtimestamp(os.path.getmtime(log_path))
                    
                    summary["log_files"].append({
                        "name": log_file,
                        "size_bytes": size,
                        "modified": modified.isoformat(),
                        "url": f"/api/logs/{log_file.replace('.log', '')}"
                    })
    except Exception as e:
        summary["error"] = str(e)
    
    return summary


def _read_log_file(log_file: str, limit: int = 100) -> Dict[str, Any]:
    """Read last N lines from a log file."""
    try:
        if not os.path.exists(log_file):
            return {
                "status": "not_found",
                "message": f"Log file not found: {log_file}",
                "logs": []
            }
        
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        # Get last N lines
        recent_lines = lines[-limit:] if lines else []
        
        return {
            "status": "success",
            "file": log_file,
            "total_lines": len(lines),
            "returned_lines": len(recent_lines),
            "logs": recent_lines
        }
    
    except Exception as e:
        return {
            "status": "error",
            "file": log_file,
            "error": str(e),
            "logs": []
        }
