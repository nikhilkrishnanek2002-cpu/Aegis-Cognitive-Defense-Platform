"""Threat assessment endpoints."""

from fastapi import APIRouter
from datetime import datetime, timedelta
from app.services.threat_service import get_threat_service
from app.services.ew_service import get_ew_service
from app.models.schemas import ThreatLevel
from app.engine.controller import _controller

router = APIRouter(prefix="/api/threats", tags=["threats"])


@router.get("/active")
async def get_active_threats():
    """Get currently active threats."""
    threat_svc = get_threat_service()
    
    if not _controller:
        return {"threats": [], "count": 0}
    
    threats = _controller.pipeline.last_threats
    
    return {
        "count": len(threats),
        "threats": [t.dict() for t in threats]
    }


@router.get("/critical")
async def get_critical_threats():
    """Get all critical threats."""
    threat_svc = get_threat_service()
    critical = await threat_svc.get_critical_threats()
    
    return {
        "count": len(critical),
        "threats": [t.dict() for t in critical]
    }


@router.get("/summary")
async def get_threat_summary():
    """Get threat summary by level."""
    threat_svc = get_threat_service()
    
    if not _controller:
        return {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "total": 0
        }
    
    threats = _controller.pipeline.last_threats
    
    summary = {
        "critical": sum(1 for t in threats if t.threat_level == ThreatLevel.CRITICAL),
        "high": sum(1 for t in threats if t.threat_level == ThreatLevel.HIGH),
        "medium": sum(1 for t in threats if t.threat_level == ThreatLevel.MEDIUM),
        "low": sum(1 for t in threats if t.threat_level == ThreatLevel.LOW),
        "total": len(threats)
    }
    
    return summary


@router.get("/history")
async def get_threat_history(limit: int = 100):
    """Get threat history (last N threats)."""
    threat_svc = get_threat_service()
    
    history = threat_svc.threat_history[-limit:]
    
    return {
        "count": len(history),
        "threats": [t.dict() for t in history]
    }


@router.get("/ew-status")
async def get_ew_status():
    """Get EW response status."""
    ew_svc = get_ew_service()
    status = await ew_svc.get_ew_status()
    
    return {
        "ew": status,
        "timestamp": datetime.utcnow().isoformat()
    }
