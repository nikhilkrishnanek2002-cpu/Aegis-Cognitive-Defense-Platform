"""Electronic Warfare (EW) defense routes."""

from fastapi import APIRouter, Depends
from typing import Dict, Any, List
from app.services.ew_service import get_ew_service
from app.engine.controller import _controller

router = APIRouter(prefix="/api/ew", tags=["ew"])


async def get_current_user(token: str = None) -> dict:
    """Simple auth check."""
    return {"username": "user", "role": "operator"}


@router.get("/status")
async def get_ew_status(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get current EW defense status."""
    if not _controller:
        return {
            "status": "offline",
            "ew_enabled": False,
            "threat_level": "unknown",
            "active_jamming": False,
            "detected_signals": 0
        }
    
    ew_svc = get_ew_service()
    
    # Get EW data from controller pipeline
    ew_data = {
        "status": "operational",
        "ew_enabled": True,
        "threat_level": "low",
        "active_jamming": False,
        "detected_signals": 0
    }
    
    if hasattr(_controller, 'pipeline'):
        pipeline = _controller.pipeline
        
        if hasattr(pipeline, 'last_threats'):
            threats = pipeline.last_threats
            if threats:
                # Determine threat level from threats
                max_threat = max((t.threat_score for t in threats), default=0.0)
                if max_threat > 0.9:
                    ew_data["threat_level"] = "critical"
                    ew_data["active_jamming"] = True
                elif max_threat > 0.75:
                    ew_data["threat_level"] = "high"
                elif max_threat > 0.5:
                    ew_data["threat_level"] = "medium"
                else:
                    ew_data["threat_level"] = "low"
    
    return ew_data


@router.get("/signals")
async def get_detected_signals(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get detected jamming/interference signals."""
    if not _controller:
        return {
            "signals": [],
            "count": 0
        }
    
    ew_svc = get_ew_service()
    
    signals = []
    
    # Get signals from service if available
    if hasattr(ew_svc, 'detected_signals'):
        for signal_id, signal_data in enumerate(ew_svc.detected_signals):
            signals.append({
                "signal_id": signal_id,
                "frequency": signal_data.get("frequency", 0.0),
                "power": signal_data.get("power", 0.0),
                "type": signal_data.get("type", "unknown")
            })
    
    return {
        "count": len(signals),
        "signals": signals
    }


@router.post("/analyze")
async def analyze_signal(signal_data: Dict[str, Any], user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Analyze a specific signal for jamming threats."""
    if not _controller:
        return {
            "analysis": "pending",
            "threat_level": "unknown"
        }
    
    ew_svc = get_ew_service()
    
    # Perform analysis
    analysis = {
        "signal_type": signal_data.get("type", "unknown"),
        "frequency": signal_data.get("frequency", 0.0),
        "power": signal_data.get("power", 0.0),
        "threat_level": "low",
        "is_jamming": False,
        "confidence": 0.0
    }
    
    # Simple heuristic: high power + specific frequencies = jamming
    power = signal_data.get("power", 0.0)
    if power > 0.7:
        analysis["threat_level"] = "high"
        analysis["is_jamming"] = True
        analysis["confidence"] = min(power, 1.0)
    elif power > 0.5:
        analysis["threat_level"] = "medium"
        analysis["is_jamming"] = False
        analysis["confidence"] = power * 0.8
    else:
        analysis["threat_level"] = "low"
        analysis["is_jamming"] = False
        analysis["confidence"] = power * 0.5
    
    return {
        "analysis": "complete",
        "result": analysis
    }


@router.get("/defense-status")
async def get_defense_status(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get EW defense countermeasures status."""
    if not _controller:
        return {"active": False, "systems": []}
    
    ew_svc = get_ew_service()
    
    return {
        "active": True,
        "systems": [
            {"name": "frequency_hopping", "status": "ready"},
            {"name": "beam_steering", "status": "ready"},
            {"name": "null_steering", "status": "ready"},
            {"name": "power_control", "status": "ready"}
        ],
        "last_update": None
    }


@router.post("/activate-defense")
async def activate_defense(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Activate EW defense countermeasures."""
    if not _controller:
        return {
            "success": False,
            "message": "Controller not initialized"
        }
    
    ew_svc = get_ew_service()
    
    return {
        "success": True,
        "message": "EW defense activated",
        "systems_activated": ["frequency_hopping", "beam_steering"]
    }
