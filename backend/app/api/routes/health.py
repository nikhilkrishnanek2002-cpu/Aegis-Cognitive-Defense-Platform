"""Health check endpoint."""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Aegis Cognitive Defense API",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check - all services initialized."""
    from app.services.radar_service import get_radar_service
    from app.services.detection_service import get_detection_service
    from app.services.tracking_service import get_tracking_service
    from app.services.threat_service import get_threat_service
    from app.services.ew_service import get_ew_service
    
    try:
        get_radar_service()
        get_detection_service()
        get_tracking_service()
        get_threat_service()
        get_ew_service()
        
        return {"ready": True}
    except Exception as e:
        return {"ready": False, "error": str(e)}
