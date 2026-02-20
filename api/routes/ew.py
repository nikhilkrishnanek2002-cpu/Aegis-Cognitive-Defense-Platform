"""
EW (Electronic Warfare) defense status route.
"""
from fastapi import APIRouter, Depends
from api.auth_utils import get_current_user
from api import state as S

router = APIRouter(prefix="/api/ew", tags=["ew"])


@router.get("/status")
async def ew_status(user: dict = Depends(get_current_user)):
    """Returns the current EW defense state."""
    return {
        "ew_enabled": True,
        "last_threat_level": "green",
        "message": "EW status is updated in real-time via /api/radar/scan or /ws/stream"
    }
