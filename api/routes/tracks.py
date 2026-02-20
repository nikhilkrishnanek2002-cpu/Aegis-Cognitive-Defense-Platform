"""
Tracking routes: current track state and reset.
"""
from fastapi import APIRouter, Depends
from api.auth_utils import get_current_user
from api import state as S

router = APIRouter(prefix="/api/tracks", tags=["tracks"])


@router.get("")
async def get_tracks(user: dict = Depends(get_current_user)):
    """Return all active confirmed tracks."""
    tracks = S.tracker.get_active_tracks()
    return {
        tid: {
            "position": list(v["position"]),
            "velocity": list(v["velocity"]),
            "state": v["state"],
            "confidence": round(v["confidence"], 4),
            "measurement_count": v["measurement_count"],
        }
        for tid, v in tracks.items()
    }


@router.delete("/reset")
async def reset_tracker(user: dict = Depends(get_current_user)):
    """Reset all tracks."""
    S.tracker.reset()
    return {"message": "Tracker reset"}
