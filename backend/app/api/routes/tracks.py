"""Tracking routes: current track state and reset."""

from fastapi import APIRouter, Depends
from typing import Dict, List, Any
from app.services.tracking_service import get_tracking_service
from app.engine.controller import _controller

router = APIRouter(prefix="/api/tracks", tags=["tracks"])


async def get_current_user(token: str = None) -> dict:
    """Simple auth check."""
    return {"username": "user", "role": "operator"}


@router.get("")
async def get_tracks(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get all active confirmed tracks."""
    if not _controller:
        return {"count": 0, "tracks": {}}
    
    tracking_svc = get_tracking_service()
    
    # Get tracks from controller pipeline if available
    tracks = {}
    
    if hasattr(_controller, 'pipeline') and hasattr(_controller.pipeline, 'last_tracks'):
        for track_id, track_data in _controller.pipeline.last_tracks.items():
            tracks[track_id] = {
                "track_id": track_id,
                "position": track_data.get("position", [0, 0, 0]),
                "velocity": track_data.get("velocity", [0, 0, 0]),
                "confidence": float(track_data.get("confidence", 0.0)),
                "age": int(track_data.get("age", 0))
            }
    
    return {
        "count": len(tracks),
        "tracks": tracks
    }


@router.get("/active")
async def get_active_tracks(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get only active (high confidence) tracks."""
    if not _controller:
        return {"count": 0, "tracks": {}}
    
    tracking_svc = get_tracking_service()
    
    active_tracks = {}
    if hasattr(_controller, 'pipeline') and hasattr(_controller.pipeline, 'last_tracks'):
        for track_id, track_data in _controller.pipeline.last_tracks.items():
            confidence = float(track_data.get("confidence", 0.0))
            if confidence > 0.6:  # Only tracks with >60% confidence
                active_tracks[track_id] = {
                    "track_id": track_id,
                    "position": track_data.get("position", [0, 0, 0]),
                    "velocity": track_data.get("velocity", [0, 0, 0]),
                    "confidence": confidence,
                    "age": int(track_data.get("age", 0))
                }
    
    return {
        "count": len(active_tracks),
        "tracks": active_tracks
    }


@router.delete("/reset")
async def reset_tracks(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Reset all tracks."""
    if not _controller:
        return {"success": False, "message": "Controller not initialized"}
    
    tracking_svc = get_tracking_service()
    
    # Reset tracking service
    if hasattr(tracking_svc, 'reset'):
        await tracking_svc.reset()
    
    # Reset controller pipeline tracks
    if hasattr(_controller, 'pipeline') and hasattr(_controller.pipeline, 'last_tracks'):
        _controller.pipeline.last_tracks.clear()
    
    return {
        "success": True,
        "message": "All tracks reset"
    }


@router.get("/summary")
async def get_tracks_summary(user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get summary statistics of tracks."""
    if not _controller:
        return {
            "total_tracks": 0,
            "active_tracks": 0,
            "avg_confidence": 0.0
        }
    
    tracking_svc = get_tracking_service()
    
    total = 0
    active = 0
    confidence_sum = 0.0
    
    if hasattr(_controller, 'pipeline') and hasattr(_controller.pipeline, 'last_tracks'):
        tracks = _controller.pipeline.last_tracks
        total = len(tracks)
        
        for track_data in tracks.values():
            confidence = float(track_data.get("confidence", 0.0))
            confidence_sum += confidence
            if confidence > 0.6:
                active += 1
    
    avg_confidence = confidence_sum / total if total > 0 else 0.0
    
    return {
        "total_tracks": total,
        "active_tracks": active,
        "avg_confidence": avg_confidence
    }
