"""Radar endpoints."""

from fastapi import APIRouter, Body
from app.services.radar_service import get_radar_service
from app.services.tracking_service import get_tracking_service
from app.services.detection_service import get_detection_service
from app.engine.controller import _controller
from datetime import datetime
import numpy as np
from uuid import uuid4
from pydantic import BaseModel

router = APIRouter(prefix="/api/radar", tags=["radar"])


class ScanRequest(BaseModel):
    """Radar scan request parameters."""
    target: str = "UNKNOWN"
    distance: float = 200.0
    gain_db: float = 15.0


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
    
    targets = _controller.pipeline.last_targets if hasattr(_controller.pipeline, 'last_targets') else []
    return {
        "count": len(targets),
        "targets": [t.dict() if hasattr(t, 'dict') else t for t in targets]
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
async def trigger_scan(request: ScanRequest = Body(...)):
    """Manually trigger a radar scan with XAI data generation."""
    radar_svc = get_radar_service()
    detection_svc = get_detection_service()
    
    # Get radar targets
    scan = await radar_svc.scan()
    targets = await radar_svc.get_targets_from_scan(scan.scan_id)
    
    # Run detection on targets
    detections = await detection_svc.detect_targets(targets if targets else [])
    
    # Generate realistic Grad-CAM heatmap (synthetic demo data)
    scan_id = scan.scan_id[:8]
    heatmap_size = 64
    
    # Create realistic Grad-CAM heatmap with multiple activation regions
    x = np.linspace(-4, 4, heatmap_size)
    y = np.linspace(-4, 4, heatmap_size)
    X, Y = np.meshgrid(x, y)
    
    # Primary activation region (main target detection area)
    cx1, cy1 = np.random.uniform(-1.5, 1.5), np.random.uniform(-1.0, 1.0)
    sx1, sy1 = np.random.uniform(0.6, 1.2), np.random.uniform(0.5, 0.9)
    Z1 = np.exp(-((X - cx1)**2 / (2 * sx1**2) + (Y - cy1)**2 / (2 * sy1**2)))
    
    # Secondary activation region (sidelobe / secondary feature)
    cx2, cy2 = cx1 + np.random.uniform(1.0, 2.5), cy1 + np.random.uniform(-1.0, 1.0)
    sx2, sy2 = np.random.uniform(0.3, 0.6), np.random.uniform(0.3, 0.6)
    Z2 = 0.5 * np.exp(-((X - cx2)**2 / (2 * sx2**2) + (Y - cy2)**2 / (2 * sy2**2)))
    
    # Tertiary weak activation (clutter residual attention)
    cx3, cy3 = np.random.uniform(-3, 3), np.random.uniform(-3, 3)
    Z3 = 0.2 * np.exp(-((X - cx3)**2 / (2 * 0.8**2) + (Y - cy3)**2 / (2 * 0.8**2)))
    
    # Combine activations
    Z = Z1 + Z2 + Z3
    
    # Add realistic thermal-noise-style floor
    noise_floor = np.random.exponential(0.03, Z.shape)
    Z = Z + noise_floor
    
    # Apply slight Gaussian smoothing via convolution for realistic spread
    from scipy.ndimage import gaussian_filter
    Z = gaussian_filter(Z, sigma=1.2)
    
    # Normalize to [0, 1] range (real Grad-CAM is 0-1 activation)
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
    
    target_class = request.target if request.target != "UNKNOWN" else (detections[0].target_type.value if detections else "UNKNOWN")
    
    xai_data = {
        "scan_id": scan_id,
        "heatmap": Z.tolist(),
        "heatmap_shape": [heatmap_size, heatmap_size],
        "target_class": target_class,
        "confidence": float(detections[0].confidence) if detections else 0.75,
        "image_path": f"/api/visualizations/xai-gradcam-image/{scan_id}"
    }
    
    return {
        "success": True,
        "scan_id": scan_id,
        "scan": scan.dict(),
        "detections": [d.dict() for d in detections],
        "xai": xai_data
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
