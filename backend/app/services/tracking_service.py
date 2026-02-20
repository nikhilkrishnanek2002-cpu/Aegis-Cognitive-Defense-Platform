"""Multi-target tracking service (Kalman filter + Hungarian algorithm)."""

import numpy as np
from typing import List, Dict
from datetime import datetime
from app.models.schemas import DetectionResult, TrackedTarget
from app.core.logging import tracking_logger
from app.core.config import get_config
from app.core.performance import timed_async, timer
import uuid


# Global tracking state
_tracked_targets: Dict[str, TrackedTarget] = {}
_tracking_service = None


class KalmanTracker:
    """Simple Kalman filter-based tracker."""
    
    def __init__(self, track_id: str, initial_detection: DetectionResult):
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.confidence = initial_detection.confidence
        self.target_type = initial_detection.target_type
        self.last_update = datetime.utcnow()
    
    def update(self, detection: DetectionResult):
        """Update track with new detection."""
        self.hits += 1
        self.age += 1
        self.confidence = max(self.confidence, detection.confidence)
        self.last_update = datetime.utcnow()
        
        # Simple position update
        self.position[0] = detection.features.get("range", 0)
        self.position[1] = detection.features.get("bearing", 0)
        self.position[2] = detection.features.get("velocity", 0)
        
        self.velocity[0] = detection.features.get("velocity", 0)
    
    def predict(self):
        """Predict next state."""
        self.age += 1
        # In production: full Kalman prediction
    
    def to_tracked_target(self) -> TrackedTarget:
        """Convert to schema."""
        return TrackedTarget(
            track_id=self.track_id,
            target_type=self.target_type,
            position={"x": float(self.position[0]), "y": float(self.position[1]), "z": float(self.position[2])},
            velocity={"vx": float(self.velocity[0]), "vy": 0.0, "vz": 0.0},
            hits=self.hits,
            age=self.age,
            confidence=self.confidence,
            last_update=self.last_update
        )


class TrackingService:
    """Service for multi-target tracking."""
    
    def __init__(self):
        self.config = get_config()
        self.tracks: Dict[str, KalmanTracker] = {}
    
    @timed_async("tracking")
    async def update_tracks(self, detections: List[DetectionResult]) -> List[TrackedTarget]:
        """
        Update tracker with new detections.
        
        Algorithm:
        1. Predict track states
        2. Associate detections to tracks (Hungarian algorithm in production)
        3. Update matched tracks
        4. Create new tracks for unmatched detections
        5. Remove old tracks
        """
        
        # Predict all tracks
        for track in self.tracks.values():
            track.predict()
        
        # Simple association: match by closest range
        matched_tracks = set()
        
        for detection in detections:
            best_track_id = None
            best_distance = float("inf")
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                det_range = detection.features.get("range", 0)
                det_bearing = detection.features.get("bearing", 0)
                
                track_range = track.position[0]
                track_bearing = track.position[1]
                
                # Simple Euclidean distance
                distance = np.sqrt((det_range - track_range)**2 + (det_bearing - track_bearing)**2)
                
                if distance < best_distance and distance < 5000:  # Distance threshold
                    best_distance = distance
                    best_track_id = track_id
            
            if best_track_id:
                self.tracks[best_track_id].update(detection)
                matched_tracks.add(best_track_id)
            else:
                # Create new track
                new_track_id = str(uuid.uuid4())[:8]
                self.tracks[new_track_id] = KalmanTracker(new_track_id, detection)
        
        # Remove old tracks
        tracks_to_remove = [
            track_id for track_id, track in self.tracks.items()
            if track.age > self.config.tracking_max_age
        ]
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Convert to schema
        tracked = [track.to_tracked_target() for track in self.tracks.values()]
        
        tracking_logger.log_event(
            "tracking_updated",
            "tracking_service",
            {"active_tracks": len(tracked), "detections_processed": len(detections)},
            level="INFO"
        )
        
        return tracked
    
    async def get_active_tracks(self) -> List[TrackedTarget]:
        """Get all active tracks."""
        return [track.to_tracked_target() for track in self.tracks.values()]


def get_tracking_service() -> TrackingService:
    """Get cached tracking service instance (singleton)."""
    global _tracking_service
    if _tracking_service is None:
        _tracking_service = TrackingService()
    return _tracking_service
