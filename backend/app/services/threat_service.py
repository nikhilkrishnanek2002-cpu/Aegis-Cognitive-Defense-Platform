"""Threat assessment and evaluation service."""

import numpy as np
from typing import List, Dict
from datetime import datetime
from app.models.schemas import TrackedTarget, Threat, ThreatLevel, TargetType
from app.core.logging import threat_logger
from app.core.config import get_config
from app.core.performance import timed_async, timer


_threat_service = None


class ThreatAssessmentEngine:
    """Engine for assessing threat levels."""
    
    def __init__(self):
        self.config = get_config()
        
        # Target threat base scores (lower = less threatening)
        self.target_threat_scores = {
            TargetType.MISSILE: 0.95,
            TargetType.AIRCRAFT: 0.70,
            TargetType.HELICOPTER: 0.75,
            TargetType.DRONE: 0.60,
            TargetType.BIRD: 0.05,
            TargetType.CLUTTER: 0.01,
            TargetType.UNKNOWN: 0.40,
        }
    
    def calculate_threat_score(self, track: TrackedTarget) -> float:
        """
        Calculate threat score (0-1) based on multiple factors.
        
        Factors:
        - Target type
        - Range (closer = more threatening)
        - Velocity (moving toward = more threatening)
        - Confidence
        """
        # Base score from target type
        base_score = self.target_threat_scores.get(track.target_type, 0.5)
        
        # Range factor: closer targets are more threatening
        range_m = track.position.get("x", 50000)
        range_factor = min(1.0, 10000.0 / max(range_m, 100))
        
        # Velocity factor: high velocity = more threatening
        velocity_mps = track.velocity.get("vx", 0)
        velocity_factor = min(1.0, abs(velocity_mps) / 300.0)
        
        # Confidence factor
        confidence_factor = track.confidence
        
        # Weighted combination
        threat_score = (
            base_score * 0.4 +
            range_factor * 0.3 +
            velocity_factor * 0.2 +
            confidence_factor * 0.1
        )
        
        return min(1.0, max(0.0, threat_score))
    
    def classify_threat_level(self, threat_score: float) -> ThreatLevel:
        """Classify threat level based on score."""
        if threat_score >= self.config.threat_threshold_critical:
            return ThreatLevel.CRITICAL
        elif threat_score >= self.config.threat_threshold_high:
            return ThreatLevel.HIGH
        elif threat_score >= 0.50:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def estimate_time_to_impact(self, track: TrackedTarget) -> float:
        """Estimate time for target to reach launch area."""
        range_m = track.position.get("x", 50000)
        velocity_mps = track.velocity.get("vx", 0)
        
        if velocity_mps <= 0:
            return float("inf")  # Not approaching
        
        time_to_impact = range_m / abs(velocity_mps)
        return max(0, time_to_impact)


class ThreatService:
    """Service for threat assessment and evaluation."""
    
    def __init__(self):
        self.engine = ThreatAssessmentEngine()
        self.threat_history: List[Threat] = []
        self.critical_threats: List[str] = []
    
    @timed_async("threat_assessment")
    async def assess_threats(self, tracks: List[TrackedTarget]) -> List[Threat]:
        """
        Assess threat level for each tracked target.
        
        Returns list of threats, filtered by threat threshold.
        """
        threats = []
        
        for track in tracks:
            threat_score = self.engine.calculate_threat_score(track)
            threat_level = self.engine.classify_threat_level(threat_score)
            tti = self.engine.estimate_time_to_impact(track)
            
            threat = Threat(
                track_id=track.track_id,
                threat_level=threat_level,
                threat_score=threat_score,
                target_type=track.target_type,
                position=track.position,
                velocity=track.velocity,
                time_to_impact_s=tti if tti != float("inf") else None,
                intercept_point=self._compute_intercept_point(track),
                timestamp=datetime.utcnow(),
                confidence=track.confidence
            )
            
            threats.append(threat)
            self.threat_history.append(threat)
            
            # Track critical threats
            if threat_level == ThreatLevel.CRITICAL:
                if track.track_id not in self.critical_threats:
                    self.critical_threats.append(track.track_id)
                    threat_logger.log_event(
                        "critical_threat_detected",
                        "threat_service",
                        {"track_id": track.track_id, "threat_score": threat_score},
                        level="ERROR"
                    )
            elif track.track_id in self.critical_threats:
                self.critical_threats.remove(track.track_id)
        
        threat_logger.log_event(
            "threat_assessment_complete",
            "threat_service",
            {"assessed": len(threats), "critical": len(self.critical_threats)},
            level="INFO"
        )
        
        return threats
    
    def _compute_intercept_point(self, track: TrackedTarget) -> Dict[str, float]:
        """
        Compute predicted intercept point.
        In production: use more sophisticated ballistic trajectory prediction.
        """
        range_m = track.position.get("x", 0)
        bearing_deg = track.position.get("y", 0)
        velocity_mps = track.velocity.get("vx", 0)
        
        if velocity_mps <= 0:
            return None
        
        tti = range_m / abs(velocity_mps)
        
        return {
            "range_m": 0.0,  # Intercept at launch area
            "bearing_deg": bearing_deg,
            "time_s": tti
        }
    
    async def get_critical_threats(self) -> List[Threat]:
        """Get all current critical threats."""
        return [t for t in self.threat_history if t.threat_level == ThreatLevel.CRITICAL]


def get_threat_service() -> ThreatService:
    """Get cached threat service instance (singleton)."""
    global _threat_service
    if _threat_service is None:
        _threat_service = ThreatService()
    return _threat_service
