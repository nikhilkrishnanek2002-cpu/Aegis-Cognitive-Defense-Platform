"""Electronic Warfare (EW) detection and response service."""

import numpy as np
from typing import List
from datetime import datetime
from app.models.schemas import EWSignal, EWResponse, Threat, ThreatLevel
from app.core.logging import ew_logger
from app.core.performance import timed_async, timer
import uuid


_ew_service = None


class EWResponseEngine:
    """Engine for EW threat detection and countermeasure response."""
    
    def __init__(self):
        self.response_types = [
            "JAMMING",
            "SPOOFING",
            "DECEPTION",
            "CHAFF",
            "FLARES"
        ]
    
    def should_trigger_response(self, threat: Threat) -> bool:
        """Determine if EW response should be triggered."""
        # Trigger on critical threats
        if threat.threat_level == ThreatLevel.CRITICAL:
            return True
        
        # Trigger on high threats with short time to impact
        if threat.threat_level == ThreatLevel.HIGH:
            tti = threat.time_to_impact_s
            if tti is not None and tti < 60:  # Less than 60 seconds
                return True
        
        return False
    
    def select_response(self, threat: Threat) -> str:
        """Select appropriate EW response based on threat."""
        # Missiles trigger jamming/spoofing
        # Aircraft trigger deception
        # Drones trigger multi-faceted response
        
        from app.models.schemas import TargetType
        
        if threat.target_type == TargetType.MISSILE:
            return "JAMMING"
        elif threat.target_type == TargetType.AIRCRAFT:
            return "DECEPTION"
        elif threat.target_type == TargetType.DRONE:
            return "SPOOFING"
        else:
            return np.random.choice(self.response_types)
    
    def compute_response_parameters(self, threat: Threat) -> dict:
        """Compute EW response parameters."""
        # In production: use sophisticated EW modeling
        
        range_m = threat.position.get("x", 10000)
        
        # Response frequency near threat frequency
        freq_offset = np.random.uniform(-50, 50)
        response_freq = 2400 + freq_offset  # S-band radar
        
        # Power based on range
        response_power = 20 + 10 * np.log10(max(1, range_m / 1000))
        
        # Duration based on threat level
        duration_ms = {
            ThreatLevel.CRITICAL: 5000,
            ThreatLevel.HIGH: 3000,
            ThreatLevel.MEDIUM: 1000,
            ThreatLevel.LOW: 500
        }.get(threat.threat_level, 1000)
        
        return {
            "frequency_mhz": response_freq,
            "power_dbm": response_power,
            "duration_ms": duration_ms
        }


class EWService:
    """Service for EW threat detection and response."""
    
    def __init__(self):
        self.engine = EWResponseEngine()
        self.active_signals: List[EWSignal] = []
        self.response_history: List[EWResponse] = []
    
    async def detect_ew_signals(self) -> List[EWSignal]:
        """
        Detect incoming EW signals (jamming, spoofing, etc).
        
        In production: continuously monitor spectrum, analyze modulation, track emitters.
        """
        signals = []
        
        # Simulate random EW signal detections
        num_signals = np.random.randint(0, 3)
        
        for i in range(num_signals):
            signal = EWSignal(
                signal_id=f"ew_signal_{i}",
                freq_mhz=np.random.uniform(2000, 6000),
                power_dbm=np.random.uniform(-50, 20),
                signal_type=np.random.choice(["JAMMING", "SPOOFING", "PROBE"]),
                timestamp=datetime.utcnow()
            )
            signals.append(signal)
            self.active_signals.append(signal)
        
        if signals:
            ew_logger.log_event(
                "ew_signals_detected",
                "ew_service",
                {"count": len(signals)},
                level="WARNING"
            )
        
        return signals
    
    @timed_async("ew_response")
    async def generate_responses(self, threats: List[Threat]) -> List[EWResponse]:
        """
        Generate EW countermeasure responses for active threats.
        """
        responses = []
        
        for threat in threats:
            if self.engine.should_trigger_response(threat):
                response_type = self.engine.select_response(threat)
                params = self.engine.compute_response_parameters(threat)
                
                response = EWResponse(
                    response_id=str(uuid.uuid4())[:8],
                    signal_id=threat.track_id,
                    response_type=response_type,
                    frequency_mhz=params["frequency_mhz"],
                    power_dbm=params["power_dbm"],
                    duration_ms=params["duration_ms"],
                    timestamp=datetime.utcnow()
                )
                
                responses.append(response)
                self.response_history.append(response)
                
                ew_logger.log_event(
                    "ew_response_triggered",
                    "ew_service",
                    {
                        "response_id": response.response_id,
                        "response_type": response_type,
                        "threat_level": threat.threat_level
                    },
                    level="WARNING"
                )
        
        return responses
    
    async def get_ew_status(self) -> dict:
        """Get current EW status."""
        return {
            "active_signals": len(self.active_signals),
            "recent_responses": len([r for r in self.response_history if (datetime.utcnow() - r.timestamp).total_seconds() < 300]),
            "total_responses": len(self.response_history)
        }


def get_ew_service() -> EWService:
    """Get cached EW service instance (singleton)."""
    global _ew_service
    if _ew_service is None:
        _ew_service = EWService()
    return _ew_service
