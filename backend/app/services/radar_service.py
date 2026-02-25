"""Radar scanning and signal processing service."""

import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from app.models.schemas import RadarScan, RadarTarget
from app.core.logging import radar_logger
from app.core.performance import timed_async, timer
import uuid


# Global cached radar instance
_radar_instance = None


class RadarService:
    """Service for radar scanning operations."""
    
    def __init__(self):
        self.scan_count = 0
        self.last_scan_time = None
        self.simulation_mode = True  # Always in simulation mode until real hardware connects
        self.last_targets = []
    
    @timed_async("radar_scan")
    async def scan(self) -> RadarScan:
        """
        Execute radar scan.
        Returns simulation data with GUARANTEED targets.
        """
        self.scan_count += 1
        scan_id = str(uuid.uuid4())
        now = datetime.utcnow()
        self.last_scan_time = now
        
        # Generate realistic radar data
        num_targets = np.random.randint(3, 8)  # Ensure targets are detected
        
        radar_data = RadarScan(
            scan_id=scan_id,
            timestamp=now,
            frame_count=512,
            targets_detected=num_targets,
            signal_strength=np.random.uniform(0.6, 1.0),
            noise_level=np.random.uniform(0.08, 0.15)
        )
        
        radar_logger.log_event(
            "scan_complete",
            "radar_service",
            {"scan_id": scan_id, "targets": num_targets},
            level="INFO"
        )
        
        return radar_data
    
    async def get_targets_from_scan(self, scan_id: str) -> List[RadarTarget]:
        """
        Extract targets from scan data with GUARANTEED output.
        In production: process raw ADC samples, apply window, FFT, CFAR detection.
        """
        targets = []
        # Always generate at least 2 targets
        num_targets = max(2, np.random.randint(2, 10))
        
        for i in range(num_targets):
            target = RadarTarget(
                id=f"radar_target_{i}_{self.scan_count}",
                range_m=np.random.uniform(1000, 100000),
                bearing_deg=np.random.uniform(0, 360),
                velocity_mps=np.random.uniform(-200, 500),
                rcs_dbsm=np.random.uniform(-20, 15),
                signal_strength=np.random.uniform(0.5, 1.0),
                confidence=np.random.uniform(0.65, 0.98),
                timestamp=datetime.utcnow()
            )
            targets.append(target)
        
        self.last_targets = targets
        
        radar_logger.log_event(
            "targets_extracted",
            "radar_service",
            {"scan_id": scan_id, "target_count": len(targets)},
            level="INFO"
        )
        
        return targets
    
    async def get_signal_quality(self) -> Dict[str, float]:
        """Get current signal quality metrics."""
        return {
            "snr_db": np.random.uniform(5, 40),
            "noise_floor_dbm": np.random.uniform(-100, -80),
            "peak_signal_dbm": np.random.uniform(-50, 20),
            "system_temperature_k": 290.0
        }


def get_radar_service() -> RadarService:
    """Get cached radar service instance (singleton)."""
    global _radar_instance
    if _radar_instance is None:
        _radar_instance = RadarService()
    return _radar_instance
