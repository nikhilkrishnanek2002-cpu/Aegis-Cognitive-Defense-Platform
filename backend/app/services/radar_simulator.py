"""Radar simulation service - generates realistic radar targets continuously."""

import asyncio
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from app.models.schemas import RadarScan, RadarTarget
from app.core.logging import radar_logger
import uuid
import time


class RadarSimulator:
    """Simulates continuous radar scanning with realistic target motion."""
    
    TARGET_TYPES = ["DRONE", "AIRCRAFT", "HELICOPTER", "BIRD", "MISSILE", "UNKNOWN"]
    
    def __init__(self):
        self.scan_count = 0
        self.last_scan_time = None
        self.active_targets: Dict[str, Dict[str, Any]] = {}
        self.simulation_mode = True
        self.simulation_speed = 1.0  # Speed multiplier for target motion
        self.target_counter = 0  # For generating meaningful target names
    
    async def _generate_new_targets(self) -> int:
        """Generate some new targets randomly."""
        num_new = np.random.randint(1, 4)
        
        for _ in range(num_new):
            self.target_counter += 1
            target_type = np.random.choice(self.TARGET_TYPES)
            target_name = f"{target_type}-{self.target_counter}"
            target_id = target_name  # Use human-readable name as ID
            
            self.active_targets[target_id] = {
                "name": target_name,
                "type": target_type,
                "x": np.random.uniform(-500, 500),
                "y": np.random.uniform(-500, 500),
                "vx": np.random.uniform(-30, 30),
                "vy": np.random.uniform(-30, 30),
                "strength": np.random.uniform(0.4, 1.0),
                "created_at": time.time(),
                "hits": 1
            }
        
        return num_new
    
    async def _update_target_positions(self):
        """Move active targets slightly (simulate tracking)."""
        for target_id, target in list(self.active_targets.items()):
            # Update position
            dt = 0.3  # Timestep in seconds
            target["x"] += target["vx"] * dt * self.simulation_speed
            target["y"] += target["vy"] * dt * self.simulation_speed
            
            # Bounce off boundaries
            if target["x"] > 500 or target["x"] < -500:
                target["vx"] *= -1
            if target["y"] > 500 or target["y"] < -500:
                target["vy"] *= -1
            
            # Clamp position
            target["x"] = np.clip(target["x"], -500, 500)
            target["y"] = np.clip(target["y"], -500, 500)
            
            # Randomly remove old targets (5% chance per cycle)
            if np.random.random() < 0.05:
                del self.active_targets[target_id]
            else:
                target["hits"] += 1
    
    async def scan(self) -> RadarScan:
        """Execute simulated radar scan."""
        self.scan_count += 1
        scan_id = str(uuid.uuid4())
        now = datetime.utcnow()
        self.last_scan_time = now
        
        # Update positions of active targets
        await self._update_target_positions()
        
        # Generate some new targets
        num_new = await self._generate_new_targets()
        
        # Ensure minimum targets
        total_targets = max(1, len(self.active_targets))
        
        radar_data = RadarScan(
            scan_id=scan_id,
            timestamp=now,
            frame_count=512,
            targets_detected=total_targets,
            signal_strength=np.random.uniform(0.7, 1.0),
            noise_level=np.random.uniform(0.05, 0.15)
        )
        
        radar_logger.log_event(
            "scan_complete",
            "radar_simulator",
            {
                "scan_id": scan_id,
                "targets": total_targets,
                "new_targets": num_new,
                "active_count": len(self.active_targets)
            },
            level="INFO"
        )
        
        return radar_data
    
    async def get_targets_from_scan(self, scan_id: str) -> List[RadarTarget]:
        """Extract simulated targets from scan."""
        targets = []
        
        for target_id, target_data in self.active_targets.items():
            # Calculate range and bearing from position
            range_m = np.sqrt(target_data["x"]**2 + target_data["y"]**2) * 100 + np.random.uniform(100, 1000)
            bearing_deg = np.degrees(np.arctan2(target_data["y"], target_data["x"])) % 360
            velocity_mps = np.sqrt(target_data["vx"]**2 + target_data["vy"]**2) * 10
            
            target = RadarTarget(
                id=target_id,
                range_m=range_m,
                bearing_deg=bearing_deg,
                velocity_mps=velocity_mps,
                rcs_dbsm=np.random.uniform(-15, 10),
                signal_strength=np.clip(target_data["strength"], 0.4, 1.0),
                confidence=0.8 + (target_data["hits"] * 0.02),
                timestamp=datetime.utcnow()
            )
            targets.append(target)
        
        # Ensure at least 1 target
        if not targets:
            self.target_counter += 1
            target_type = np.random.choice(self.TARGET_TYPES)
            target_name = f"{target_type}-{self.target_counter}"
            target_id = target_name
            
            self.active_targets[target_id] = {
                "name": target_name,
                "type": target_type,
                "x": np.random.uniform(-200, 200),
                "y": np.random.uniform(-200, 200),
                "vx": np.random.uniform(-20, 20),
                "vy": np.random.uniform(-20, 20),
                "strength": np.random.uniform(0.6, 0.95),
                "created_at": time.time(),
                "hits": 1
            }
            
            target = RadarTarget(
                id=target_id,
                range_m=np.random.uniform(1000, 50000),
                bearing_deg=np.random.uniform(0, 360),
                velocity_mps=np.random.uniform(0, 100),
                rcs_dbsm=np.random.uniform(-10, 15),
                signal_strength=0.85,
                confidence=0.85,
                timestamp=datetime.utcnow()
            )
            targets.append(target)
        
        radar_logger.log_event(
            "targets_extracted",
            "radar_simulator",
            {"scan_id": scan_id, "target_count": len(targets)},
            level="INFO"
        )
        
        return targets
    
    async def get_signal_quality(self) -> Dict[str, float]:
        """Get simulated signal quality metrics."""
        return {
            "snr_db": np.random.uniform(15, 35),
            "noise_floor_dbm": np.random.uniform(-95, -85),
            "peak_signal_dbm": np.random.uniform(-30, 10),
            "system_temperature_k": 290.0 + np.random.uniform(-5, 5)
        }
    
    def get_active_target_count(self) -> int:
        """Get number of active simulated targets."""
        return len(self.active_targets)
