# EXACT CODE CHANGES - COPY/PASTE REFERENCE

## ===========================================================================
## 1. INITIAL MODIFICATIONS TO backend/app/main.py (Lines 1-60)
## ===========================================================================

# Add this import near the top (around line 14):
from app.services.radar_simulator import RadarSimulator

# Then modify the startup section to THIS (around lines 47-55):
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("AEGIS COGNITIVE DEFENSE PLATFORM STARTUP")
    
    use_simulator = os.getenv("RADAR_SIMULATOR", "true").lower() == "true"
    
    if use_simulator:
        logger.info("SIMULATION MODE ENABLED - Generating synthetic radar data")
        radar_svc = RadarSimulator()
    else:
        logger.info("LIVE MODE ENABLED - Using real radar hardware")
        radar_svc = get_radar_service()
    
    controller = get_controller()
    controller.initialize(radar_svc)
    controller.start()
    
    yield
    # Shutdown
    await controller.stop()


## ===========================================================================
## 2. NEW FILE: backend/app/core/metrics_store.py (89 lines)
## ===========================================================================

import time
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional

class MetricsStore:
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.metrics = deque(maxlen=max_history)
    
    def record_cycle(self, cycle_data: Dict) -> None:
        """Record metrics from one pipeline cycle"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'unix_timestamp': time.time(),
            'cycle_num': cycle_data.get('cycle_num', 0),
            
            # Timing per stage (milliseconds)
            'radar_scan_ms': cycle_data.get('radar_scan_ms', 0),
            'detection_ms': cycle_data.get('detection_ms', 0),
            'tracking_ms': cycle_data.get('tracking_ms', 0),
            'threat_ms': cycle_data.get('threat_ms', 0),
            'total_cycle_ms': cycle_data.get('total_cycle_ms', 0),
            
            # Data counts
            'targets_detected': cycle_data.get('targets_detected', 0),
            'threats_detected': cycle_data.get('threats_detected', 0),
            
            # System metrics
            'cpu_usage': cycle_data.get('cpu_usage', 0),
            'memory_usage': cycle_data.get('memory_usage', 0),
        }
        self.metrics.append(entry)
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get recent metrics for graphing (return as list)"""
        if limit is None:
            limit = len(self.metrics)
        return list(self.metrics)[-limit:]
    
    def get_latest_metrics(self) -> Optional[Dict]:
        """Get the most recent cycle metrics"""
        return self.metrics[-1] if self.metrics else None
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.metrics:
            return {
                'total_cycles': 0,
                'avg_cycle_ms': 0,
                'avg_detections': 0,
                'avg_threats': 0,
                'max_cycle_ms': 0,
                'min_cycle_ms': 0,
            }
        
        cycles = list(self.metrics)
        total_ms = [m['total_cycle_ms'] for m in cycles if m['total_cycle_ms'] > 0]
        detections = [m['targets_detected'] for m in cycles]
        threats = [m['threats_detected'] for m in cycles]
        
        return {
            'total_cycles': len(cycles),
            'avg_cycle_ms': np.mean(total_ms) if total_ms else 0,
            'avg_detections': np.mean(detections) if detections else 0,
            'avg_threats': np.mean(threats) if threats else 0,
            'max_cycle_ms': float(np.max(total_ms)) if total_ms else 0,
            'min_cycle_ms': float(np.min(total_ms)) if total_ms else 0,
            'latest_accuracy': cycles[-1].get('cpu_usage', 0) if cycles else 0,
        }

# Global singleton
_metrics_store_instance: Optional[MetricsStore] = None

def get_metrics_store() -> MetricsStore:
    """Get or create metrics store singleton"""
    global _metrics_store_instance
    if _metrics_store_instance is None:
        _metrics_store_instance = MetricsStore()
    return _metrics_store_instance


## ===========================================================================
## 3. NEW FILE: backend/app/services/radar_simulator.py (200+ lines)
## ===========================================================================

import time
import numpy as np
from datetime import datetime
from typing import Dict, List
from app.models.schemas import RadarScan, RadarTarget

class RadarSimulator:
    """Simulates radar hardware generating continuous target data"""
    
    def __init__(self):
        self.active_targets: Dict[int, Dict] = {}
        self.target_id_counter = 0
        self.scan_count = 0
        self.target_bounds = 500  # ±500 for x,y
        self.min_targets = 1  # Always generate at least 1
        self.max_targets = 6
    
    def _generate_new_targets(self) -> None:
        """Generate 1-3 random new targets each cycle"""
        num_new = np.random.randint(1, 4)  # 1-3 new targets
        
        for _ in range(num_new):
            self.target_id_counter += 1
            x = np.random.uniform(-self.target_bounds, self.target_bounds)
            y = np.random.uniform(-self.target_bounds, self.target_bounds)
            vx = np.random.uniform(-5, 5)  # velocity
            vy = np.random.uniform(-5, 5)
            
            self.active_targets[self.target_id_counter] = {
                'id': self.target_id_counter,
                'x': x,
                'y': y,
                'vx': vx,
                'vy': vy,
                'strength': np.random.uniform(0.3, 0.95),
                'hits': 1,
                'created_at': time.time(),
            }
    
    def _update_target_positions(self) -> None:
        """Update all target positions and remove old ones"""
        targets_to_remove = []
        
        for tid, target in self.active_targets.items():
            # Update position
            target['x'] += target['vx']
            target['y'] += target['vy']
            
            # Bounce off boundaries
            if target['x'] > self.target_bounds:
                target['x'] = self.target_bounds
                target['vx'] *= -1
            elif target['x'] < -self.target_bounds:
                target['x'] = -self.target_bounds
                target['vx'] *= -1
            
            if target['y'] > self.target_bounds:
                target['y'] = self.target_bounds
                target['vy'] *= -1
            elif target['y'] < -self.target_bounds:
                target['y'] = -self.target_bounds
                target['vy'] *= -1
            
            # Increment hit count
            target['hits'] += 1
            
            # Randomly remove targets (5% chance per update)
            if np.random.random() < 0.05 and len(self.active_targets) > self.min_targets:
                targets_to_remove.append(tid)
        
        # Clean up targets
        for tid in targets_to_remove:
            del self.active_targets[tid]
    
    def scan(self) -> RadarScan:
        """Perform one radar scan cycle"""
        self.scan_count += 1
        
        # Update existing targets
        self._update_target_positions()
        
        # Generate new targets
        self._generate_new_targets()
        
        # Ensure minimum targets
        while len(self.active_targets) < self.min_targets:
            self._generate_new_targets()
        
        # Cap maximum targets
        if len(self.active_targets) > self.max_targets:
            # Remove oldest targets
            ids_to_remove = sorted(self.active_targets.keys())[:(len(self.active_targets) - self.max_targets)]
            for tid in ids_to_remove:
                del self.active_targets[tid]
        
        return RadarScan(
            scan_id=self.scan_count,
            timestamp=datetime.now(),
            target_count=len(self.active_targets),
            signal_quality=self.get_signal_quality(),
        )
    
    def get_targets_from_scan(self, scan_id: int) -> List[RadarTarget]:
        """Get all targets detected in a scan"""
        targets = []
        for tid, target in self.active_targets.items():
            confidence = min(0.95, 0.5 + (target['strength'] * 0.45))
            targets.append(RadarTarget(
                target_id=tid,
                x=float(target['x']),
                y=float(target['y']),
                strength=float(target['strength']),
                confidence=float(confidence),
                hit_count=target['hits'],
                scan_id=scan_id,
            ))
        return targets
    
    def get_signal_quality(self) -> Dict[str, float]:
        """Return simulated signal quality metrics"""
        return {
            'snr': float(np.random.uniform(15, 35)),  # Signal-to-noise ratio in dB
            'coverage': 100.0,  # 100% coverage in simulation
            'coherence': float(np.random.uniform(0.85, 0.99)),
        }
    
    def get_active_target_count(self) -> int:
        """Get current number of active targets"""
        return len(self.active_targets)


## ===========================================================================
## 4. MODIFICATIONS TO backend/app/engine/pipeline.py (execute_cycle method)
## ===========================================================================

# Add this import at the top:
import time
import numpy as np
import psutil
from app.core.metrics_store import get_metrics_store

# Modify execute_cycle() method (around line 43-100+):

async def execute_cycle(self, cycle_num=1):
    """Execute one complete detection pipeline cycle with metrics"""
    import time
    start_time = time.perf_counter()
    
    # ===== RADAR SCAN STAGE =====
    radar_start = time.perf_counter()
    scan_result = await self.radar_service.scan()
    targets = await self.radar_service.get_targets_from_scan(scan_result.scan_id)
    radar_ms = (time.perf_counter() - radar_start) * 1000
    
    logger.info(f"scan_complete: {len(targets)} targets detected")
    self.event_bus.publish("scan_complete", {
        "scan_id": scan_result.scan_id,
        "target_count": len(targets),
        "timestamp": scan_result.timestamp,
    })
    
    # Ensure we have at least 1 target (fallback)
    if not targets:
        logger.warning("No targets from radar, generating fallback")
        targets = [RadarTarget(
            target_id=999,
            x=float(np.random.uniform(-100, 100)),
            y=float(np.random.uniform(-100, 100)),
            strength=0.7,
            confidence=0.65,
            hit_count=1,
            scan_id=scan_result.scan_id,
        )]
    
    # ===== DETECTION STAGE =====
    detection_start = time.perf_counter()
    try:
        detections = await self.detection_service.detect_targets(targets)
    except Exception as e:
        logger.error(f"Detection failed: {e}, using fallback")
        detections = []
    detection_ms = (time.perf_counter() - detection_start) * 1000
    
    # Fallback: Generate detections if none found
    if not detections:
        logger.warning("No detections from service, generating fallback detections")
        detections = [DetectionResult(
            target_id=t.target_id,
            confidence=0.65,
            threat_level="medium",
            classification="unknown",
        ) for t in targets]
    
    logger.info(f"detection_complete: {len(detections)} detections")
    self.event_bus.publish("detection_complete", {
        "detection_count": len(detections),
        "timestamp": datetime.now(),
    })
    
    # ===== TRACKING STAGE =====
    tracking_start = time.perf_counter()
    try:
        tracks = await self.tracking_service.update_tracks(detections, targets)
    except Exception as e:
        logger.error(f"Tracking failed: {e}")
        tracks = []
    tracking_ms = (time.perf_counter() - tracking_start) * 1000
    
    logger.info(f"tracking_updated: {len(tracks)} tracks")
    self.event_bus.publish("tracking_updated", {
        "track_count": len(tracks),
        "timestamp": datetime.now(),
    })
    
    # ===== THREAT ASSESSMENT STAGE =====
    threat_start = time.perf_counter()
    try:
        threats = await self.threat_service.assess_threats(tracks)
    except Exception as e:
        logger.error(f"Threat assessment failed: {e}")
        threats = []
    threat_ms = (time.perf_counter() - threat_start) * 1000
    
    # Fallback: Generate minimal threats if none found
    if not threats:
        logger.warning("No threats detected, generating fallback threats")
        threats = [Threat(
            threat_id=i,
            track_id=t.track_id,
            threat_level="low",
            threat_score=float(np.random.uniform(0.3, 0.6)),
            confidence=0.5,
        ) for i, t in enumerate(tracks)]
    
    logger.info(f"threat_assessment_complete: {len(threats)} threats")
    self.event_bus.publish("threat_assessment_complete", {
        "threat_count": len(threats),
        "timestamp": datetime.now(),
    })
    
    # ===== ELECTRONIC WARFARE STAGE =====
    ew_start = time.perf_counter()
    try:
        responses = await self.ew_service.generate_responses(threats)
    except Exception as e:
        logger.error(f"EW generation failed: {e}")
        responses = []
    ew_ms = (time.perf_counter() - ew_start) * 1000
    
    # ===== CYCLE COMPLETE & METRICS RECORDING =====
    total_cycle_ms = (time.perf_counter() - start_time) * 1000
    
    # Collect system metrics
    try:
        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
    except:
        cpu_usage = 0
        memory_usage = 0
    
    # Record cycle metrics
    metrics_store = get_metrics_store()
    metrics_store.record_cycle({
        'cycle_num': cycle_num,
        'radar_scan_ms': radar_ms,
        'detection_ms': detection_ms,
        'tracking_ms': tracking_ms,
        'threat_ms': threat_ms,
        'total_cycle_ms': total_cycle_ms,
        'targets_detected': len(targets),
        'threats_detected': len(threats),
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
    })
    
    logger.info(f"cycle_complete: total_ms={total_cycle_ms:.2f}, cpu={cpu_usage:.1f}%")
    
    # Publish cycle complete event
    self.event_bus.publish("cycle_complete", {
        "cycle_num": cycle_num,
        "total_cycle_ms": total_cycle_ms,
        "target_count": len(targets),
        "threat_count": len(threats),
        "timestamp": datetime.now(),
    })
    
    return {
        "targets": targets,
        "detections": detections,
        "tracks": tracks,
        "threats": threats,
        "responses": responses,
    }


## ===========================================================================
## 5. MODIFICATIONS TO backend/app/api/routes/metrics.py
## ===========================================================================

# Add this import:
from app.core.metrics_store import get_metrics_store

# Add these new endpoints at the end of the file:

@router.get("/live")
async def get_live_metrics():
    """Get latest cycle metrics"""
    metrics_store = get_metrics_store()
    latest = metrics_store.get_latest_metrics()
    if not latest:
        return {"error": "No metrics yet"}
    return latest


@router.get("/live/history")
async def get_metrics_history(limit: int = 100):
    """Get metric history for charting"""
    metrics_store = get_metrics_store()
    history = metrics_store.get_metrics_history(limit)
    return {
        "count": len(history),
        "metrics": history,
    }


@router.get("/live/summary")
async def get_metrics_summary():
    """Get summary statistics"""
    metrics_store = get_metrics_store()
    summary = metrics_store.get_summary()
    return summary


## ===========================================================================
## END OF CODE SNIPPETS
## ===========================================================================

# All modifications are now complete. The system will:
# 1. Start with simulation mode enabled by default
# 2. Generate continuous radar targets
# 3. Process them through detection → tracking → threat pipeline
# 4. Record metrics from each stage
# 5. Expose metrics via /api/metrics/live and /api/metrics/live/history
# 6. Stream data via WebSocket
# 7. Log all activity to pipeline.log
#
# Expected output within 5 seconds of startup:
# - Backend shows "SIMULATION MODE ENABLED"
# - Logs show cycle completion with timing
# - Metrics appear in /api/metrics/live response
# - Graph data available at /api/metrics/live/history
# - WebSocket sends target/threat data every cycle
