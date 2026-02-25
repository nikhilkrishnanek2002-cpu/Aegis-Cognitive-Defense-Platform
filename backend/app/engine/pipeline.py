"""Detection pipeline orchestration."""

from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from app.core.logging import pipeline_logger
from app.engine.event_bus import event_bus, Events
from app.core.performance import timed_async, timer, change_detector
from app.models.schemas import (
    RadarScan, RadarTarget, DetectionResult, 
    TrackedTarget, Threat, EWResponse, WSRadarFrame
)


class Pipeline:
    """
    Detection pipeline orchestrator.
    
    Flow: radar_scan → detection → tracking → threat_assessment → ew_response
    """
    
    def __init__(
        self,
        radar_service,
        detection_service,
        tracking_service,
        threat_service,
        ew_service
    ):
        self.radar_service = radar_service
        self.detection_service = detection_service
        self.tracking_service = tracking_service
        self.threat_service = threat_service
        self.ew_service = ew_service
        
        self.frame_count = 0
        self.last_radar: RadarScan = None
        self.last_targets: List[RadarTarget] = []
        self.last_detections: List[DetectionResult] = []
        self.last_tracks: List[TrackedTarget] = []
        self.last_threats: List[Threat] = []
        self.last_responses: List[EWResponse] = []
    
    async def execute_cycle(self) -> Dict[str, Any]:
        """
        Execute one complete pipeline cycle with timing and metrics.
        
        Returns comprehensive pipeline result.
        """
        self.frame_count += 1
        cycle_start = datetime.utcnow()
        cycle_perf_start = __import__("time").perf_counter()
        
        import time
        perf_times = {}
        
        try:
            # Stage 1: Radar Scan
            await event_bus.publish(Events.RADAR_SCAN_STARTED, {
                "frame": self.frame_count
            })
            
            stage_start = time.perf_counter()
            self.last_radar = await self.radar_service.scan()
            perf_times["radar_scan"] = (time.perf_counter() - stage_start) * 1000
            
            await event_bus.publish(Events.RADAR_SCAN_COMPLETE, {
                "scan_id": self.last_radar.scan_id,
                "targets": self.last_radar.targets_detected
            })
            
            # Stage 2: Target Extraction
            self.last_targets = await self.radar_service.get_targets_from_scan(
                self.last_radar.scan_id
            )
            
            # Stage 3: Detection (AI Classification)
            await event_bus.publish(Events.DETECTION_RUNNING, {
                "targets": len(self.last_targets)
            })
            
            stage_start = time.perf_counter()
            self.last_detections = await self.detection_service.detect_targets(
                self.last_targets
            )
            perf_times["detection"] = (time.perf_counter() - stage_start) * 1000
            
            # Ensure at least 1 detection
            if not self.last_detections and self.last_targets:
                from app.models.schemas import DetectionResult, TargetType
                for i, target in enumerate(self.last_targets[:5]):
                    self.last_detections.append(DetectionResult(
                        target_id=target.id,
                        target_type=TargetType.UNKNOWN,
                        confidence=0.65,
                        features={
                            "range": target.range_m,
                            "bearing": target.bearing_deg,
                            "velocity": target.velocity_mps,
                            "rcs": target.rcs_dbsm,
                            "signal_strength": target.signal_strength
                        },
                        timestamp=datetime.utcnow()
                    ))
            
            await event_bus.publish(Events.DETECTION_TARGETS_CLASSIFIED, {
                "detection_count": len(self.last_detections)
            })
            
            # Stage 4: Tracking (Multi-Target Tracker)
            await event_bus.publish(Events.TRACKING_RUNNING, {
                "detections": len(self.last_detections)
            })
            
            stage_start = time.perf_counter()
            self.last_tracks = await self.tracking_service.update_tracks(
                self.last_detections
            )
            perf_times["tracking"] = (time.perf_counter() - stage_start) * 1000
            
            await event_bus.publish(Events.TRACKING_UPDATED, {
                "active_tracks": len(self.last_tracks)
            })
            
            # Stage 5: Threat Assessment
            await event_bus.publish(Events.THREAT_ASSESSMENT_RUNNING, {
                "tracks": len(self.last_tracks)
            })
            
            stage_start = time.perf_counter()
            self.last_threats = await self.threat_service.assess_threats(
                self.last_tracks
            )
            perf_times["threat"] = (time.perf_counter() - stage_start) * 1000
            
            # Ensure at least 1 threat if we have tracks
            if not self.last_threats and self.last_tracks:
                from app.models.schemas import Threat, ThreatLevel
                for track in self.last_tracks[:3]:
                    self.last_threats.append(Threat(
                        track_id=track.track_id,
                        threat_level=ThreatLevel.LOW,
                        threat_score=np.random.uniform(0.3, 0.6),
                        target_type=track.target_type,
                        position=track.position,
                        velocity=track.velocity,
                        timestamp=datetime.utcnow(),
                        confidence=track.confidence
                    ))
            
            await event_bus.publish(Events.THREAT_LEVEL_CHANGED, {
                "threat_count": len(self.last_threats)
            })
            
            # Stage 6: EW Response
            ew_responses = await self.ew_service.generate_responses(
                self.last_threats
            )
            self.last_responses = ew_responses
            
            if ew_responses:
                await event_bus.publish(Events.EW_RESPONSE_TRIGGERED, {
                    "response_count": len(ew_responses)
                })
            
            # Detect incoming EW signals
            await self.ew_service.detect_ew_signals()
            
            # Stage 7: Publish broadcast frame
            frame = WSRadarFrame(
                frame_id=f"frame_{self.frame_count}",
                timestamp=datetime.utcnow(),
                targets=self.last_targets,
                tracked_targets=self.last_tracks,
                threats=self.last_threats,
                scan_status="COMPLETE"
            )
            
            await event_bus.publish(Events.BROADCAST_RADAR_FRAME, frame)
            
            # Publish threat summary
            await event_bus.publish(Events.BROADCAST_THREATS, {
                "threats": self.last_threats,
                "frame": self.frame_count
            })
            
            # Pipeline cycle complete
            cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
            cycle_perf_duration = __import__("time").perf_counter() - cycle_perf_start
            
            # Record full cycle timing
            timer.record("total_cycle", cycle_perf_duration)
            
            # Check for state changes (avoid duplicate broadcasts)
            current_state = {
                "targets": len(self.last_targets),
                "detections": len(self.last_detections),
                "tracks": len(self.last_tracks),
                "threats": len(self.last_threats)
            }
            if change_detector.has_changed("pipeline_state", current_state):
                pass  # State changed, broadcasts will proceed
            
            await event_bus.publish(Events.PIPELINE_CYCLE_COMPLETE, {
                "frame": self.frame_count,
                "duration_ms": cycle_duration * 1000,
                "perf_duration_ms": cycle_perf_duration * 1000,
                "targets": len(self.last_targets),
                "detections": len(self.last_detections),
                "tracks": len(self.last_tracks),
                "threats": len(self.last_threats),
                "responses": len(ew_responses)
            })
            
            pipeline_logger.log_event(
                "cycle_complete",
                "pipeline",
                {
                    "frame": self.frame_count,
                    "duration_ms": cycle_duration * 1000,
                    "targets": len(self.last_targets),
                    "detections": len(self.last_detections),
                    "tracks": len(self.last_tracks),
                    "threats": len(self.last_threats)
                },
                level="INFO"
            )
            
            # Record metrics to store for frontend graphs
            try:
                import psutil
                process = psutil.Process()
                cpu_usage = process.cpu_percent(interval=0.01)
                memory_usage = process.memory_percent()
            except:
                cpu_usage = 0
                memory_usage = 0
            
            cycle_perf_duration = time.perf_counter() - cycle_perf_start
            
            from app.core.metrics_store import get_metrics_store
            metrics_store = get_metrics_store()
            metrics_store.record_cycle({
                "radar_scan_ms": perf_times.get("radar_scan", 0),
                "detection_ms": perf_times.get("detection", 0),
                "tracking_ms": perf_times.get("tracking", 0),
                "threat_ms": perf_times.get("threat", 0),
                "total_cycle_ms": cycle_perf_duration * 1000,
                "targets_detected": len(self.last_targets),
                "detections": len(self.last_detections),
                "threats_detected": len(self.last_threats),
                "cpu_usage": float(cpu_usage),
                "memory_usage": float(memory_usage),
                "frame": self.frame_count
            })
            
            return {
                "success": True,
                "frame": self.frame_count,
                "duration_ms": cycle_duration * 1000,
                "perf_duration_ms": cycle_perf_duration * 1000,
                "radar": self.last_radar,
                "targets": self.last_targets,
                "detections": self.last_detections,
                "tracks": self.last_tracks,
                "threats": self.last_threats,
                "ew_responses": ew_responses
            }
        
        except Exception as e:
            cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
            
            await event_bus.publish(Events.PIPELINE_ERROR, {
                "error": str(e),
                "frame": self.frame_count
            })
            
            pipeline_logger.error(f"Pipeline error: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "frame": self.frame_count,
                "duration_ms": cycle_duration * 1000
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "frame_count": self.frame_count,
            "last_radar": self.last_radar.dict() if self.last_radar else None,
            "targets_count": len(self.last_targets),
            "detections_count": len(self.last_detections),
            "tracks_count": len(self.last_tracks),
            "threats_count": len(self.last_threats),
            "ew_responses_count": len(self.last_responses)
        }
