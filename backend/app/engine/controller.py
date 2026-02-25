"""Main controller - async event loop for radar/detection pipeline."""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from app.core.config import get_config
from app.core.logging import pipeline_logger
from app.engine.event_bus import event_bus, Events
from app.engine.pipeline import Pipeline


class RadarController:
    """
    Main event loop controller.
    
    Runs radar scans on regular interval and executes detection pipeline.
    """
    
    def __init__(
        self,
        radar_service,
        detection_service,
        tracking_service,
        threat_service,
        ew_service
    ):
        self.config = get_config()
        self.radar_service = radar_service
        self.detection_service = detection_service
        self.tracking_service = tracking_service
        self.threat_service = threat_service
        self.ew_service = ew_service
        
        self.pipeline = Pipeline(
            radar_service,
            detection_service,
            tracking_service,
            threat_service,
            ew_service
        )
        self.running = False
        self.scan_interval = self.config.radar_scan_interval
        self.task: Optional[asyncio.Task] = None
        self.cycle_count = 0
        self.startup_time = None
        self.initialization_status = "PENDING"
        self.initialization_errors: List[str] = []
    
    async def initialize(self) -> bool:
        """Initialize and verify all services are ready.
        
        Returns True if all services initialized successfully.
        """
        pipeline_logger.info("─" * 60)
        pipeline_logger.info("SERVICE INITIALIZATION")
        pipeline_logger.info("─" * 60)
        
        try:
            # Initialize detection service
            pipeline_logger.info("Initializing detection service...")
            if hasattr(self.detection_service, 'initialize'):
                if not await self.detection_service.initialize():
                    err = self.detection_service.initialization_error or "Unknown error"
                    self.initialization_errors.append(f"Detection: {err}")
                    pipeline_logger.warning(f"  ⚠ Detection service init failed: {err}")
            
            # Initialize tracking service
            pipeline_logger.info("Initializing tracking service...")
            if hasattr(self.tracking_service, 'initialize'):
                if not await self.tracking_service.initialize():
                    err = self.tracking_service.initialization_error or "Unknown error"
                    self.initialization_errors.append(f"Tracking: {err}")
                    pipeline_logger.warning(f"  ⚠ Tracking service init failed: {err}")
            
            # Initialize threat service
            pipeline_logger.info("Initializing threat service...")
            if hasattr(self.threat_service, 'initialize'):
                if not await self.threat_service.initialize():
                    err = self.threat_service.initialization_error or "Unknown error"
                    self.initialization_errors.append(f"Threat: {err}")
                    pipeline_logger.warning(f"  ⚠ Threat service init failed: {err}")
            
            # Initialize EW service
            pipeline_logger.info("Initializing EW service...")
            if hasattr(self.ew_service, 'initialize'):
                if not await self.ew_service.initialize():
                    err = self.ew_service.initialization_error or "Unknown error"
                    self.initialization_errors.append(f"EW: {err}")
                    pipeline_logger.warning(f"  ⚠ EW service init failed: {err}")
            
            # Check radar service
            pipeline_logger.info("✓ Radar service ready")
            
            if self.initialization_errors:
                pipeline_logger.warning("Some services failed initialization, but continuing with fallbacks")
                self.initialization_status = "PARTIAL"
            else:
                self.initialization_status = "READY"
            
            pipeline_logger.info("─" * 60)
            pipeline_logger.info("✓ INITIALIZATION COMPLETE")
            pipeline_logger.info("─" * 60)
            
            return True
            
        except Exception as e:
            self.initialization_status = "FAILED"
            self.initialization_errors.append(f"Critical: {str(e)}")
            pipeline_logger.error(f"Initialization failed: {e}")
            return False
    
    async def start(self) -> None:
        """Start the controller."""
        if self.running:
            return
        
        # Initialize services first
        if not await self.initialize():
            raise RuntimeError("Failed to initialize services")
        
        self.running = True
        self.startup_time = datetime.utcnow()
        self.task = asyncio.create_task(self._run_loop())
        
        pipeline_logger.info("✓ RadarController started")
    
    async def stop(self) -> None:
        """Stop the controller gracefully."""
        self.running = False
        
        if self.task:
            try:
                await asyncio.wait_for(self.task, timeout=5.0)
            except asyncio.TimeoutError:
                self.task.cancel()
        
        pipeline_logger.info("✓ RadarController stopped")
    
    async def _run_loop(self) -> None:
        """
        Main event loop.
        
        while True:
            execute pipeline cycle
            sleep scan_interval
        """
        pipeline_logger.info(f"Pipeline loop started (interval: {self.scan_interval}s)")
        
        try:
            while self.running:
                try:
                    # Execute complete pipeline
                    result = await self.pipeline.execute_cycle()
                    self.cycle_count += 1
                    
                    # Publish cycle complete metrics
                    await event_bus.publish(Events.BROADCAST_SYSTEM_STATUS, {
                        "cycle_count": self.cycle_count,
                        "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds(),
                        "success": result.get("success", False),
                        "initialization_status": self.initialization_status
                    })
                    
                except Exception as e:
                    pipeline_logger.error(f"Cycle error: {e}")
                    await asyncio.sleep(0.1)  # Brief delay before retry
                    continue
                
                # Sleep until next scan
                await asyncio.sleep(self.scan_interval)
        
        except asyncio.CancelledError:
            pipeline_logger.info("Pipeline loop cancelled")
        except Exception as e:
            pipeline_logger.error(f"Pipeline loop fatal error: {e}")
            self.running = False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get controller status for health checks."""
        return {
            "running": self.running,
            "cycle_count": self.cycle_count,
            "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds() if self.startup_time else 0,
            "initialization_status": self.initialization_status,
            "initialization_errors": self.initialization_errors,
            "scan_interval": self.scan_interval,
            "last_result": getattr(self.pipeline, 'last_result', None)
        }
    
    async def get_status(self) -> dict:
        """Get controller status."""
        uptime = None
        if self.startup_time:
            uptime = (datetime.utcnow() - self.startup_time).total_seconds()
        
        return {
            "running": self.running,
            "cycle_count": self.cycle_count,
            "scan_interval_s": self.scan_interval,
            "uptime_seconds": uptime,
            "pipeline_status": await self.pipeline.get_status()
        }


# Global controller instance
_controller: Optional[RadarController] = None


def get_controller(
    radar_service=None,
    detection_service=None,
    tracking_service=None,
    threat_service=None,
    ew_service=None
) -> RadarController:
    """Get or create controller instance (lazy initialization)."""
    global _controller
    if _controller is None:
        if any(s is None for s in [radar_service, detection_service, tracking_service, threat_service, ew_service]):
            raise ValueError("All services required on first initialization")
        _controller = RadarController(
            radar_service,
            detection_service,
            tracking_service,
            threat_service,
            ew_service
        )
    return _controller
