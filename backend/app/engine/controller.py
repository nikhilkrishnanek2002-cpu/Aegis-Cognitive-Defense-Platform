"""Main controller - async event loop for radar/detection pipeline."""

import asyncio
from datetime import datetime
from typing import Optional
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
    
    async def start(self) -> None:
        """Start the controller."""
        if self.running:
            return
        
        self.running = True
        self.startup_time = datetime.utcnow()
        self.task = asyncio.create_task(self._run_loop())
        
        pipeline_logger.info("RadarController started")
    
    async def stop(self) -> None:
        """Stop the controller gracefully."""
        self.running = False
        
        if self.task:
            try:
                await asyncio.wait_for(self.task, timeout=5.0)
            except asyncio.TimeoutError:
                self.task.cancel()
        
        pipeline_logger.info("RadarController stopped")
    
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
                        "success": result.get("success", False)
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
    radar_service,
    detection_service,
    tracking_service,
    threat_service,
    ew_service
) -> RadarController:
    """Get or create controller instance."""
    global _controller
    if _controller is None:
        _controller = RadarController(
            radar_service,
            detection_service,
            tracking_service,
            threat_service,
            ew_service
        )
    return _controller
