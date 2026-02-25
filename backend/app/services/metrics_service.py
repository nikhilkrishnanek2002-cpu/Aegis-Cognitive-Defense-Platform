"""Metrics collection and reporting service."""

import psutil
from datetime import datetime
from typing import Dict, Any
from app.core.logging import pipeline_logger


class MetricsCollector:
    """Collect system and pipeline metrics."""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.process = psutil.Process()
        self.total_cycles = 0
        self.total_detections = 0
        self.total_threats = 0
    
    def record_cycle(self, cycle_data: Dict[str, Any]):
        """Record metrics from a pipeline cycle."""
        self.total_cycles += 1
        self.total_detections += cycle_data.get("detections", 0)
        self.total_threats += cycle_data.get("threats", 0)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
            mem_info = self.process.memory_info()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_mb": mem_info.rss / (1024 * 1024),
                "memory_percent": self.process.memory_percent(),
                "num_threads": self.process.num_threads(),
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
            }
        except Exception as e:
            pipeline_logger.error(f"Failed to collect system metrics: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu_percent": 0,
                "memory_mb": 0,
                "memory_percent": 0,
                "num_threads": 0,
                "uptime_seconds": 0,
                "error": str(e)
            }
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline-level metrics."""
        return {
            "total_cycles": self.total_cycles,
            "total_detections": self.total_detections,
            "total_threats": self.total_threats,
            "avg_detections_per_cycle": self.total_detections / max(1, self.total_cycles),
            "avg_threats_per_cycle": self.total_threats / max(1, self.total_cycles)
        }
    
    def get_full_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": self.get_system_metrics(),
            "pipeline": self.get_pipeline_metrics()
        }


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector singleton."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
