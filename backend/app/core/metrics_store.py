"""Metrics storage and management for live dashboards."""

import time
from datetime import datetime
from collections import deque
from typing import Dict, Any, List


class MetricsStore:
    """Store streaming metrics for frontend graphs."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.cycle_metrics: deque = deque(maxlen=max_history)
        self.latest_metrics: Dict[str, Any] = {}
    
    def record_cycle(self, cycle_data: Dict[str, Any]):
        """Record metrics from a pipeline cycle."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "unix_timestamp": time.time(),
            **cycle_data
        }
        
        self.cycle_metrics.append(entry)
        self.latest_metrics = entry
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics history for graphs."""
        return list(self.cycle_metrics)[-limit:]
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest cycle metrics."""
        if not self.latest_metrics:
            return self._empty_metrics()
        return self.latest_metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if not self.cycle_metrics:
            return self._empty_metrics()
        
        metrics = list(self.cycle_metrics)
        
        cycles_ms = [m.get("total_cycle_ms", 0) for m in metrics]
        detections = [m.get("targets_detected", 0) for m in metrics]
        threats = [m.get("threats_detected", 0) for m in metrics]
        
        return {
            "latest_cycle_ms": cycles_ms[-1] if cycles_ms else 0,
            "avg_cycle_ms": sum(cycles_ms) / len(cycles_ms) if cycles_ms else 0,
            "min_cycle_ms": min(cycles_ms) if cycles_ms else 0,
            "max_cycle_ms": max(cycles_ms) if cycles_ms else 0,
            "avg_detections": sum(detections) / len(detections) if detections else 0,
            "avg_threats": sum(threats) / len(threats) if threats else 0,
            "total_cycles": len(metrics),
            "cpu_usage_avg": np.mean([m.get("cpu_usage", 0) for m in metrics]),
            "memory_usage_avg": np.mean([m.get("memory_usage", 0) for m in metrics])
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "unix_timestamp": time.time(),
            "radar_scan_ms": 0,
            "detection_ms": 0,
            "tracking_ms": 0,
            "threat_ms": 0,
            "total_cycle_ms": 0,
            "targets_detected": 0,
            "threats_detected": 0,
            "cpu_usage": 0,
            "memory_usage": 0
        }


# Global metrics store
_metrics_store = None


def get_metrics_store() -> MetricsStore:
    """Get metrics store singleton."""
    global _metrics_store
    if _metrics_store is None:
        _metrics_store = MetricsStore()
    return _metrics_store


import numpy as np
