"""Backend performance optimizations and utilities."""

import asyncio
import time
from functools import wraps
from typing import Callable, Any, Dict
import logging


# ─── Performance Instrumentation ──────────────────────────────────────────────

class PerformanceTimer:
    """Track performance metrics for pipeline stages."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {
            "radar_scan": [],
            "detection": [],
            "tracking": [],
            "threat_assessment": [],
            "ew_response": [],
            "websocket_send": [],
            "total_cycle": []
        }
    
    def record(self, stage: str, duration_ms: float):
        """Record stage duration."""
        self.metrics[stage].append(duration_ms)
        
        # Keep only last 100 measurements
        if len(self.metrics[stage]) > 100:
            self.metrics[stage].pop(0)
    
    def get_stats(self, stage: str) -> Dict[str, float]:
        """Get statistics for a stage."""
        times = self.metrics.get(stage, [])
        if not times:
            return {"avg": 0, "min": 0, "max": 0, "count": 0}
        
        return {
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "count": len(times),
            "latest": times[-1]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all stages."""
        return {stage: self.get_stats(stage) for stage in self.metrics.keys()}


# Global timer instance
timer = PerformanceTimer()


def timed_async(stage_name: str):
    """Decorator for timing async functions."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                timer.record(stage_name, duration_ms)
        return wrapper
    return decorator


def timed(stage_name: str):
    """Decorator for timing sync functions."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                timer.record(stage_name, duration_ms)
        return wrapper
    return decorator


# ─── Async Improvements ───────────────────────────────────────────────────────

def ensure_async(func: Callable) -> Callable:
    """Convert sync function to async if needed."""
    if asyncio.iscoroutinefunction(func):
        return func
    
    async def async_wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    return async_wrapper


# ─── Response Caching ──────────────────────────────────────────────────────

class SimpleCache:
    """In-memory cache with TTL."""
    
    def __init__(self, ttl_seconds: float = 5.0):
        self.ttl = ttl_seconds
        self.cache: Dict[str, tuple] = {}  # key -> (value, timestamp)
    
    def get(self, key: str) -> Any:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            elapsed = time.time() - timestamp
            
            if elapsed < self.ttl:
                return value
            else:
                del self.cache[key]
        
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        self.cache[key] = (value, time.time())
    
    def is_expired(self, key: str) -> bool:
        """Check if key has expired."""
        return self.get(key) is None


# Global caches
metrics_cache = SimpleCache(ttl_seconds=2.0)
status_cache = SimpleCache(ttl_seconds=3.0)
tracks_cache = SimpleCache(ttl_seconds=1.0)


# ─── WebSocket Optimization ────────────────────────────────────────────────

class BroadcastQueue:
    """Async queue for WebSocket broadcasts to avoid blocking."""
    
    def __init__(self, max_size: int = 1000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.dropped_count = 0
    
    async def put(self, message: Dict[str, Any]):
        """Add message to queue (drop if full)."""
        try:
            self.queue.put_nowait(message)
        except asyncio.QueueFull:
            self.dropped_count += 1
    
    async def get(self) -> Dict[str, Any]:
        """Get next message from queue."""
        return await self.queue.get()
    
    def qsize(self) -> int:
        """Get queue size."""
        return self.queue.qsize()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_size": self.qsize(),
            "dropped_messages": self.dropped_count
        }


# Global broadcast queue
broadcast_queue = BroadcastQueue()


# ─── JSON Optimization ────────────────────────────────────────────────────

def numpy_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_native(item) for item in obj]
    
    return obj


# ─── Duplicate Detection ────────────────────────────────────────────────────

class StateChangeDetector:
    """Detect if state actually changed to avoid unnecessary broadcasts."""
    
    def __init__(self):
        self.last_state = {}
    
    def has_changed(self, key: str, new_value: Any) -> bool:
        """Check if value changed."""
        old_value = self.last_state.get(key)
        
        if old_value != new_value:
            self.last_state[key] = new_value
            return True
        
        return False


change_detector = StateChangeDetector()
