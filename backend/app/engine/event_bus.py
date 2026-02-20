"""Event bus for publish-subscribe pattern."""

from typing import Callable, Dict, List, Any
import asyncio


class EventBus:
    """
    Central event bus for decoupled pub/sub communication.
    
    Handlers are called sequentially for each event.
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()
    
    async def subscribe(self, event_name: str, handler: Callable) -> None:
        """Register a handler for an event."""
        async with self._lock:
            if event_name not in self._subscribers:
                self._subscribers[event_name] = []
            self._subscribers[event_name].append(handler)
    
    async def unsubscribe(self, event_name: str, handler: Callable) -> None:
        """Unregister a handler."""
        async with self._lock:
            if event_name in self._subscribers:
                self._subscribers[event_name].remove(handler)
    
    async def publish(self, event_name: str, payload: Any = None) -> None:
        """
        Publish an event to all subscribers.
        
        Calls handlers sequentially. If a handler is async, awaits it.
        """
        async with self._lock:
            handlers = self._subscribers.get(event_name, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(payload)
                else:
                    handler(payload)
            except Exception as e:
                # Log but don't fail the entire event bus
                print(f"Error in handler for {event_name}: {e}")
    
    async def publish_async(self, event_name: str, payload: Any = None) -> None:
        """Publish event and return immediately (fire and forget)."""
        asyncio.create_task(self.publish(event_name, payload))
    
    def get_subscribers_count(self, event_name: str) -> int:
        """Get number of subscribers for an event."""
        return len(self._subscribers.get(event_name, []))


# Global event bus instance
event_bus = EventBus()


# Event names - define all events here for clarity
class Events:
    """Event name constants."""
    
    # Radar events
    RADAR_SCAN_STARTED = "radar:scan_started"
    RADAR_SCAN_COMPLETE = "radar:scan_complete"
    RADAR_TARGETS_DETECTED = "radar:targets_detected"
    
    # Detection events
    DETECTION_RUNNING = "detection:running"
    DETECTION_TARGETS_CLASSIFIED = "detection:targets_classified"
    DETECTION_ERROR = "detection:error"
    
    # Tracking events
    TRACKING_RUNNING = "tracking:running"
    TRACKING_UPDATED = "tracking:updated"
    TRACKING_LOST = "tracking:lost"
    
    # Threat assessment events
    THREAT_ASSESSMENT_RUNNING = "threat:assessment_running"
    THREAT_LEVEL_CHANGED = "threat:level_changed"
    THREAT_CRITICAL = "threat:critical"
    
    # EW response events
    EW_DETECTION = "ew:detection"
    EW_RESPONSE_TRIGGERED = "ew:response_triggered"
    EW_COUNTERMEASURE = "ew:countermeasure"
    
    # Pipeline events
    PIPELINE_CYCLE_COMPLETE = "pipeline:cycle_complete"
    PIPELINE_ERROR = "pipeline:error"
    
    # Broadcast events
    BROADCAST_RADAR_FRAME = "broadcast:radar_frame"
    BROADCAST_THREATS = "broadcast:threats"
    BROADCAST_SYSTEM_STATUS = "broadcast:system_status"
