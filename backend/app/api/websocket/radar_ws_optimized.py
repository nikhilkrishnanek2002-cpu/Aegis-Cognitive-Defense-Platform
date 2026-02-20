"""Optimized WebSocket handler with performance monitoring."""

import asyncio
import json
from typing import Set
from fastapi import WebSocket, WebSocketDisconnect
from app.core.logging import websocket_logger
from app.engine.event_bus import event_bus, Events
from app.core.performance import timed_async, timer, broadcast_queue, numpy_to_native
import time


# Track connected clients
connected_clients: Set[WebSocket] = set()
websocket_stats = {
    "connections": 0,
    "disconnections": 0,
    "messages_sent": 0,
    "messages_failed": 0
}


async def ws_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time radar stream.
    
    Optimizations:
    - Async JSON serialization
    - Non-blocking broadcasts
    - Heartbeat monitoring
    - State change detection
    """
    await websocket.accept()
    connected_clients.add(websocket)
    
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    websocket_stats["connections"] += 1
    
    websocket_logger.info(f"Client connected: {client_id}")
    
    try:
        # Subscribe to broadcast events with non-blocking handler
        await event_bus.subscribe(Events.BROADCAST_RADAR_FRAME, 
                                  lambda frame: asyncio.create_task(_send_safe(websocket, "radar_frame", frame)))
        
        await event_bus.subscribe(Events.BROADCAST_THREATS,
                                  lambda threats: asyncio.create_task(_send_safe(websocket, "threats", threats)))
        
        await event_bus.subscribe(Events.BROADCAST_SYSTEM_STATUS,
                                  lambda status: asyncio.create_task(_send_safe(websocket, "system_status", status)))
        
        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                
                if data:
                    try:
                        msg = json.loads(data)
                        await _handle_client_command(websocket, msg)
                    except json.JSONDecodeError:
                        pass
            
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": time.time()
                    })
                except Exception:
                    break
    
    except WebSocketDisconnect:
        websocket_logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        websocket_logger.error(f"WebSocket error: {e}")
    finally:
        websocket_stats["disconnections"] += 1
        connected_clients.discard(websocket)
        try:
            await websocket.close()
        except:
            pass


async def _send_safe(websocket: WebSocket, msg_type: str, data) -> None:
    """Send message safely with error handling."""
    try:
        start = time.perf_counter()
        
        # Convert to dict if needed
        if hasattr(data, 'dict'):
            data_dict = data.dict()
        elif isinstance(data, dict) and "threats" in data:
            # Handle threats payload
            threats = data.get("threats", [])
            data_dict = [t.dict() if hasattr(t, 'dict') else t for t in threats]
        else:
            data_dict = data
        
        # Convert numpy types
        data_dict = numpy_to_native(data_dict)
        
        message = {
            "type": msg_type,
            "data": data_dict,
            "timestamp": time.time()
        }
        
        await websocket.send_json(message)
        
        # Record timing
        duration_ms = (time.perf_counter() - start) * 1000
        timer.record("websocket_send", duration_ms)
        websocket_stats["messages_sent"] += 1
        
    except Exception as e:
        websocket_stats["messages_failed"] += 1
        websocket_logger.debug(f"Error sending {msg_type}: {e}")


async def _handle_client_command(websocket: WebSocket, command: dict) -> None:
    """Handle commands from client."""
    cmd_type = command.get("type")
    
    if cmd_type == "ping":
        await websocket.send_json({"type": "pong"})
    
    elif cmd_type == "subscribe":
        channel = command.get("channel")
        await websocket.send_json({
            "type": "subscription_confirmed",
            "channel": channel
        })
    
    elif cmd_type == "get_status":
        from app.engine.controller import _controller
        if _controller:
            status = await _controller.get_status()
            await websocket.send_json({
                "type": "status_response",
                "data": status
            })
    
    elif cmd_type == "get_performance":
        from app.core.performance import timer
        await websocket.send_json({
            "type": "performance_metrics",
            "data": timer.get_all_stats()
        })


async def broadcast_to_all(message: dict) -> None:
    """
    Broadcast message to all connected clients.
    Removes disconnected clients.
    """
    dead_clients = set()
    
    for websocket in connected_clients:
        try:
            await websocket.send_json(message)
        except Exception:
            dead_clients.add(websocket)
    
    for websocket in dead_clients:
        connected_clients.discard(websocket)


def get_connected_client_count() -> int:
    """Get number of connected WebSocket clients."""
    return len(connected_clients)


def get_websocket_stats() -> dict:
    """Get WebSocket statistics."""
    return {
        **websocket_stats,
        "active_clients": len(connected_clients),
        **broadcast_queue.get_stats()
    }
