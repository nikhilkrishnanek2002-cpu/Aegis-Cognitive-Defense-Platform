"""WebSocket handler for real-time broadcasting to frontend."""

import asyncio
import json
from typing import Set
from fastapi import WebSocket, WebSocketDisconnect
from app.core.logging import websocket_logger
from app.engine.event_bus import event_bus, Events
from app.core.config import get_config


# Track connected clients
connected_clients: Set[WebSocket] = set()


async def ws_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time radar stream.
    
    Manages client connection and broadcasts radar frames + threats.
    """
    await websocket.accept()
    connected_clients.add(websocket)
    
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    websocket_logger.info(f"Client connected: {client_id}")
    
    try:
        # Subscribe to broadcast events
        await event_bus.subscribe(Events.BROADCAST_RADAR_FRAME, 
                                  lambda frame: asyncio.create_task(_send_radar_frame(websocket, frame)))
        
        await event_bus.subscribe(Events.BROADCAST_THREATS,
                                  lambda threats: asyncio.create_task(_send_threats(websocket, threats)))
        
        await event_bus.subscribe(Events.BROADCAST_SYSTEM_STATUS,
                                  lambda status: asyncio.create_task(_send_system_status(websocket, status)))
        
        # Keep connection alive, listen for client messages
        while True:
            try:
                # Receive with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                
                # Echo or handle client commands
                if data:
                    msg = json.loads(data)
                    await _handle_client_command(websocket, msg)
            
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": asyncio.get_event_loop().time()
                    })
                except Exception:
                    break
            
            except json.JSONDecodeError:
                pass
    
    except WebSocketDisconnect:
        websocket_logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        websocket_logger.error(f"WebSocket error: {e}")
    finally:
        connected_clients.discard(websocket)
        try:
            await websocket.close()
        except:
            pass


async def _send_radar_frame(websocket: WebSocket, frame) -> None:
    """Send radar frame to client."""
    try:
        if isinstance(frame, dict):
            frame_data = frame
        else:
            frame_data = frame.dict() if hasattr(frame, 'dict') else frame
        
        await websocket.send_json({
            "type": "radar_frame",
            "data": frame_data
        })
    except Exception as e:
        websocket_logger.debug(f"Error sending radar frame: {e}")


async def _send_threats(websocket: WebSocket, threats_payload) -> None:
    """Send threat list to client."""
    try:
        if isinstance(threats_payload, dict):
            threats = threats_payload.get("threats", [])
        else:
            threats = threats_payload
        
        threat_data = [t.dict() if hasattr(t, 'dict') else t for t in threats]
        
        await websocket.send_json({
            "type": "threats",
            "data": threat_data
        })
    except Exception as e:
        websocket_logger.debug(f"Error sending threats: {e}")


async def _send_system_status(websocket: WebSocket, status) -> None:
    """Send system status to client."""
    try:
        await websocket.send_json({
            "type": "system_status",
            "data": status
        })
    except Exception as e:
        websocket_logger.debug(f"Error sending system status: {e}")


async def _handle_client_command(websocket: WebSocket, command: dict) -> None:
    """Handle commands from client."""
    cmd_type = command.get("type")
    
    if cmd_type == "ping":
        await websocket.send_json({"type": "pong"})
    
    elif cmd_type == "subscribe":
        channel = command.get("channel")
        websocket_logger.debug(f"Client subscribed to: {channel}")
        await websocket.send_json({
            "type": "subscription_confirmed",
            "channel": channel
        })
    
    elif cmd_type == "get_status":
        # Return current pipeline status
        from app.engine.controller import _controller
        if _controller:
            status = await _controller.get_status()
            await websocket.send_json({
                "type": "status_response",
                "data": status
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
