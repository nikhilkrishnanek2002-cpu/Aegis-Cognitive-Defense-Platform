"""
WebSocket live radar frame broadcaster.
Runs a background task that generates radar scans every second
and pushes the result to all connected WebSocket clients.
"""
import asyncio
import json
import time
from typing import Set
from fastapi import WebSocket, WebSocketDisconnect

# Track connected clients
_connected: Set[WebSocket] = set()
_latest_frame: dict = {}  # Last generated frame cached for new connections


async def ws_endpoint(websocket: WebSocket):
    """WebSocket endpoint handler — clients call /ws/stream."""
    await websocket.accept()
    _connected.add(websocket)
    try:
        # Send the latest cached frame immediately on connect
        if _latest_frame:
            await websocket.send_text(json.dumps(_latest_frame))
        # Keep alive — listen for disconnect
        while True:
            try:
                # Wait for client ping or disconnect
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        _connected.discard(websocket)


async def broadcast_frame(frame: dict):
    """Broadcast a radar frame to all connected WebSocket clients."""
    global _latest_frame, _connected
    _latest_frame = frame
    frame_str = json.dumps(frame)
    dead = set()
    for ws in list(_connected):
        try:
            await ws.send_text(frame_str)
        except Exception:
            dead.add(ws)
    _connected -= dead


async def radar_broadcast_loop():
    """
    Background loop: runs the radar pipeline every 1s and broadcasts
    results to all connected WebSocket clients.
    """
    from api.routes.radar import _run_full_pipeline
    target_cycle = ["drone", "aircraft", "missile", "helicopter", "bird", "clutter"]
    idx = 0
    while True:
        try:
            target = target_cycle[idx % len(target_cycle)]
            frame = _run_full_pipeline(target=target, distance=200.0, gain_db=15.0)
            frame["type"] = "radar_frame"
            frame["server_time"] = time.time()
            await broadcast_frame(frame)
            idx += 1
        except Exception as e:
            await broadcast_frame({"type": "error", "message": str(e)})
        await asyncio.sleep(1.0)
