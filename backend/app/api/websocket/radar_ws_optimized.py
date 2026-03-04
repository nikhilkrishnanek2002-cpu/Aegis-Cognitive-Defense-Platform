"""Optimized WebSocket handler with performance monitoring."""

import asyncio
import json
import math
from typing import Set
from fastapi import WebSocket, WebSocketDisconnect
from app.core.logging import websocket_logger
from app.engine.event_bus import event_bus, Events
from app.core.performance import timed_async, timer, broadcast_queue, numpy_to_native
import time
import random


# Track connected clients
connected_clients: Set[WebSocket] = set()
websocket_stats = {
    "connections": 0,
    "disconnections": 0,
    "messages_sent": 0,
    "messages_failed": 0
}


def _generate_realistic_rd_map(size: int = 128) -> list:
    """Generate a realistic Range-Doppler map with noise floor, targets, and clutter."""
    # Noise floor (Rayleigh-distributed like real radar thermal noise)
    rd_map = [[random.expovariate(5.0) for _ in range(size)] for _ in range(size)]
    
    # Add 2-4 point targets (strong returns in Range-Doppler space)
    num_targets = random.randint(2, 4)
    t = time.time()
    for _ in range(num_targets):
        r_idx = random.randint(10, size - 15)
        d_idx = random.randint(5, size - 10)
        amplitude = random.uniform(0.6, 1.0)
        # Spread each target over a few cells (realistic point-spread function)
        for dr in range(-3, 4):
            for dd in range(-2, 3):
                ri = r_idx + dr
                di = d_idx + dd
                if 0 <= ri < size and 0 <= di < size:
                    spread = math.exp(-(dr**2 / 3.0 + dd**2 / 2.0))
                    rd_map[ri][di] += amplitude * spread
    
    # Add zero-Doppler clutter ridge (ground/sea clutter at Doppler=0)
    center_doppler = size // 2
    for r in range(size):
        clutter_amp = random.uniform(0.15, 0.35)
        for dd in range(-4, 5):
            di = center_doppler + dd
            if 0 <= di < size:
                spread = math.exp(-dd**2 / 4.0)
                rd_map[r][di] += clutter_amp * spread
    
    # Normalize to [0, 1]
    max_val = max(max(row) for row in rd_map)
    if max_val > 0:
        rd_map = [[v / max_val for v in row] for row in rd_map]
    
    return rd_map


def _generate_realistic_spectrogram(n_time: int = 128, n_freq: int = 512) -> list:
    """Generate a realistic spectrogram with carrier, harmonics, and noise."""
    t = time.time()
    spec = [[random.expovariate(8.0) for _ in range(n_freq)] for _ in range(n_time)]
    
    # Add a drifting carrier signal
    carrier_freq = random.randint(150, 350)
    drift_rate = random.uniform(-0.3, 0.3)
    for ti in range(n_time):
        fc = int(carrier_freq + drift_rate * ti) % n_freq
        amplitude = random.uniform(0.7, 1.0)
        for df in range(-6, 7):
            fi = fc + df
            if 0 <= fi < n_freq:
                spread = math.exp(-df**2 / 8.0)
                spec[ti][fi] += amplitude * spread
    
    # Add 1-2 harmonics
    for harmonic in range(2, random.randint(3, 4)):
        hf = (carrier_freq * harmonic) % n_freq
        for ti in range(n_time):
            hfc = int(hf + drift_rate * harmonic * ti) % n_freq
            amplitude = random.uniform(0.15, 0.35) / harmonic
            for df in range(-4, 5):
                fi = hfc + df
                if 0 <= fi < n_freq:
                    spread = math.exp(-df**2 / 5.0)
                    spec[ti][fi] += amplitude * spread
    
    # Add intermittent broadband burst (radar pulse or jammer)
    burst_start = random.randint(0, n_time - 10)
    burst_len = random.randint(3, 8)
    for ti in range(burst_start, min(burst_start + burst_len, n_time)):
        for fi in range(n_freq):
            spec[ti][fi] += random.uniform(0.05, 0.2)
    
    # Normalize to [0, 1]
    max_val = max(max(row) for row in spec)
    if max_val > 0:
        spec = [[v / max_val for v in row] for row in spec]
    
    return spec


def _generate_realistic_gradcam(size: int = 64) -> list:
    """Generate a realistic Grad-CAM activation map."""
    # Start with low-level noise floor
    heatmap = [[random.uniform(0.0, 0.08) for _ in range(size)] for _ in range(size)]
    
    # Add 1-3 activation blobs
    num_blobs = random.randint(1, 3)
    for b in range(num_blobs):
        cx = random.uniform(size * 0.15, size * 0.85)
        cy = random.uniform(size * 0.15, size * 0.85)
        sx = random.uniform(size * 0.06, size * 0.15)
        sy = random.uniform(size * 0.06, size * 0.15)
        amp = 1.0 if b == 0 else random.uniform(0.25, 0.6)
        for i in range(size):
            for j in range(size):
                g = amp * math.exp(-((i - cy)**2 / (2 * sy**2) + (j - cx)**2 / (2 * sx**2)))
                heatmap[i][j] += g
    
    # Normalize to [0, 1]
    max_val = max(max(row) for row in heatmap)
    if max_val > 0:
        heatmap = [[v / max_val for v in row] for row in heatmap]
    
    return heatmap


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


async def ws_stream_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time radar stream (frontend format).
    Transforms backend data to match frontend RadarFrame schema.
    """
    await websocket.accept()
    connected_clients.add(websocket)
    
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    websocket_stats["connections"] += 1
    
    websocket_logger.info(f"Stream client connected: {client_id}")
    
    try:
        # Subscribe to radar frame events and transform data
        async def transform_and_send_frame(frame):
            """Transform backend frame to frontend format."""
            try:
                if hasattr(frame, 'dict'):
                    frame_dict = frame.dict()
                else:
                    frame_dict = frame
                
                # Transform to frontend format
                transformed_frame = {
                    "detected": str(len(frame_dict.get("targets", []))),
                    "confidence": 0.85,
                    "priority": "HIGH",
                    "is_alert": len(frame_dict.get("threats", [])) > 0,
                    "threshold": 0.65,
                    "num_detections": len(frame_dict.get("tracked_targets", [])),
                    "active_tracks": {
                        track.get("track_id", f"t{i}"): {
                            "position": [
                                track.get("position", {}).get("x", 0),
                                track.get("position", {}).get("y", 0)
                            ],
                            "velocity": [
                                track.get("velocity", {}).get("vx", 0),
                                track.get("velocity", {}).get("vy", 0)
                            ],
                            "state": "TRACKED",
                            "confidence": track.get("confidence", 0.8)
                        }
                        for i, track in enumerate(frame_dict.get("tracked_targets", []))
                    },
                    "ew": {
                        "active": False,
                        "threat_level": "LOW",
                        "num_threats": len(frame_dict.get("threats", []))
                    },
                    "cognitive": {
                        "is_adaptive": True,
                        "suggested_gain_db": 20.0
                    },
                    "photonic": {
                        "bandwidth_mhz": 500.0,
                        "noise_power": 0.1,
                        "clutter_power": 0.2,
                        "pulse_width_us": 1.0,
                        "chirp_slope_thz": 0.5,
                        "ttd_vector": [0.0] * 64
                    },
                    "rd_map": _generate_realistic_rd_map(128),
                    "spec": _generate_realistic_spectrogram(128, 512),
                    "meta": [0] * 10,
                    "timestamp": int(time.time() * 1000),
                    "xai": {
                        "scan_id": frame_dict.get("frame_id", ""),
                        "heatmap": _generate_realistic_gradcam(64),
                        "heatmap_shape": [64, 64],
                        "target_class": "MIXED",
                        "confidence": round(random.uniform(0.78, 0.96), 2)
                    }
                }
                
                await websocket.send_json(transformed_frame)
                websocket_stats["messages_sent"] += 1
                
            except Exception as e:
                websocket_logger.debug(f"Error sending frame: {e}")
                websocket_stats["messages_failed"] += 1
        
        await event_bus.subscribe(Events.BROADCAST_RADAR_FRAME, 
                                  lambda frame: asyncio.create_task(transform_and_send_frame(frame)))
        
        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data:
                    try:
                        msg = json.loads(data)
                        if msg.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                    except json.JSONDecodeError:
                        pass
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": int(time.time() * 1000)
                    })
                except Exception:
                    break
    
    except WebSocketDisconnect:
        websocket_logger.info(f"Stream client disconnected: {client_id}")
    except Exception as e:
        websocket_logger.error(f"Stream WebSocket error: {e}")
    finally:
        websocket_stats["disconnections"] += 1
        connected_clients.discard(websocket)
        try:
            await websocket.close()
        except:
            pass

