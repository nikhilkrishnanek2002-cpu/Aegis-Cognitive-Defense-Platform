"""
Radar processing pipeline: signal generation, detection, AI classification, tracking, EW.
"""
import time
import numpy as np
import psutil
import hashlib
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
import cv2
import json
import uuid
import base64
import os
from io import BytesIO
from PIL import Image
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Tuple
from collections import deque
from threading import Lock

from api.auth_utils import get_current_user
from api import state as S
from src.config import get_config
from src.signal_generator import generate_radar_signal
from src.detection import detect_targets_from_raw
from src.feature_extractor import get_all_features
if HAS_TORCH:
    from src.model_pytorch import build_pytorch_model
else:
    def build_pytorch_model(*args, **kwargs):
        return None
from src.cognitive_logic import adaptive_threshold
from src.logger import log_event
from src.ai_hardening import AIReliabilityHardener, GradCAMExplainer

router = APIRouter(prefix="/api/radar", tags=["radar"])

LABELS = ["Drone", "Aircraft", "Bird", "Helicopter", "Clutter"]
PRIORITY = {
    "Drone": "High", "Aircraft": "Medium", "Bird": "Low",
    "Helicopter": "High", "Clutter": "Low"
}

_cfg = get_config()

# ============================================================================
# SYSTEM HEALTH MONITORING & ADAPTIVE RESOLUTION
# ============================================================================

class SystemHealthMonitor:
    """Monitor system resources and health metrics"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.latency_history = deque(maxlen=history_size)
        self.lock = Lock()
        
    def update(self):
        """Update system metrics"""
        try:
            with self.lock:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory_info.percent)
        except Exception as e:
            log_event(f"Health monitor error: {e}", level="warning")
    
    def get_avg_cpu(self) -> float:
        """Get average CPU usage (0-100)"""
        with self.lock:
            if not self.cpu_history:
                return 0.0
            return float(np.mean(list(self.cpu_history)))
    
    def get_avg_memory(self) -> float:
        """Get average memory usage (0-100)"""
        with self.lock:
            if not self.memory_history:
                return 0.0
            return float(np.mean(list(self.memory_history)))
    
    def add_latency(self, latency_ms: float):
        """Record operation latency"""
        with self.lock:
            self.latency_history.append(latency_ms)
    
    def get_avg_latency(self) -> float:
        """Get average latency (milliseconds)"""
        with self.lock:
            if not self.latency_history:
                return 0.0
            return float(np.mean(list(self.latency_history)))
    
    def is_system_healthy(self) -> bool:
        """Check if system is healthy enough for full processing"""
        cpu = self.get_avg_cpu()
        mem = self.get_avg_memory()
        
        # Unhealthy if CPU or memory critically high
        return cpu < 85.0 and mem < 85.0
    
    def is_system_under_load(self) -> bool:
        """Check if system is under moderate load"""
        cpu = self.get_avg_cpu()
        mem = self.get_avg_memory()
        
        return cpu > 70.0 or mem > 70.0

_health_monitor = SystemHealthMonitor()

def _calculate_adaptive_heatmap_size(threat_level: str, ew_active: bool = False) -> int:
    """
    Adaptively select heatmap resolution based on system state and threat.
    
    Resolution scaling strategy:
    - Green alert + healthy system: 128×128 (balanced quality/performance)
    - Green alert + under load: 96×96 (reduced quality, improved performance)
    - Yellow alert: 128×128 (maintain quality)
    - Red alert (threat): 256×256 (maximum detail for analysis)
    - System unhealthy: 64×64 (minimum quality, maximum performance)
    """
    
    # Update health monitor
    _health_monitor.update()
    
    cpu_load = _health_monitor.get_avg_cpu()
    mem_load = _health_monitor.get_avg_memory()
    
    # Emergency fallback if system critically stressed
    if cpu_load > 85.0 or mem_load > 85.0:
        log_event(f"System stressed: CPU={cpu_load:.1f}% MEM={mem_load:.1f}% → using minimal resolution", 
                 level="warning")
        return 32  # Minimal resolution, maximum performance
    
    # Threat-based scaling
    threat_size = {
        'green': 128,    # Normal: balanced
        'yellow': 128,   # Caution: maintain quality
        'red': 256       # Critical: high detail (but check load first)
    }
    
    base_size = threat_size.get(threat_level, 128)
    
    # Load-based adjustment (prefer downscaling under load)
    if _health_monitor.is_system_under_load():
        # Under load: reduce to next lower power-of-2
        size_options = [32, 64, 96, 128, 160, 192, 224, 256]
        idx = size_options.index(base_size) if base_size in size_options else len(size_options) - 1
        if idx > 0:
            base_size = size_options[idx - 1]  # One level lower
            log_event(f"Load detected: downscaling heatmap {base_size} (load: CPU={cpu_load:.1f}% MEM={mem_load:.1f}%)",
                     level="info")
    
    # EW active: prioritize detail for threat analysis
    if ew_active and threat_level == 'red' and not _health_monitor.is_system_under_load():
        base_size = min(256, base_size)  # Ensure high detail
    
    # Ensure valid size (power of 2 preferred)
    valid_sizes = [32, 64, 96, 128, 160, 192, 224, 256]
    selected = min(valid_sizes, key=lambda x: abs(x - base_size))
    
    return selected

# ============================================================================
# HEATMAP CACHING LAYER
# ============================================================================

class HeatmapCache:
    """Simple in-memory cache for generated heatmaps with memory limits"""
    
    def __init__(self, max_items: int = 1000, max_memory_mb: int = 100):
        self.max_items = max_items
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, dict] = {}
        self.lock = Lock()
        self.current_memory = 0
    
    def _compute_key(self, detection_id: str, model_version: str, size: int) -> str:
        """Create cache key from detection + model + size"""
        key_str = f"{detection_id}:{model_version}:{size}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[dict]:
        """Retrieve cached heatmap"""
        with self.lock:
            if key in self.cache:
                self.cache[key]['hits'] += 1
                return self.cache[key]['data']
        return None
    
    def put(self, key: str, heatmap_data: dict) -> bool:
        """Store heatmap in cache, return success"""
        with self.lock:
            # Estimate memory usage
            data_size = len(json.dumps(heatmap_data).encode())
            
            # Check if we'd exceed memory limit
            if self.current_memory + data_size > self.max_memory_bytes:
                log_event(f"Cache full: {self.current_memory / 1024 / 1024:.1f} MB used, evicting", 
                         level="info")
                # Evict least recently used
                if self.cache:
                    lru_key = min(self.cache.keys(), 
                                key=lambda k: self.cache[k].get('last_access', 0))
                    lru_size = len(json.dumps(self.cache[lru_key]['data']).encode())
                    del self.cache[lru_key]
                    self.current_memory -= lru_size
            
            # Check if we'd exceed item limit
            if len(self.cache) >= self.max_items:
                # Remove oldest entry
                if self.cache:
                    oldest_key = min(self.cache.keys(),
                                   key=lambda k: self.cache[k].get('timestamp', 0))
                    oldest_size = len(json.dumps(self.cache[oldest_key]['data']).encode())
                    del self.cache[oldest_key]
                    self.current_memory -= oldest_size
            
            # Store entry
            self.cache[key] = {
                'data': heatmap_data,
                'timestamp': time.time(),
                'last_access': time.time(),
                'hits': 0
            }
            self.current_memory += data_size
            
            return True
    
    def clear_expired(self, ttl_seconds: int = 3600):
        """Remove entries older than TTL"""
        with self.lock:
            now = time.time()
            expired_keys = [k for k, v in self.cache.items() 
                          if now - v['timestamp'] > ttl_seconds]
            
            for key in expired_keys:
                data_size = len(json.dumps(self.cache[key]['data']).encode())
                del self.cache[key]
                self.current_memory -= data_size
            
            if expired_keys:
                log_event(f"Cache: cleared {len(expired_keys)} expired entries", level="info")

_heatmap_cache = HeatmapCache(max_items=500, max_memory_mb=200)

# ============================================================================
# MULTI-RESOLUTION GRADCAM GENERATION
# ============================================================================

def _generate_synthetic_gradcam(size: int = 128) -> np.ndarray:
    """
    Generate synthetic Grad-CAM as fallback with adaptive resolution.
    
    Resolution Options:
    - 32×32:   Ultra-lightweight (1 KB) - emergency fallback
    - 64×64:   Lightweight (16 KB) - high load
    - 96×96:   Medium (36 KB) - moderate load
    - 128×128: Standard (64 KB) - normal operations ← DEFAULT
    - 256×256: High-detail (256 KB) - critical analysis
    """
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2) / 2)
    Z = Z + np.random.normal(0, 0.05, Z.shape)
    Z = np.clip(Z, 0, 1)
    return Z

def _generate_multiresolution_heatmaps(base_cam: np.ndarray, 
                                      compute_detail: bool = False) -> Dict[str, list]:
    """
    Generate heatmaps at multiple resolutions for progressive loading.
    
    Always generates: thumbnail (32×32), standard (128×128)
    Optionally generates: detail (256×256) if compute_detail=True
    """
    
    try:
        # Thumbnail (always fast, always generated)
        thumbnail = cv2.resize(base_cam, (32, 32), interpolation=cv2.INTER_LINEAR)
        
        # Standard (main display)
        standard = cv2.resize(base_cam, (128, 128), interpolation=cv2.INTER_LINEAR)
        
        # Detail (optional, higher cost)
        detail = None
        if compute_detail and not _health_monitor.is_system_under_load():
            detail = cv2.resize(base_cam, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        return {
            'thumbnail': thumbnail.tolist(),     # 1 KB
            'standard': standard.tolist(),       # 64 KB
            'detail': detail.tolist() if detail is not None else None  # 256 KB or None
        }
    
    except Exception as e:
        log_event(f"Multi-resolution generation error: {e}", level="warning")
        # Fallback: only return standard
        standard = cv2.resize(base_cam, (128, 128), interpolation=cv2.INTER_LINEAR)
        return {
            'thumbnail': cv2.resize(base_cam, (32, 32)).tolist(),
            'standard': standard.tolist(),
            'detail': None
        }

# Load model once at module level (cached)
def _load_model():
    if not HAS_TORCH:
        return None, None
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    m = build_pytorch_model(num_classes=len(LABELS))
    from src.security_utils import safe_path
    model_path = safe_path("radar_model_pytorch.pt")
    if os.path.exists(model_path):
        try:
            sd = torch.load(model_path, map_location=device, weights_only=True)
            m.load_state_dict(sd)
        except Exception as e:
            log_event(f"Model load error: {e}", level="warning")
    m.to(device).eval()
    return m, device

_model, _device = _load_model()

# Initialize XAI hardener for Grad-CAM
try:
    _xai_hardener = AIReliabilityHardener(_model)
    _xai_hardener.set_labels(LABELS)
except Exception as e:
    log_event(f"XAI hardener init error: {e}", level="warning")
    _xai_hardener = None


class ScanRequest(BaseModel):
    target: str = "drone"
    distance: float = 200.0
    gain_db: float = 15.0
    source: str = "simulated"


def _run_full_pipeline(target: str, distance: float, gain_db: float):
    """Core radar pipeline — returns a structured JSON-ready dict."""
    signal = generate_radar_signal(target.lower(), distance)
    signal *= 10 ** (gain_db / 20)

    detect_res = detect_targets_from_raw(
        signal, fs=4096, n_range=128, n_doppler=128,
        method="ca", guard=2, train=8, pfa=1e-6
    )
    rd_map, spec, meta, photonic = get_all_features(signal)
    detections = detect_res.get("detections", [])

    ai_results = []
    IMG_SIZE = 128
    crop_size = int(_cfg.get("detection", {}).get("crop_size", 32))
    half = crop_size // 2

    if detections:
        try:
            spec_resized_full = cv2.resize(spec, (rd_map.shape[1], rd_map.shape[0]))
        except Exception:
            spec_resized_full = np.abs(spec)

        for det in detections:
            i, j, val = int(det[0]), int(det[1]), det[2]
            pad_y = max(0, half - i, (i + half) - rd_map.shape[0] + 1)
            pad_x = max(0, half - j, (j + half) - rd_map.shape[1] + 1)
            rd_p = np.pad(rd_map, ((pad_y, pad_y), (pad_x, pad_x))) if pad_x or pad_y else rd_map
            sp_p = np.pad(spec_resized_full, ((pad_y, pad_y), (pad_x, pad_x))) if pad_x or pad_y else spec_resized_full
            ip, jp = i + pad_y, j + pad_x
            rd_crop = cv2.resize(rd_p[ip - half:ip + half, jp - half:jp + half].astype(np.float32), (IMG_SIZE, IMG_SIZE))
            sp_crop = cv2.resize(sp_p[ip - half:ip + half, jp - half:jp + half].astype(np.float32), (IMG_SIZE, IMG_SIZE))
            rd_n = (rd_crop - rd_crop.mean()) / (rd_crop.std() + 1e-8)
            sp_n = (sp_crop - sp_crop.mean()) / (sp_crop.std() + 1e-8)

            rd_t = torch.from_numpy(rd_n).float().unsqueeze(0).unsqueeze(0).to(_device)
            sp_t = torch.from_numpy(sp_n).float().unsqueeze(0).unsqueeze(0).to(_device)
            me_t = torch.from_numpy(meta).float().unsqueeze(0).to(_device)
            with torch.no_grad():
                out = _model(rd_t, sp_t, me_t)
                ps = torch.softmax(out, dim=1)
                conf, idx = float(torch.max(ps)), int(torch.argmax(ps))
                label = LABELS[idx] if idx < len(LABELS) else "Clutter"
            ai_results.append({"det": [i, j], "label": label, "confidence": conf, "value": val})

    best = max(ai_results, key=lambda x: x["confidence"]) if ai_results else None
    detected = best["label"] if best else "Clutter"
    confidence = best["confidence"] if best else 0.0

    # Multi-target tracker update
    tracker_dets = [(r["det"][0], r["det"][1], r["confidence"]) for r in ai_results]
    active_tracks = S.tracker.update(tracker_dets)

    # EW defense
    ew_result = S.ew_defense.analyze(
        signal=signal,
        detections=detections,
        ai_labels=[r["label"] for r in ai_results],
        ai_confidences=[r["confidence"] for r in ai_results],
    )

    # Cognitive controller
    avg_conf = float(np.mean([r["confidence"] for r in ai_results])) if ai_results else 0.0
    avg_trk = float(np.mean([t["confidence"] for t in active_tracks.values()])) if active_tracks else 0.0
    ctrl_state = S.cognitive_controller.observe(
        detection_confidence=avg_conf,
        tracking_confidence=avg_trk,
        num_active_tracks=len([t for t in active_tracks.values() if t["state"] == "confirmed"]),
        total_detections=len(detections),
        false_positives=max(0, len(detections) - len(ai_results)),
        current_gain=gain_db,
        max_gain=40.0,
    )
    S.cognitive_controller.learn(ctrl_state)
    cognitive_action = S.cognitive_controller.decide(ctrl_state)

    thresh = adaptive_threshold(photonic["noise_power"])
    is_alert = confidence > thresh

    # ========================================================================
    # ADVANCED GRAD-CAM GENERATION (with adaptive resolution, caching, etc)
    # ========================================================================
    
    xai_data = None
    scan_id = str(uuid.uuid4())[:8]
    grad_cam_start_time = time.time()
    
    try:
        # 1. Determine adaptive heatmap size based on system state and threat level
        ew_active = ew_result.get("ew_active", False)
        threat_level = ew_result.get("threat_level", "green")
        adaptive_size = _calculate_adaptive_heatmap_size(threat_level, ew_active)
        
        # 2. Try cache first (check if we already generated this)
        cache_key = _heatmap_cache._compute_key(
            f"best_det_{str(best)}" if best else "no_best",
            "v2.0",
            adaptive_size
        )
        cached_xai = _heatmap_cache.get(cache_key)
        
        if cached_xai is not None:
            log_event(f"Cache hit: Grad-CAM for size {adaptive_size}×{adaptive_size}", level="info")
            xai_data = cached_xai
        else:
            # 3. Generate fresh Grad-CAM with adaptive resolution
            cam = None
            
            # Try real Grad-CAM first (if model available)
            if best and _xai_hardener and _model is not None:
                try:
                    best_det = best["det"]
                    i, j = int(best_det[0]), int(best_det[1])
                    
                    # Get RD and spec crops for Grad-CAM
                    half = crop_size // 2
                    pad_y = max(0, half - i, (i + half) - rd_map.shape[0] + 1)
                    pad_x = max(0, half - j, (j + half) - rd_map.shape[1] + 1)
                    rd_p = np.pad(rd_map, ((pad_y, pad_y), (pad_x, pad_x))) if pad_x or pad_y else rd_map
                    sp_p = np.pad(spec_resized_full, ((pad_y, pad_y), (pad_x, pad_x))) if pad_x or pad_y else spec_resized_full
                    ip, jp = i + pad_y, j + pad_x
                    rd_crop = cv2.resize(rd_p[ip - half:ip + half, jp - half:jp + half].astype(np.float32), (IMG_SIZE, IMG_SIZE))
                    
                    # Normalize
                    rd_n = (rd_crop - rd_crop.mean()) / (rd_crop.std() + 1e-8)
                    rd_t = torch.from_numpy(rd_n).float().unsqueeze(0).unsqueeze(0).to(_device)
                    
                    # Generate Grad-CAM (real model)
                    predicted_idx = LABELS.index(best["label"]) if best["label"] in LABELS else 0
                    cam = _xai_hardener.explainer.generate(rd_t, predicted_idx)
                    
                    if cam is not None:
                        cam = np.clip(cam, 0, 1)
                        log_event(f"Generated real Grad-CAM ({adaptive_size}×{adaptive_size}) for {best['label']}", 
                                 level="info")
                    
                except Exception as e:
                    log_event(f"Real Grad-CAM generation failed: {e}, using fallback", level="warning")
                    cam = None
            
            # Fallback: generate synthetic Grad-CAM if real failed or unavailable
            if cam is None:
                cam = _generate_synthetic_gradcam(size=adaptive_size)
                log_event(f"Generated synthetic Grad-CAM ({adaptive_size}×{adaptive_size})", level="info")
            
            # 4. Generate multi-resolution versions for progressive frontend loading
            compute_detail = (threat_level == 'red' and 
                            not _health_monitor.is_system_under_load())
            multi_res = _generate_multiresolution_heatmaps(cam, compute_detail=compute_detail)
            
            # 5. Build XAI response with multi-resolution support
            target_label = best["label"] if best else detected
            xai_data = {
                "scan_id": scan_id,
                "heatmap": multi_res['standard'],           # Main display: 128×128
                "heatmap_thumbnail": multi_res['thumbnail'], # Quick preview: 32×32
                "heatmap_detail": multi_res['detail'],       # Detailed analysis: 256×256 or None
                "heatmap_shape": [128, 128],                # Standard shape
                "heatmap_shape_detail": [256, 256] if multi_res['detail'] else None,
                "target_class": target_label,
                "confidence": round(best["confidence"], 4) if best else confidence,
                "image_path": f"/api/visualizations/xai-gradcam-image/{scan_id}",
                "generation_mode": "real" if best and _xai_hardener and _model else "synthetic",
                "adaptive_resolution": adaptive_size,
                "multi_resolution_support": True
            }
            
            # 6. Save to cache for future reuse
            _heatmap_cache.put(cache_key, xai_data)
            
            # 7. Save PNG for file storage
            reports_dir = os.path.join("results", "reports")
            os.makedirs(reports_dir, exist_ok=True)
            cam_img = (cam * 255).astype(np.uint8)
            cam_img_path = os.path.join(reports_dir, f"gradcam_{scan_id}.png")
            Image.fromarray(cam_img).save(cam_img_path)
        
        grad_cam_time = time.time() - grad_cam_start_time
        _health_monitor.add_latency(grad_cam_time * 1000)  # Record latency
        
        log_event(f"Grad-CAM complete for {scan_id}: {grad_cam_time:.2f}s, "
                 f"sys_cpu={_health_monitor.get_avg_cpu():.1f}%, "
                 f"sys_mem={_health_monitor.get_avg_memory():.1f}%", 
                 level="info")
        
    except Exception as e:
        log_event(f"Grad-CAM generation critical error: {e}", level="error")
        # Emergency fallback: minimal Grad-CAM
        try:
            emergency_cam = _generate_synthetic_gradcam(size=32)  # Minimal size
            xai_data = {
                "scan_id": scan_id,
                "heatmap": cv2.resize(emergency_cam, (128, 128)).tolist(),
                "heatmap_thumbnail": cv2.resize(emergency_cam, (32, 32)).tolist(),
                "heatmap_detail": None,
                "target_class": detected,
                "confidence": confidence,
                "image_path": f"/api/visualizations/xai-gradcam-image/{scan_id}",
                "generation_mode": "emergency",
                "adaptive_resolution": 32,
                "multi_resolution_support": True,
                "error_fallback": True
            }
            log_event(f"Generated emergency fallback Grad-CAM for {scan_id}", level="warning")
        except Exception as e2:
            log_event(f"Emergency Grad-CAM also failed: {e2}", level="error")
            xai_data = None

    return {
        "scan_id": scan_id,
        "timestamp": time.time(),
        "detected": detected,
        "confidence": round(confidence, 4),
        "priority": PRIORITY.get(detected, "Low"),
        "is_alert": is_alert,
        "threshold": round(thresh, 4),
        "num_detections": len(detections),
        "ai_results": ai_results,
        "system_health": {
            "cpu_percent": round(_health_monitor.get_avg_cpu(), 1),
            "memory_percent": round(_health_monitor.get_avg_memory(), 1),
            "avg_latency_ms": round(_health_monitor.get_avg_latency(), 1),
            "is_healthy": _health_monitor.is_system_healthy()
        },
        "active_tracks": {
            tid: {
                "position": list(v["position"]),
                "velocity": list(v["velocity"]),
                "state": v["state"],
                "confidence": round(v["confidence"], 4),
            }
            for tid, v in active_tracks.items()
        },
        "ew": {
            "active": ew_result.get("ew_active", False),
            "threat_level": ew_result.get("threat_level", "green"),
            "num_threats": len(ew_result.get("threats", [])),
        },
        "cognitive": {
            "is_adaptive": cognitive_action.is_adaptive,
            "suggested_gain_db": round(cognitive_action.gain_db, 2),
        },
        "photonic": {
            "bandwidth_mhz": round(photonic["instantaneous_bandwidth"] / 1e6, 2),
            "noise_power": round(float(photonic["noise_power"]), 6),
            "clutter_power": round(float(photonic["clutter_power"]), 6),
            "pulse_width_us": round(photonic["pulse_width"] * 1e6, 2),
            "chirp_slope_thz": round(photonic["chirp_slope"] / 1e12, 2),
            "ttd_vector": photonic["ttd_vector"].tolist() if hasattr(photonic["ttd_vector"], 'tolist') else list(photonic["ttd_vector"]),
        },
        "rd_map": rd_map.tolist(),
        "spec": spec.tolist(),
        "meta": meta.tolist(),
        "xai": xai_data,
    }


@router.post("/scan")
async def scan(body: ScanRequest, user: dict = Depends(get_current_user)):
    result = _run_full_pipeline(body.target, body.distance, body.gain_db)
    return result


@router.get("/labels")
async def get_labels():
    return {"labels": LABELS, "priorities": PRIORITY}
