"""
Radar processing pipeline: signal generation, detection, AI classification, tracking, EW.
"""
import time
import numpy as np
import torch
import cv2
import json
import uuid
import base64
import os
from io import BytesIO
from PIL import Image
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from api.auth_utils import get_current_user
from api import state as S
from src.config import get_config
from src.signal_generator import generate_radar_signal
from src.detection import detect_targets_from_raw
from src.feature_extractor import get_all_features
from src.model_pytorch import build_pytorch_model
from src.cognitive_logic import adaptive_threshold
from src.logger import log_event
from src.ai_hardening import AIReliabilityHardener, GradCAMExplainer

router = APIRouter(prefix="/api/radar", tags=["radar"])

LABELS = ["Drone", "Aircraft", "Bird", "Helicopter", "Missile", "Clutter"]
PRIORITY = {
    "Drone": "High", "Aircraft": "Medium", "Bird": "Low",
    "Helicopter": "High", "Missile": "Critical", "Clutter": "Low"
}

_cfg = get_config()

# Load model once at module level (cached)
def _load_model():
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

    # Generate Grad-CAM heatmap for best detection
    xai_data = None
    scan_id = str(uuid.uuid4())[:8]
    try:
        if best and _xai_hardener:
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
            
            # Generate Grad-CAM (default uses features layer)
            predicted_idx = LABELS.index(best["label"]) if best["label"] in LABELS else 0
            cam = _xai_hardener.explainer.generate(rd_t, predicted_idx)
            
            if cam is not None:
                # Normalize CAM to 0-255 for image
                cam_img = (cam * 255).astype(np.uint8)
                
                # Save as PNG
                reports_dir = os.path.join("results", "reports")
                os.makedirs(reports_dir, exist_ok=True)
                cam_img_path = os.path.join(reports_dir, f"gradcam_{scan_id}.png")
                Image.fromarray(cam_img).save(cam_img_path)
                
                # Save as JSON for frontend visualization
                cam_json_path = os.path.join(reports_dir, f"gradcam_{scan_id}.json")
                xai_data = {
                    "scan_id": scan_id,
                    "heatmap": cam.tolist(),  # Convert numpy array to list for JSON
                    "heatmap_shape": cam.shape,
                    "target_class": best["label"],
                    "confidence": round(best["confidence"], 4),
                    "image_path": f"/api/visualizations/xai-gradcam-image/{scan_id}",
                }
                with open(cam_json_path, "w") as f:
                    json.dump(xai_data, f)
                
                log_event(f"Generated Grad-CAM heatmap for scan {scan_id}: {best['label']}", level="info")
    except Exception as e:
        log_event(f"Grad-CAM generation error: {e}", level="warning")

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
