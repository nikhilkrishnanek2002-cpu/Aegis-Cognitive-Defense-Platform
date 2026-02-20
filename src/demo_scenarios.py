"""Demo scenarios that simulate radar timelines for the Streamlit dashboard.

Each helper synthesizes radar returns, runs the standard detection + tracking
pipeline, and returns timeline entries that the UI can render without needing
real hardware.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import get_config
from .detection import detect_targets_from_raw
from .logger import log_event
from .signal_generator import generate_radar_signal
from .tracker import MultiTargetTracker


def _get_detection_params() -> Dict:
    cfg = get_config().get("detection", {})
    return {
        "fs": int(cfg.get("fs", 4096)),
        "n_range": int(cfg.get("n_range", 128)),
        "n_doppler": int(cfg.get("n_doppler", 128)),
        "method": cfg.get("method", "ca"),
        "guard": int(cfg.get("guard", 2)),
        "train": int(cfg.get("train", 8)),
        "pfa": float(cfg.get("pfa", 1e-6)),
    }


def _build_tracker() -> MultiTargetTracker:
    tracker_cfg = get_config().get("tracker", {})
    return MultiTargetTracker(tracker_cfg)


def _run_detection_pipeline(signal: np.ndarray, tracker: MultiTargetTracker, params: Dict) -> Tuple[Dict, Dict]:
    detection_result = detect_targets_from_raw(
        signal,
        fs=params["fs"],
        n_range=params["n_range"],
        n_doppler=params["n_doppler"],
        method=params["method"],
        guard=params["guard"],
        train=params["train"],
        pfa=params["pfa"],
    )
    tracks = tracker.update(detection_result["detections"])
    return detection_result, tracks


def _timeline_entry(step: int, signal: np.ndarray, detection_result: Dict, tracks: Dict, description: str,
                    extra: Dict | None = None) -> Dict:
    entry = {
        "step": step,
        "timestamp": step,  # relative timeline units
        "description": description,
        "signal_power": float(np.mean(np.abs(signal) ** 2)),
        "num_detections": detection_result["stats"].get("num_detections", 0),
        "detections": detection_result["detections"],
        "tracks": tracks,
    }
    if extra:
        entry.update(extra)
    return entry


def run_drone_approach_demo(num_steps: int = 6) -> Dict:
    """Simulate a single drone closing the radar aperture."""
    log_event("Starting drone approach demo", level="info")
    params = _get_detection_params()
    tracker = _build_tracker()

    start_distance = 800.0
    end_distance = 120.0
    distances = np.linspace(start_distance, end_distance, num_steps)

    timeline: List[Dict] = []
    for idx, distance in enumerate(distances):
        signal = generate_radar_signal("drone", distance=float(distance), fs=params["fs"])
        detection_result, tracks = _run_detection_pipeline(signal, tracker, params)
        entry = _timeline_entry(
            idx,
            signal,
            detection_result,
            tracks,
            description=f"Drone closing to ~{distance:.0f} m",
            extra={"distance_m": float(distance)},
        )
        timeline.append(entry)

    return {
        "scenario": "drone_approach",
        "timeline": timeline,
        "metadata": {"steps": num_steps, "start_distance_m": start_distance, "end_distance_m": end_distance},
    }


def run_jamming_attack_demo(num_steps: int = 5) -> Dict:
    """Simulate a high-power noise jammer degrading detections."""
    log_event("Starting jamming attack demo", level="warning")
    params = _get_detection_params()
    tracker = _build_tracker()

    jammer_levels = np.linspace(0.2, 1.0, num_steps)
    timeline: List[Dict] = []

    for idx, level in enumerate(jammer_levels):
        signal = generate_radar_signal("aircraft", distance=400.0, fs=params["fs"])
        noise = (
            np.random.normal(0, 1, len(signal)) + 1j * np.random.normal(0, 1, len(signal))
        ).astype(np.complex64)
        noise *= (level * 5.0)
        burst = np.zeros_like(noise)
        start = (idx * 256) % (len(signal) // 2)
        stop = min(len(signal), start + len(signal) // 3)
        burst[start:stop] = noise[start:stop]
        window = np.hanning(stop - start)
        burst[start:stop] *= window
        signal = signal + burst

        detection_result, tracks = _run_detection_pipeline(signal, tracker, params)
        entry = _timeline_entry(
            idx,
            signal,
            detection_result,
            tracks,
            description="Wideband jammer active",
            extra={"jammer_level": float(level), "burst_window": (int(start), int(stop))},
        )
        timeline.append(entry)

    return {
        "scenario": "jamming_attack",
        "timeline": timeline,
        "metadata": {"steps": num_steps, "max_jammer_level": float(jammer_levels[-1])},
    }


def run_multi_target_demo(num_steps: int = 6) -> Dict:
    """Simulate multiple cooperative/incoming targets for tracker stress testing."""
    log_event("Starting multi-target demo", level="info")
    params = _get_detection_params()
    tracker = _build_tracker()

    targets = [
        {"type": "drone", "distance": 700.0, "velocity": -80.0},
        {"type": "aircraft", "distance": 1200.0, "velocity": -40.0},
        {"type": "bird", "distance": 350.0, "velocity": -20.0},
    ]

    timeline: List[Dict] = []
    for idx in range(num_steps):
        composite = np.zeros(params["fs"], dtype=np.complex64)
        snapshot = []
        for target in targets:
            target["distance"] = max(80.0, target["distance"] + target["velocity"] * 0.1)
            snapshot.append({"type": target["type"], "distance_m": round(target["distance"], 2)})
            sig = generate_radar_signal(target["type"], distance=target["distance"], fs=params["fs"])
            composite += sig.astype(np.complex64)
        composite /= max(len(targets), 1)

        detection_result, tracks = _run_detection_pipeline(composite, tracker, params)
        entry = _timeline_entry(
            idx,
            composite,
            detection_result,
            tracks,
            description="Multiple targets maneuvering",
            extra={"targets": snapshot},
        )
        timeline.append(entry)

    return {
        "scenario": "multi_target",
        "timeline": timeline,
        "metadata": {"steps": num_steps, "num_targets": len(targets)},
    }
