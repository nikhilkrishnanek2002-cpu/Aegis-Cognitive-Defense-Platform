"""AI Cognitive Photonic Radar - Advanced Defense System"""

import json
import os
import time

import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

matplotlib.use('Agg')

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from src.config import get_config
from src.logger import init_logging, log_event, read_logs
from src.startup_checks import run_startup_checks
from src.feature_extractor import get_all_features
from src.detection import detect_targets_from_raw
from src.model_pytorch import build_pytorch_model
from src.auth import authenticate
from src.security_utils import safe_path
from src.signal_generator import generate_radar_signal
from src.rtl_sdr_receiver import RTLRadar
from src.tracker import MultiTargetTracker
from src.cognitive_controller import CognitiveRadarController
from src.ew_defense import EWDefenseController
from src.db import init_db, ensure_admin_exists
from src.stream_bus import get_producer
from src.xai_pytorch import grad_cam_pytorch
from src.cognitive_logic import adaptive_threshold
from src.demo_scenarios import (
    run_drone_approach_demo,
    run_jamming_attack_demo,
    run_multi_target_demo,
)

# ===============================
# ENVIRONMENT SAFETY
# ===============================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Initialize config, structured logging and startup checks
cfg = get_config()
init_logging(cfg)
_startup = run_startup_checks()



# ===============================
# STREAMLIT CONFIG (FIRST CALL)
# ===============================
st.set_page_config(page_title="AI Cognitive Photonic Radar", layout="wide")

# ===============================
# CUSTOM CSS: COMMAND CENTER VIBE
# ===============================
# Default Streamlit styling applied automatically.
# ===============================
# INIT DATABASE
# ===============================
init_db()
ensure_admin_exists()

# ===============================
# CONSTANTS
# ===============================
LABELS = ["Drone", "Aircraft", "Bird", "Helicopter", "Missile", "Clutter"]

PRIORITY = {
    "Drone": "High",
    "Aircraft": "Medium",
    "Bird": "Low",
    "Helicopter": "High",
    "Missile": "Critical",
    "Clutter": "Low"
}

METRICS_JSON_PATH = os.path.join("outputs", "reports", "metrics.json")
METRIC_IMAGE_PATHS = {
    "Confusion Matrix": os.path.join("results", "reports", "confusion_matrix.png"),
    "ROC Curve": os.path.join("results", "reports", "roc_curve.png"),
    "Precision-Recall Curve": os.path.join("results", "reports", "precision_recall.png"),
    "Training Curves": os.path.join("results", "reports", "training_history.png"),
}

# ===============================
# SESSION STATE
# ===============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.role = None
    st.session_state.history = []
    st.session_state.auth_mode = "login"

if "tracker" not in st.session_state:
    tracker_cfg = cfg.get('tracker', {})
    st.session_state.tracker = MultiTargetTracker(tracker_cfg)
    st.session_state.tracker_enabled = tracker_cfg.get('enabled', True)

if "cognitive_controller" not in st.session_state:
    ctrl_cfg = cfg.get('cognitive_controller', {})
    st.session_state.cognitive_controller = CognitiveRadarController(ctrl_cfg)
    st.session_state.controller_enabled = ctrl_cfg.get('enabled', True)
    st.session_state.manual_override = False

if "ew_defense" not in st.session_state:
    ew_cfg = cfg.get('ew_defense', {})
    st.session_state.ew_defense = EWDefenseController(ew_cfg)
    st.session_state.ew_enabled = ew_cfg.get('enabled', True)

if "track_history" not in st.session_state:
    st.session_state.track_history = []
if "sensitivity_offset" not in st.session_state:
    st.session_state.sensitivity_offset = 0.0
if "demo_timeline" not in st.session_state:
    st.session_state.demo_timeline = None


# ===============================
# LOAD PYTORCH MODEL
# ===============================
@st.cache_resource
def load_model():
    # respect startup GPU availability
    use_cuda = _startup.get("gpu_available", False) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    radar_model = build_pytorch_model(num_classes=len(LABELS))
    model_path = safe_path("radar_model_pytorch.pt")
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            radar_model.load_state_dict(state_dict)
            st.success("✅ Radar AI model weights loaded successfully.")
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
    else:
        st.warning("⚠️ Model weights not found in results/. Using untrained model.")
    
    radar_model.to(device)
    radar_model.eval()
    return radar_model, device

radar_model, device = load_model()


@st.cache_data(show_spinner=False)
def load_metrics_report(path):
    if not os.path.exists(path):
        return None, f"Metrics report missing at {path}"
    try:
        with open(path, "r", encoding="utf-8") as src:
            return json.load(src), None
    except json.JSONDecodeError as exc:
        return None, f"Metrics report is corrupted: {exc}"
    except Exception as exc: # pragma: no cover - defensive path
        return None, f"Unable to load metrics report: {exc}"


def display_metric_image(title, path):
    if os.path.exists(path):
        st.image(path, caption=title, use_container_width=True)
    else:
        st.warning(f"{title} not found at {path}")

# ===============================
# AUTHENTICATION CONTROL FLOW
# ===============================
if not st.session_state.logged_in:
    login_placeholder = st.empty()
    with login_placeholder.container():
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        
        if st.session_state.auth_mode == "login":
            st.markdown('<h1 class="auth-title">🔐 SECURITY CLEARANCE</h1>', unsafe_allow_html=True)
            with st.form("login"):
                username = st.text_input("Operator ID")
                password = st.text_input("Access Code", type="password")
                if st.form_submit_button("AUTHORIZE"):
                    ok, role = authenticate(username, password)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.user = username
                        st.session_state.role = role
                        st.success("✅ AUTHORIZED. ACCESS GRANTED.")
                        time.sleep(1)
                        # Clear the placeholder before rerunning to ensure it's gone
                        login_placeholder.empty()
                        st.rerun()
                    else:
                        st.error("❌ AUTHORIZATION DENIED")
            
            if st.button("New Operator? Register Here"):
                st.session_state.auth_mode = "register"
                st.rerun()

        else:
            st.markdown('<h1 class="auth-title">📝 OPERATOR REGISTRATION</h1>', unsafe_allow_html=True)
            from src.auth import register_user
            with st.form("register"):
                new_user = st.text_input("Choose Operator ID")
                new_pass = st.text_input("Set Access Code", type="password")
                confirm_pass = st.text_input("Confirm Access Code", type="password")
                role = st.selectbox("Assigned Role", ["viewer", "analyst"])
                
                if st.form_submit_button("REGISTER"):
                    if not new_user or not new_pass:
                        st.error("Fields cannot be empty")
                    elif new_pass != confirm_pass:
                        st.error("Passwords do not match")
                    else:
                        success, msg = register_user(new_user, new_pass, role)
                        if success:
                            st.success("✅ Registration Successful. Please login.")
                            st.session_state.auth_mode = "login"
                            # No st.rerun here to let user see success message
                        else:
                            st.error(f"❌ {msg}")
            
            if st.button("Back to Login"):
                st.session_state.auth_mode = "login"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ===============================
# MAIN DASHBOARD (ONLY RENDERED AFTER LOGIN)
# ===============================
# ===============================
# SIDEBAR
# ===============================
st.sidebar.image("https://img.icons8.com/neon/96/radar.png", width=80)
st.sidebar.title("🛰️ COMMAND")
st.sidebar.write(f"👤 **Operator:** {st.session_state.user}")
st.sidebar.write(f"🔑 **Role:** {st.session_state.role}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

animate = st.sidebar.checkbox("Enable Animation", True)
gain = st.sidebar.slider("Gain (dB)", 1, 40, 15)
distance = st.sidebar.slider("Distance (m)", 10, 1000, 200)
source = st.sidebar.radio("Signal Source", ["Simulated", "RTL-SDR"])

if source == "RTL-SDR":
    st.sidebar.warning("🎯 Target Type is ignored in RTL-SDR mode (using live data)")
    target = st.sidebar.selectbox("Target Type (Disabled)", LABELS, disabled=True, key="target_type_disabled")
else:
    target = st.sidebar.selectbox("Target Type", LABELS, key="target_type_active")

# ===============================
# DASHBOARD
# ===============================
st.markdown("## 📡 AI-Enabled Cognitive Photonic Radar")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Real-Time Analytics",
        "Explainable AI (XAI)",
        "Photonic Parameters",
        "Research Metrics",
        "System Logs",
        "Admin Panel",
    ]
)

# ===============================
# PRE-PROCESSING FOR ANALYTICS (Needed for XAI too)
# ===============================
target_low = target.lower()

if source == "RTL-SDR":
    from src.rtl_sdr_receiver import HAS_RTLSDR
    if not HAS_RTLSDR:
        st.warning("⚠️  RTL-SDR Library (librtlsdr) not found. Falling back to simulation.")
        signal = generate_radar_signal(target_low, distance)
    else:
        try:
            sdr = RTLRadar()
            signal = sdr.read_samples(4096)
            sdr.close()
        except Exception as e:
            st.error(f"SDR Hardware Error: {e}. Falling back to simulation.")
            signal = generate_radar_signal(target_low, distance)
else:
    signal = generate_radar_signal(target_low, distance)

    signal *= 10 ** (gain / 20)

# Run classical detection chain and only run AI if CFAR finds detections
detect_res = detect_targets_from_raw(signal, fs=4096, n_range=128, n_doppler=128, method='ca', guard=2, train=8, pfa=1e-6)
rd_map = detect_res['rd_map']
rd_map, spec, meta, photonic = get_all_features(signal)

detections = detect_res.get('detections', [])
ai_results = []
IMG_SIZE = 128
det_cfg = cfg.get('detection', {})
crop_size = int(det_cfg.get('crop_size', 32))

if len(detections) > 0:
    log_event(f"Processing {len(detections)} CFAR detections in AI pipeline", level="info")
    # ensure spectrogram aligns with rd_map dimensions for cropping
    try:
        spec_resized_full = cv2.resize(spec, (rd_map.shape[1], rd_map.shape[0]))
    except Exception as e:
        log_event(f"Spectrogram resize error: {e}. Using absolute value fallback.", level="warning")
        spec_resized_full = np.abs(spec)

    half = crop_size // 2
    for det in detections:
        i, j, val = det
        i = int(i); j = int(j)

        # pad rd_map and spec if crop goes out of bounds
        pad_y = max(0, half - i, (i + half) - rd_map.shape[0] + 1)
        pad_x = max(0, half - j, (j + half) - rd_map.shape[1] + 1)
        if pad_x > 0 or pad_y > 0:
            rd_pad = np.pad(rd_map, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
            spec_pad = np.pad(spec_resized_full, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
            i += pad_y
            j += pad_x
        else:
            rd_pad = rd_map
            spec_pad = spec_resized_full

        y1 = i - half; y2 = i + half
        x1 = j - half; x2 = j + half
        rd_crop = rd_pad[y1:y2, x1:x2]
        spec_crop = spec_pad[y1:y2, x1:x2]

        # Resize to model input
        rd_img = cv2.resize(rd_crop.astype(np.float32), (IMG_SIZE, IMG_SIZE))
        spec_img = cv2.resize(spec_crop.astype(np.float32), (IMG_SIZE, IMG_SIZE))

        # Normalize
        rd_norm_local = (rd_img - np.mean(rd_img)) / (np.std(rd_img) + 1e-8)
        spec_norm_local = (spec_img - np.mean(spec_img)) / (np.std(spec_img) + 1e-8)

        rd_t_local = torch.from_numpy(rd_norm_local).float().unsqueeze(0).unsqueeze(0).to(device)
        spec_t_local = torch.from_numpy(spec_norm_local).float().unsqueeze(0).unsqueeze(0).to(device)
        meta_t_local = torch.from_numpy(meta).float().unsqueeze(0).to(device)

        with torch.no_grad():
            out = radar_model(rd_t_local, spec_t_local, meta_t_local)
            ps = torch.softmax(out, dim=1)
            conf, idx = float(torch.max(ps)), int(torch.argmax(ps))
            label = LABELS[idx] if idx < len(LABELS) else 'Clutter'
            ai_results.append({"det": (i, j), "label": label, "confidence": conf, "value": val})

    # choose highest-confidence detection for top-level UI
    best = max(ai_results, key=lambda x: x['confidence']) if ai_results else None
    if best is not None:
        detected = best['label']
        confidence = best['confidence']
    else:
        detected = "Clutter"
        confidence = 0.0

else:
    log_event("No CFAR detections found; skipping AI inference", level="info")
    detected = "Clutter"
    confidence = 0.0

# ===== MULTI-TARGET TRACKING =====
if st.session_state.tracker_enabled:
    # Convert AI results to tracker detections: (range_idx, doppler_idx, value)
    tracker_detections = [(res['det'][0], res['det'][1], res['confidence']) for res in ai_results]
    
    # Update multi-target tracker
    active_tracks = st.session_state.tracker.update(tracker_detections)
    
    if active_tracks:
        log_event(f"Tracking: {len(active_tracks)} active targets", level="info")
    
    # Store track history for UI
    if 'track_history' not in st.session_state:
        st.session_state.track_history = []
    st.session_state.track_history.append({
        'time': time.time(),
        'tracks': active_tracks,
        'detected': detected,
        'confidence': confidence
    })
    # Keep last 100 updates
    st.session_state.track_history = st.session_state.track_history[-100:]
else:
    active_tracks = {}

# ===== ELECTRONIC WARFARE DEFENSE =====
if st.session_state.ew_enabled:
    ai_labels = [res['label'] for res in ai_results]
    ai_confidences = [res['confidence'] for res in ai_results]
    
    ew_result = st.session_state.ew_defense.analyze(
        signal=signal,
        detections=detections,
        ai_labels=ai_labels,
        ai_confidences=ai_confidences
    )
    
    if ew_result['ew_active']:
        log_event(f"EW ALERT: Threat level {ew_result['threat_level']}, {len(ew_result['threats'])} threats detected", level="warning")
        for threat in ew_result['threats']:
            log_event(f"  - {threat.threat_type}: conf={threat.confidence:.2f}, sev={threat.severity}", level="warning")
        for cm in ew_result['countermeasures']:
            log_event(f"  ⚡ Countermeasure: {cm.action_type} ({cm.reason})", level="info")
    
    # Filter detections: only keep real detections
    if ew_result['real_detections']:
        ai_results_filtered = [res for res, is_real in zip(ai_results, ew_result['real_detections']) if is_real]
    else:
        ai_results_filtered = ai_results
else:
    ew_result = {'threats': [], 'ew_active': False, 'threat_level': 'green', 'real_detections': None}
    ai_results_filtered = ai_results

# ===== COGNITIVE CONTROL ADAPTATION =====
if st.session_state.controller_enabled:
    # Compute confidence metrics (use filtered detections if EW active)
    det_results = ai_results_filtered if st.session_state.ew_enabled and ew_result['real_detections'] else ai_results
    avg_det_conf = np.mean([res['confidence'] for res in det_results]) if det_results else 0.0
    avg_trk_conf = np.mean([t['confidence'] for t in active_tracks.values()]) if active_tracks else 0.0
    num_tracks = len([t for t in active_tracks.values() if t['state'] == 'confirmed'])
    false_positives = len(detections) - len(det_results) if len(detections) > len(det_results) else 0
    
    # Observe state
    curr_state = st.session_state.cognitive_controller.observe(
        detection_confidence=avg_det_conf,
        tracking_confidence=avg_trk_conf,
        num_active_tracks=num_tracks,
        total_detections=len(detections),
        false_positives=false_positives,
        current_gain=gain,
        max_gain=40.0
    )
    
    # Learn from previous observation
    reward = st.session_state.cognitive_controller.learn(curr_state)
    
    # Decide next action (waveform parameters)
    cognitive_action = st.session_state.cognitive_controller.decide(curr_state)
    
    # Apply cognitive action if not in manual override
    if cognitive_action.is_adaptive:
            gain = cognitive_action.gain_db
            distance = cognitive_action.distance_m
            target = cognitive_action.target_type
            log_event(f"Cognitive adaptation: gain={gain:.1f}dB, dist={distance:.0f}m, target={target}", level="info")

rd_map_resized = cv2.resize(rd_map, (128, 128))
spec_resized = cv2.resize(spec, (128, 128))

rd_norm = rd_map_resized / (np.max(rd_map_resized) + 1e-8)
spec_norm = spec_resized / (np.max(spec_resized) + 1e-8)
rd_t = torch.from_numpy(rd_norm).float().unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 128, 128)
spec_t = torch.from_numpy(spec_norm).float().unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 128, 128)
meta_t = torch.from_numpy(meta).float().unsqueeze(0).to(device)
with torch.no_grad():
    output = radar_model(rd_t, spec_t, meta_t)
    probs = torch.softmax(output, dim=1)
    confidence = float(torch.max(probs))
    detected_idx = int(torch.argmax(probs))
    detected = LABELS[detected_idx] if detected_idx < len(LABELS) else "Clutter"

st.session_state.track_history = st.session_state.track_history[-50:]

# Cognitive threshold
thresh = adaptive_threshold(photonic['noise_power']) + st.session_state.get('sensitivity_offset', 0.0)
is_alert = confidence > thresh

# Kafka Streaming (Optional)
if "kafka_producer" not in st.session_state:
    st.session_state.kafka_producer = get_producer()

try:
    producer = st.session_state.kafka_producer
    if producer:
        producer.send("radar-stream", {
            "time": time.time(),
            "target": detected,
            "confidence": float(confidence),
            "distance": float(distance)
        })
except Exception as e:
    log_event(f"Kafka producer error: {e}", level="error")

# ===============================
# TAB 1: ANALYTICS
# ===============================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        # --- Interactive Plotly Range-Doppler Map ---
        fig_rd = go.Figure(data=go.Heatmap(
            z=10 * np.log10(rd_map + 1e-12),
            colorscale='Viridis',
            colorbar=dict(title='Power (dB)'),
            hovertemplate='Range: %{y}<br>Doppler: %{x}<br>Power: %{z:.1f} dB<extra></extra>'
        ))
        fig_rd.update_layout(
            title='Range-Doppler Map',
            xaxis_title='Doppler / Velocity (bins)',
            yaxis_title='Range (bins)',
            margin=dict(l=40, r=40, t=40, b=40),
            height=350
        )
        st.plotly_chart(fig_rd, use_container_width=True)

        # --- Interactive Plotly Micro-Doppler Spectrogram ---
        fig_spec = go.Figure(data=go.Heatmap(
            z=10 * np.log10(spec + 1e-12),
            colorscale='Magma',
            colorbar=dict(title='Power (dB)'),
            hovertemplate='Time: %{x}<br>Freq: %{y}<br>Power: %{z:.1f} dB<extra></extra>'
        ))
        fig_spec.update_layout(
            title='Micro-Doppler Spectrogram',
            xaxis_title='Time (bins)',
            yaxis_title='Frequency (Hz)',
            margin=dict(l=40, r=40, t=40, b=40),
            height=350
        )
        st.plotly_chart(fig_spec, use_container_width=True)

        # Tracking Plot (3D Upgrade)
        st.subheader("🎯 Target Tracking (Kalman Filter) — 3D View")
        if st.session_state.track_history and len(st.session_state.track_history) > 0:
            # Organize data by track_id
            tracks_by_id = {} # {track_id: {'x': [], 'y': [], 't': []}}
            
            # Start time for relative z-axis
            start_time = st.session_state.track_history[0]['time']

            for entry in st.session_state.track_history:
                t_rel = entry['time'] - start_time
                if 'tracks' in entry and entry['tracks']:
                    for track_id, track_data in entry['tracks'].items():
                        if 'position' in track_data:
                            if track_id not in tracks_by_id:
                                tracks_by_id[track_id] = {'x': [], 'y': [], 't': []}
                            
                            x, y = track_data['position']
                            tracks_by_id[track_id]['x'].append(x)
                            tracks_by_id[track_id]['y'].append(y)
                            tracks_by_id[track_id]['t'].append(t_rel)
            
            if tracks_by_id:
                fig3d = go.Figure()
                
                # Generate unique colors for different tracks
                colors = ['cyan', 'lime', 'magenta', 'yellow', 'orange', 'white']
                
                for i, (tid, data) in enumerate(tracks_by_id.items()):
                    color = colors[i % len(colors)]
                    # Plot full trajectory
                    fig3d.add_trace(go.Scatter3d(
                        x=data['x'], y=data['y'], z=data['t'],
                        mode='lines+markers',
                        name=f"Track {tid[:4]}",
                        line=dict(color=color, width=4),
                        marker=dict(size=3, color=color)
                    ))
                    
                    # Highlight latest point
                    fig3d.add_trace(go.Scatter3d(
                        x=[data['x'][-1]], y=[data['y'][-1]], z=[data['t'][-1]],
                        mode='markers',
                        name=f"Latest {tid[:4]}",
                        marker=dict(size=6, color='white', symbol='diamond'),
                        showlegend=False
                    ))

                fig3d.update_layout(
                    scene=dict(
                        xaxis_title='Range (bins)',
                        yaxis_title='Doppler (bins)',
                        zaxis_title='Time (s)',
                    ),
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=400,
                )
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.info("No active tracks in history.")
        else:
            st.info("No tracking history yet — generate data to populate 3D track view.")

    with col2:
        if is_alert:
            st.error(f"🚨 ALERT: {detected} Detected!")
        else:
            st.info("Searching for targets...")

        st.metric("Detected Target", detected)
        
        # Threat Level Gauge
        threat_color = "green"
        if PRIORITY[detected] == "High": threat_color = "orange"
        if PRIORITY[detected] == "Critical": threat_color = "red"

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"THREAT LEVEL: {PRIORITY[detected]}", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': threat_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(0, 255, 0, 0.3)'},
                    {'range': [50, 80], 'color': 'rgba(255, 255, 0, 0.3)'},
                    {'range': [80, 100], 'color': 'rgba(255, 0, 0, 0.3)'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': thresh * 100}
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.write(f"Confidence: {confidence * 100:.2f}%")
        st.write(f"Cognitive Threshold: {thresh:.2f}")
        st.write(f"Priority: **{PRIORITY[detected]}**")

        st.markdown("---")
        st.subheader("Phase Statistics")
        if len(meta) >= 3:
            st.write(f"Mean Phase: {meta[0]:.4f}")
            st.write(f"Variance: {meta[1]:.4f}")
            st.write(f"Coherence: {meta[2]:.4f}")
        else:
            st.write("Metadata incomplete")

    st.markdown("### 🎬 Demo Scenarios")
    demo_col1, demo_col2 = st.columns([3, 1])
    scenario_choice = demo_col1.selectbox(
        "Select Scenario",
        ["Drone Approach", "Jamming Attack", "Multi-Target"],
        key="demo_scenario_choice",
    )
    if demo_col2.button("Run Demo", key="run_demo_button"):
        if scenario_choice == "Drone Approach":
            st.session_state.demo_timeline = run_drone_approach_demo()
        elif scenario_choice == "Jamming Attack":
            st.session_state.demo_timeline = run_jamming_attack_demo()
        else:
            st.session_state.demo_timeline = run_multi_target_demo()
        st.success(f"{scenario_choice} demo generated")

    timeline_payload = st.session_state.get("demo_timeline")
    if timeline_payload and timeline_payload.get("timeline"):
        timeline = timeline_payload["timeline"]
        table_rows = []
        for entry in timeline:
            notes = []
            if "distance_m" in entry:
                notes.append(f"distance={entry['distance_m']:.0f}m")
            if "jammer_level" in entry:
                notes.append(f"jammer={entry['jammer_level']:.2f}")
            if "targets" in entry:
                notes.append(f"targets={len(entry['targets'])}")
            table_rows.append({
                "Step": entry.get("step"),
                "Description": entry.get("description"),
                "Detections": entry.get("num_detections"),
                "Tracks": len(entry.get("tracks", {})),
                "Signal Power": entry.get("signal_power"),
                "Notes": ", ".join(notes),
            })

        st.dataframe(pd.DataFrame(table_rows))
        with st.expander("Latest Step Detail", expanded=False):
            st.json(timeline[-1])
    else:
        st.info("Run a demo scenario to generate timeline data.")

# ===============================
# TAB 2: XAI
# ===============================
with tab2:
    st.subheader("Explainable AI: Grad-CAM Visualizations")
    st.write("Visualizing which parts of the input maps influenced the AI's decision.")
    
    col_xai1, col_xai2 = st.columns(2)
    
    # We need to enable gradients for Grad-CAM
    cam_rd = grad_cam_pytorch(radar_model, rd_t, spec_t, meta_t, radar_model.rd_branch.conv2)
    cam_spec = grad_cam_pytorch(radar_model, rd_t, spec_t, meta_t, radar_model.spec_branch.conv2)

    with col_xai1:
        st.write("**RD Map Heatmap**")
        if cam_rd.any():
            fig_rd_xai, ax_rd = plt.subplots()
            ax_rd.imshow(rd_norm, cmap='gray')
            ax_rd.imshow(cam_rd, cmap='jet', alpha=0.5)
            st.pyplot(fig_rd_xai)
            plt.close(fig_rd_xai)
        else:
            st.warning("Grad-CAM unavailable for RD Map")

    with col_xai2:
        st.write("**Spectrogram Heatmap**")
        if cam_spec.any():
            fig_sp_xai, ax_sp = plt.subplots()
            ax_sp.imshow(spec_norm, cmap='gray')
            ax_sp.imshow(cam_spec, cmap='jet', alpha=0.5)
            st.pyplot(fig_sp_xai)
            plt.close(fig_sp_xai)
        else:
            st.warning("Grad-CAM unavailable for Spectrogram")

# ===============================
# TAB 3: PHOTONIC PARAMETERS
# ===============================
with tab3:
    st.subheader("Photonic Radar Parameters")

    st.write(f"Bandwidth: {photonic['instantaneous_bandwidth']/1e6:.2f} MHz")
    st.write(f"Chirp Slope: {photonic['chirp_slope']/1e12:.2f} THz/s")
    st.write(f"Pulse Width: {photonic['pulse_width']*1e6:.2f} μs")
    st.write(f"Noise Power: {photonic['noise_power']:.6f}")
    st.write(f"Clutter Power: {photonic['clutter_power']:.6f}")

    st.markdown("#### TTD Beamforming Vector")
    fig_beam = go.Figure(data=go.Scatter(
        y=photonic["ttd_vector"],
        mode='lines+markers',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=6, color='#60a5fa'),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    fig_beam.update_layout(
        title="True Time Delay Profile",
        xaxis_title="Antenna Element Index",
        yaxis_title="Delay (ns)",
        margin=dict(l=40, r=40, t=40, b=40),
        height=300
    )
    st.plotly_chart(fig_beam, use_container_width=True)

# ===============================
# TAB 4: RESEARCH METRICS
# ===============================
with tab4:
    st.subheader("Research Metrics")
    metrics_data, metrics_error = load_metrics_report(METRICS_JSON_PATH)

    if metrics_error:
        st.warning(metrics_error)
    elif metrics_data is not None:
        metadata = metrics_data.get("metadata", {})
        macro_avg = metrics_data.get("macro_avg", {})
        weighted_avg = metrics_data.get("weighted_avg", {})

        def _fmt(value):
            if isinstance(value, (int, float)):
                return f"{value:.3f}"
            return value if value is not None else "—"

        summary_rows = [
            {"Metric": "Model", "Value": metadata.get("model_name", "—")},
            {"Metric": "Timestamp", "Value": metadata.get("timestamp", "—")},
            {"Metric": "Samples", "Value": metadata.get("n_samples", "—")},
            {"Metric": "Classes", "Value": metadata.get("n_classes", "—")},
            {"Metric": "Accuracy", "Value": _fmt(metrics_data.get("accuracy"))},
            {"Metric": "Macro Precision", "Value": _fmt(macro_avg.get("precision"))},
            {"Metric": "Macro Recall", "Value": _fmt(macro_avg.get("recall"))},
            {"Metric": "Macro F1", "Value": _fmt(macro_avg.get("f1"))},
            {"Metric": "Weighted Precision", "Value": _fmt(weighted_avg.get("precision"))},
            {"Metric": "Weighted Recall", "Value": _fmt(weighted_avg.get("recall"))},
            {"Metric": "Weighted F1", "Value": _fmt(weighted_avg.get("f1"))},
        ]

        st.markdown("### Experiment Summary")
        st.table(pd.DataFrame(summary_rows))

        report = metrics_data.get("classification_report", {})
        per_class_entries = {k: v for k, v in report.items() if isinstance(v, dict)}
        if per_class_entries:
            st.markdown("### Classification Report")
            report_df = pd.DataFrame(per_class_entries).transpose()
            st.dataframe(report_df)
    else:
        st.info("No metrics data available yet.")

    st.markdown("### Performance Visualizations")
    col_rm1, col_rm2 = st.columns(2)
    with col_rm1:
        display_metric_image("Confusion Matrix", METRIC_IMAGE_PATHS["Confusion Matrix"])
    with col_rm2:
        display_metric_image("ROC Curve", METRIC_IMAGE_PATHS["ROC Curve"])

    col_rm3, col_rm4 = st.columns(2)
    with col_rm3:
        display_metric_image("Precision-Recall Curve", METRIC_IMAGE_PATHS["Precision-Recall Curve"])
    with col_rm4:
        display_metric_image("Training Curves", METRIC_IMAGE_PATHS["Training Curves"])

# ===============================
# TAB 5: LOGS
# ===============================
with tab5:
    st.subheader("Detection History")

    # Only add to history if something interesting happened or periodically
    # To avoid infinite growth during animation, we limit how often we log.
    if "last_log_time" not in st.session_state:
        st.session_state.last_log_time = 0
    
    current_time = time.time()
    if current_time - st.session_state.last_log_time > 2.0: # Log every 2 seconds
        entry = {
            "Time": time.strftime("%H:%M:%S"),
            "Target": detected,
            "Confidence": f"{confidence*100:.1f}%",
            "Priority": PRIORITY[detected]
        }
        st.session_state.history.append(entry)
        st.session_state.last_log_time = current_time
        
        if is_alert:
            log_event(f"ALERT: {detected} detected with {confidence*100:.1f}% confidence")

    df = pd.DataFrame(st.session_state.history[-20:])
    st.table(df)

    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Threat Report (CSV)",
            data=csv,
            file_name=f"radar_report_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
        )

    st.markdown("#### System Logs")
    st.code("".join(read_logs(20)))

# ===============================
# TAB 6: ADMIN PANEL
# ===============================
with tab6:
    if st.session_state.role != "admin":
        st.warning("⚠️ Access Denied: Admin privileges required.")
    else:
        st.subheader("🛠️ Administrative Controls")
        
        admin_subtab1, admin_subtab2, admin_subtab3 = st.tabs(["User Management", "System Health", "Advanced Config"])
        
        with admin_subtab1:
            from src.user_manager import list_users, create_user, delete_user, update_user_role
            
            st.markdown("### User Management")
            users = list_users()
            user_df = pd.DataFrame(users, columns=["Username", "Role"])
            st.table(user_df)
            
            with st.expander("Add New User"):
                with st.form("add_user"):
                    new_user = st.text_input("Username")
                    new_pass = st.text_input("Password", type="password")
                    new_role = st.selectbox("Role", ["viewer", "operator", "admin"])
                    if st.form_submit_button("Create User"):
                        if new_user and new_pass:
                            create_user(new_user, new_pass, new_role)
                            st.success(f"User {new_user} created!")
                            st.rerun()
                        else:
                            st.error("Fields cannot be empty")
            
            with st.expander("Delete/Update User"):
                target_user = st.selectbox("Select User", [u[0] for u in users if u[0] != st.session_state.user])
                col_del, col_upd = st.columns(2)
                with col_del:
                    if st.button("Delete User", type="primary"):
                        delete_user(target_user)
                        st.success(f"User {target_user} deleted")
                        st.rerun()
                with col_upd:
                    new_role_val = st.selectbox("New Role", ["viewer", "operator", "admin"], key="new_role_val")
                    if st.button("Update Role"):
                        update_user_role(target_user, new_role_val)
                        st.success(f"Role updated for {target_user}")
                        st.rerun()

        with admin_subtab2:
            st.markdown("### System Health")
            if HAS_PSUTIL:
                cpu_usage = psutil.cpu_percent()
                mem_usage = psutil.virtual_memory().percent
            else:
                cpu_usage = "N/A"
                mem_usage = "N/A"
            
            col_h1, col_h2, col_h3 = st.columns(3)
            col_h1.metric("CPU Usage", f"{cpu_usage}%" if HAS_PSUTIL else "N/A")
            col_h2.metric("Memory Usage", f"{mem_usage}%" if HAS_PSUTIL else "N/A")
            col_h3.metric("DB Status", "Connected" if os.path.exists("results/users.db") else "Error")
            
            st.markdown("#### Hardware Status")
            from src.rtl_sdr_receiver import HAS_RTLSDR
            st.write(f"RTL-SDR Driver: {'✅ Detected' if HAS_RTLSDR else '❌ Missing'}")
            
            from src.stream_bus import HAS_KAFKA
            st.write(f"Kafka Integration: {'✅ Active' if HAS_KAFKA else '⚠️ Disabled (Library Missing)'}")

        with admin_subtab3:
            st.markdown("### Advanced Configuration")
            st.info("System-wide parameters (Require restart to take full effect)")
            new_threshold = st.slider("Global Sensitivity Offset", -0.2, 0.2, 0.0)
            st.session_state.sensitivity_offset = new_threshold
            
            if st.button("Clear System Logs", type="secondary"):
                with open("results/system.log", "w") as f:
                    f.write("")
                st.success("Logs cleared")
                st.rerun()

# ===============================
# SAFE AUTO-REFRESH
# ===============================
if animate:
    time.sleep(0.5)
    # Streamlit recommends using fragment for local updates or 
    # being careful with rerun in loops.
    # To prevent rapid-fire reruns that can cause "script already running" errors,
    # we ensure a minimum delay.
    st.rerun()
