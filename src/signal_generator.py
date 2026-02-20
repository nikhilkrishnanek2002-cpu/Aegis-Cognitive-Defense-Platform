# ===== src/signal_generator.py =====
import numpy as np
from scipy.signal import chirp

from .config import get_config
from .photonic_signal_model import generate_photonic_rf

USE_RTL_SDR = False  # 🔴 SET TRUE when hardware connected

if USE_RTL_SDR:
    from src.rtl_sdr_receiver import RTLRadar
    rtl = RTLRadar()


def generate_radar_signal(target_type, distance=100, fs=4096):
    cfg = get_config()
    photonic_cfg = cfg.get("photonic_model", {})
    use_photonic = photonic_cfg.get("enabled", False)

    # ---- Common Target Physics (RCS & fluctuation) ----
    # Distance attenuation path loss (1/R^4 for radar)
    dist_jitter = distance * np.random.uniform(0.98, 1.02)
    loss_factor = min(1e11 / (dist_jitter ** 4 + 1e-12), 10.0)
    attenuation = np.sqrt(loss_factor)

    # Calculate RCS fluctuation based on target type
    if target_type == "drone":
        rcs_fluctuation = np.random.gamma(2, 2)
    elif target_type == "aircraft":
        rcs_fluctuation = np.random.exponential(1)
    elif target_type == "missile":
        rcs_fluctuation = np.random.exponential(1)
    elif target_type == "helicopter":
        rcs_fluctuation = np.random.lognormal(0, 0.5)
    elif target_type == "bird":
        rcs_fluctuation = np.random.rayleigh(1)
    elif target_type == "clutter":
        rcs_fluctuation = 0.0
        attenuation = 0.0
    else:
        rcs_fluctuation = 1.0

    # If photonic model enabled, generate base signal from it
    if use_photonic:
        seed = photonic_cfg.get("seed", None)
        t, signals = generate_photonic_rf(duration=1.0, fs=fs, num_channels=1, seed=seed)
        # Photonic signal is the "Transmitted" signal. 
        # We must apply target physics (reflection coeff + attenuation) to get "Received" signal.
        # We assume the photonic generator creates a unit-amplitude carrier.
        
        raw_sig = signals[0].astype(np.complex64) + 0j
        
        # Apply target physics
        if target_type != "clutter":
            base_signal = raw_sig * attenuation * rcs_fluctuation
        else:
            base_signal = np.zeros_like(raw_sig)

    else:
        # ---- Advanced Synthetic Generation ----
        t = np.linspace(0, 1, fs)

        # Randomize chirp parameters slightly for each generation
        f0_offset = np.random.uniform(-50, 50)
        f1_offset = np.random.uniform(-100, 100)
        
        # Base Signal Generation (Modulation depends on target)
        if target_type == "drone":
            base_sig = chirp(t, 100+f0_offset, 1, 200+f1_offset) 
            micro_doppler = 0.2 * np.sin(2 * np.pi * 50 * t) 
        elif target_type == "aircraft":
            base_sig = chirp(t, 300+f0_offset, 1, 500+f1_offset)
            micro_doppler = 0.05 * np.sin(2 * np.pi * 10 * t)
        elif target_type == "missile":
            base_sig = chirp(t, 800+f0_offset, 1, 1500+f1_offset)
            micro_doppler = 0.0 
        elif target_type == "helicopter":
            base_sig = chirp(t, 200+f0_offset, 1, 300+f1_offset) 
            micro_doppler = 0.8 * np.sin(2 * np.pi * 120 * t)
        elif target_type == "bird":
            base_sig = chirp(t, 50+f0_offset, 1, 80+f1_offset) 
            micro_doppler = 0.3 * np.sin(2 * np.pi * 2 * t)
        elif target_type == "clutter":
            base_sig = 0.0
            micro_doppler = 0.0
        else:
            base_sig = np.random.normal(0, 0.1, len(t))
            micro_doppler = 0

        if target_type != "clutter":
            # Combine Base + Micro-Doppler + Physics
            target_signal = (base_sig + micro_doppler) * attenuation * rcs_fluctuation
            base_signal = target_signal * np.exp(1j * np.pi / 4) # IQ conversion
        else:
            base_signal = np.zeros(len(t), dtype=np.complex64)
        
    # ==========================================
    # COMMON ENVIRONMENTAL & CHANNEL EFFECTS
    # ==========================================
    # Ensure t is correct length if photonic used different implicit fs (assuming fs is strict)
    if len(base_signal) != len(t):
        t = np.linspace(0, 1, len(base_signal))
    
    # 1. Weaver-generate Clutter (Weibull Distribution)
    # Shape parameter k < 2 gives heavy tails (spiky clutter)
    k_weibull = 0.8 
    scale_weibull = 0.15
    clutter_amp = scale_weibull * np.random.weibull(k_weibull, size=len(t))
    clutter_phase = np.random.uniform(0, 2*np.pi, len(t))
    clutter = clutter_amp * np.exp(1j * clutter_phase)

    # 2. Multipath Interference (Ground bounce)
    # Delayed and attenuated copy of the signal
    multipath_delay = int(fs * 0.05) # 50ms delay
    multipath_coef = 0.3 # Reflection coefficient
    
    multipath_sig = np.zeros_like(base_signal)
    if multipath_delay < len(t):
        multipath_sig[multipath_delay:] = base_signal[:-multipath_delay] * multipath_coef
    
    # 3. Thermal Noise
    thermal_noise = (np.random.normal(0, 0.01, len(t)) + 1j*np.random.normal(0, 0.01, len(t)))
    
    # 4. Total Signal Composition
    total_signal = base_signal + multipath_sig + clutter + thermal_noise
    
    return total_signal.astype(np.complex64)
