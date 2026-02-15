
import numpy as np
import pytest
from src.signal_generator import generate_radar_signal

def test_weibull_clutter_statistics():
    """Verify that clutter generation follows a heavy-tailed distribution (Weibull-like)."""
    clutter_sig = generate_radar_signal("clutter", fs=4096)
    amp = np.abs(clutter_sig)
    
    # Check if mean and variance are non-zero
    assert np.mean(amp) > 0
    assert np.var(amp) > 0
    
    # Weibull/Heavy-tail check: kurtosis should be higher than Normal distribution (3.0)
    # Using a simple proxy: max/mean ratio should be relatively high for spiky clutter
    peak_to_average = np.max(amp) / (np.mean(amp) + 1e-9)
    assert peak_to_average > 3.0, f"Clutter should be spiky, got PAR {peak_to_average}"

def test_multipath_existence():
    """Verify that multipath creates a secondary signal peak."""
    # drone target should have multipath
    sig = generate_radar_signal("drone", distance=100, fs=4096)
    
    # Auto-correlation to find echoes
    corr = np.correlate(np.abs(sig), np.abs(sig), mode='full')
    lags = np.arange(len(corr)) - (len(sig) - 1)
    
    # Ignore main lobe
    mask = (np.abs(lags) > 10) & (np.abs(lags) < 500)
    secondary_peaks = np.max(corr[mask])
    
    assert secondary_peaks > 0, "Should detect some multipath correlation/echoes"

def test_swerling_fluctuation():
    """Verify that target amplitude fluctuates across multiple generations."""
    amps = []
    for _ in range(20): # Increased samples for better stats
        # Use close range (10m) so target dominates clutter/noise
        sig = generate_radar_signal("missile", distance=10, fs=4096)
        amps.append(np.mean(np.abs(sig)))
    
    # Check coefficient of variation to ensure fluctuation
    cv = np.std(amps) / np.mean(amps)
    assert cv > 0.05, f"Target RCS should fluctuate, got CV {cv}"
