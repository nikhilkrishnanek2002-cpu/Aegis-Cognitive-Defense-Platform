"""Convenience interface for running adversarial radar signal evaluations."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .adversarial_attacks import AdversarialRadarAttacks


def _suite(logger=None) -> AdversarialRadarAttacks:
    return AdversarialRadarAttacks(logger=logger)


def gaussian_noise_injection(signal: np.ndarray, snr_db: float, logger=None) -> np.ndarray:
    """Apply AWGN at the specified SNR."""
    return _suite(logger).add_gaussian_noise(signal, snr_db)


def frequency_shift(signal: np.ndarray, shift_hz: float, sample_rate: float = 1e6, logger=None) -> np.ndarray:
    """Apply a Doppler-like frequency translation attack."""
    return _suite(logger).frequency_shift_attack(signal, shift_hz, sample_rate=sample_rate)


def amplitude_scaling(signal: np.ndarray, scale: float, logger=None) -> np.ndarray:
    """Scale the amplitude/power envelope of the radar return."""
    return _suite(logger).amplitude_scaling_attack(signal, scale)


def replay_delay(signal: np.ndarray, delay_samples: int, logger=None) -> np.ndarray:
    """Inject a delayed replay copy to emulate multipath or spoofing."""
    return _suite(logger).replay_attack(signal, delay_samples)


def spoof_target(signal: np.ndarray, fake_range: float, fake_velocity: float, *, carrier_freq: float = 10e9,
                 sample_rate: float = 1e6, logger=None) -> np.ndarray:
    """Inject a synthetic target signature at the requested range/velocity."""
    return _suite(logger).spoof_target_attack(signal, fake_range, fake_velocity,
                                             carrier_freq=carrier_freq, sample_rate=sample_rate)


def run_attack_suite(model: torch.nn.Module,
                     signals: np.ndarray,
                     labels: np.ndarray,
                     attack_params: Optional[Dict] = None,
                     *,
                     device: str = "cpu",
                     batch_size: int = 32,
                     logger=None) -> Dict:
    """Execute the comprehensive adversarial evaluation suite and return a structured report."""
    return _suite(logger).run_attack_suite(
        model=model,
        signals=signals,
        labels=labels,
        attack_params=attack_params,
        device=device,
        batch_size=batch_size,
    )
